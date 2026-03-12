import torch
from typing import Any, Dict, Union, List, Tuple, Optional
from transformers import GenerationConfig
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_truncate, nested_concat, nested_numpify, find_batch_size, IterableDatasetShard
from transformers.trainer_utils import has_length, EvalLoopOutput, denumpify_detensorize
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.integrations.deepspeed import deepspeed_init

from uie_collator import SUPPORTED_DECODER_MODELS, check_model
from uie_dataset_lora import ANSWER_PREFIX


def skip_instructions(model, predictions_ids, tokenizer, ignore_idx=-100):
    predictions_ids = np.where(predictions_ids == ignore_idx, tokenizer.pad_token_id, predictions_ids)

    predictions = tokenizer.batch_decode(
        predictions_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    final_predictions = []
    if check_model(model.config._name_or_path, SUPPORTED_DECODER_MODELS):
        for pred in predictions:

            if ANSWER_PREFIX in pred:
                splits = pred.split(ANSWER_PREFIX)
                final_predictions.append(splits[-1].strip())
            else:
                final_predictions.append('')
    else:
        final_predictions = predictions

    return final_predictions


# Keys allowed to be logged to wandb when WandbFilterCallback is used (step, loss, task loss, orthogonal, l2, grad norms).
WANDB_ALLOWED_KEYS = frozenset({
    "loss", "learning_rate", "epoch", "global_step",
    "task_loss", "orthogonal_loss", "l2_loss",
    "ella_loss", "ella_loss_no_drop", "ella_loss_after_drop", "ella_loss_no_drop_full", "ella_loss_after_drop_full",
    "ella_loss_full", "grad_norm_ce", "grad_norm_ella",
    "train_loss", "train_task_loss", "train_orthogonal_loss", "train_ella_loss", "train_l2_loss",
    "train_grad_norm_ce", "train_grad_norm_ella",
    "train_runtime", "train_samples_per_second", "train_steps_per_second",
})

# Predict metrics to send to wandb (empty = don't log rouge/exact_match etc. to wandb; add keys if you want some).
WANDB_PREDICT_ALLOWED_KEYS = frozenset({"predict_samples"})


def _to_scalar(v):
    """Convert to Python float/int for wandb; accept numpy/tensor scalars."""
    if isinstance(v, (int, float)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _log_grad_norms_to_wandb(trainer, logs: dict) -> None:
    """Log only grad_norm_ce and grad_norm_ella to wandb immediately. Call when they were just added to logs."""
    report_to = getattr(trainer.args, "report_to", None)
    if not report_to or ("wandb" not in (report_to if isinstance(report_to, (list, tuple)) else [report_to])):
        return
    if not trainer.is_world_process_zero():
        return
    to_log = {}
    for key in ("grad_norm_ce", "grad_norm_ella"):
        if key in logs:
            v = _to_scalar(logs[key])
            if v is not None:
                to_log[f"train/{key}"] = v
    if not to_log:
        return
    try:
        import wandb
        import os
        if wandb.run is None:
            # Ensure a run exists (e.g. when logging before WandbCallback has inited)
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "CL"),
                name=getattr(trainer.args, "run_name", None),
            )
        step = trainer.state.global_step
        wandb.log(to_log, step=step)
    except Exception:
        pass


def log_cl_metrics_to_wandb(trainer, logs: dict) -> None:
    """Explicitly log CL metrics (ella_loss, l2_loss, grad_norm_ce, grad_norm_ella) to wandb.
    Call this from ELLA trainers' log() after adding metrics to logs, so they appear even if
    callback order or filtering would otherwise drop them."""
    report_to = getattr(trainer.args, "report_to", None)
    if not report_to or ("wandb" not in (report_to if isinstance(report_to, (list, tuple)) else [report_to])):
        return
    try:
        import wandb
        if wandb.run is None:
            return
    except Exception:
        return
    if not trainer.is_world_process_zero():
        return
    step = trainer.state.global_step
    to_log = {}
    for k, v in logs.items():
        if k not in WANDB_ALLOWED_KEYS:
            continue
        scalar = _to_scalar(v)
        if scalar is not None:
            to_log[f"train/{k}"] = scalar
    if to_log:
        to_log["train/global_step"] = step
        wandb.log(to_log, step=step)


class WandbFilterCallback(TrainerCallback):
    """Keep only step/loss/ella_loss/l2_loss in logs so wandb does not get rouge, accuracy, or per-subtask metrics."""

    def __init__(self, allowed_keys=None):
        self.allowed_keys = allowed_keys or WANDB_ALLOWED_KEYS

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        filtered = {k: v for k, v in logs.items() if k in self.allowed_keys}
        logs.clear()
        logs.update(filtered)


class DenserEvalCallback(TrainerCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        log_eval_steps = [1, 50, 100, 200]

        # Log
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True

        # Save
        # if args.save_strategy

        return control


class UIETrainer(Seq2SeqTrainer):

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch: int = None) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            num_items_in_batch (`int`, *optional*):
                Number of items in the batch.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        ########################### Regularization ##########################
        orthogonal_loss = 0.
        for name, param in self.model.named_parameters():
            if "lora_A" in name:
                for name_, param_ in self.model.named_parameters():
                    if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
                        orthogonal_loss += torch.abs(torch.mm(param, param_.T)).sum() # [r * dim] * [dim * r]
                        break # target modules have been matched

        # l2-normalization for loranew_A/B
        l2_loss = 0.
        for name, param in self.model.named_parameters():
            if "loranew_" in name:
                l2_loss += torch.norm(param, p=2)

        lamda_1 = self.args.lamda_1
        lamda_2 = self.args.lamda_2

        # Scalars for logging (same dict goes to console and wandb when report_to includes "wandb")
        task_loss_val = loss.item()
        orth_val = orthogonal_loss.item()
        l2_val = l2_loss.item()
        if not hasattr(self, "_orth_l2_sum"):
            self._orth_l2_sum = {"task_loss": 0.0, "orthogonal_loss": 0.0, "l2_loss": 0.0}
            self._orth_l2_n = 0
        self._orth_l2_sum["task_loss"] += task_loss_val
        self._orth_l2_sum["orthogonal_loss"] += orth_val
        self._orth_l2_sum["l2_loss"] += l2_val
        self._orth_l2_n += 1

        loss = loss + orthogonal_loss * lamda_1 + l2_loss * lamda_2
        ######################################################################

        # Store loss before backward since accelerator.backward() returns None
        loss_detached = loss.detach()
        self.accelerator.backward(loss)

        return loss_detached

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        """Inject task_loss, orthogonal_loss and l2_loss (avg over last logging interval) so they appear in console and wandb."""
        if getattr(self, "args", None) and getattr(self.args, "log_cl_metrics", True) and getattr(self, "_orth_l2_n", 0) > 0:
            n = self._orth_l2_n
            logs["task_loss"] = round(self._orth_l2_sum["task_loss"] / n, 6)
            logs["orthogonal_loss"] = round(self._orth_l2_sum["orthogonal_loss"] / n, 6)
            logs["l2_loss"] = round(self._orth_l2_sum["l2_loss"] / n, 6)
            self._orth_l2_sum = {"task_loss": 0.0, "orthogonal_loss": 0.0, "l2_loss": 0.0}
            self._orth_l2_n = 0
        super().log(logs, start_time)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here (only needed for ZeRO Stage 3)
        if self.is_deepspeed_enabled and self.deepspeed is None and is_deepspeed_zero3_enabled():
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            # For ZeRO Stage 3, use full DeepSpeed prepare; for Stage 2 or non-DeepSpeed, use eval mode
            if self.is_deepspeed_enabled and is_deepspeed_zero3_enabled():
                model = self.accelerator.prepare(model)
            else:
                model = self.accelerator.prepare_model(model, evaluation_mode=True)

            if model is not self.model:
                self.model_wrapped = model

            if self.is_deepspeed_enabled and is_deepspeed_zero3_enabled():
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(dataset=eval_dataset, preds=all_preds, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        metrics["global_step"] = self.state.global_step

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        synced_gpus = True if is_deepspeed_zero3_enabled() else False

        attention_mask = inputs.get("attention_mask", None)

        # Remove keys that should not go into GenerationConfig
        gen_kwargs.pop("synced_gpus", None)
        gen_kwargs.pop("attention_mask", None)

        generation_config = GenerationConfig(**gen_kwargs)

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            input_ids=generation_inputs, 
            generation_config=generation_config,
            attention_mask=attention_mask,
            synced_gpus=synced_gpus,
        )

        bs, source_len = inputs['input_ids'].shape
        # in case the batch is shorter than max length, the output should be padded
        if check_model(self.model.config._name_or_path, SUPPORTED_DECODER_MODELS):
            max_length = source_len + gen_kwargs["max_new_tokens"]
        else:
            max_length = gen_kwargs["max_new_tokens"]

        if generated_tokens.shape[-1] < max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, max_length)

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_new_tokens"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_new_tokens"])
        else:
            labels = None

        return (loss, generated_tokens, labels)
