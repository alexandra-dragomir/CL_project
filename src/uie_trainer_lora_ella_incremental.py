"""
ELLA with incremental (layer-wise) lambda: different lambda_1 for first/middle/last third of transformer blocks.

Use when --lamda_1_list "early,mid,late" is set (e.g. "0.1,1.0,10.0").
Only lambda_1 is varied; lambda_2 remains a single value.
"""

import re
import torch
from typing import Any, Dict, Union, List, Tuple, Optional
from transformers.trainer import *

from uie_trainer_lora_ella import UIETrainerELLA


def _prefix_to_block_index(prefix: str) -> Optional[Tuple[str, int]]:
    """Return ('encoder', block_i) or ('decoder', block_j) or None if not a block LoRA.
    Matches .encoder.block.X. or .decoder.block.Y. in the parameter path.
    """
    enc = re.search(r"\.encoder\.block\.(\d+)\.", prefix)
    dec = re.search(r"\.decoder\.block\.(\d+)\.", prefix)
    if enc is not None:
        return ("encoder", int(enc.group(1)))
    if dec is not None:
        return ("decoder", int(dec.group(1)))
    return None


class UIETrainerELLAIncremental(UIETrainerELLA):
    """
    ELLA with incremental lambda: split transformer blocks into 3 segments (first/middle/last third)
    and apply lamda_1_1, lamda_1_2, lamda_1_3 respectively. Requires args.lamda_1_list = "v1,v2,v3".
    """

    def _get_prefix_block_segment_map(self, prefixes: List[str]) -> Tuple[Optional[Dict[str, int]], int]:
        """Return (prefix -> segment 0/1/2, total_blocks). Segment 0=first third, 1=middle, 2=last."""
        block_indices = {}
        max_enc, max_dec = -1, -1
        for p in prefixes:
            out = _prefix_to_block_index(p)
            if out is None:
                continue
            enc_dec, idx = out
            block_indices[p] = (enc_dec, idx)
            if enc_dec == "encoder":
                max_enc = max(max_enc, idx)
            else:
                max_dec = max(max_dec, idx)
        if not block_indices:
            return None, 0
        num_enc = max_enc + 1
        num_dec = max_dec + 1
        total_blocks = num_enc + num_dec
        prefix_to_segment = {}
        for p, (enc_dec, idx) in block_indices.items():
            global_idx = idx if enc_dec == "encoder" else num_enc + idx
            segment = min(2, global_idx * 3 // total_blocks) if total_blocks else 0
            prefix_to_segment[p] = segment
        return prefix_to_segment, total_blocks

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch: int = None) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps

        ########################### ELLA with incremental lambda ##########################
        ella_loss = 0.0

        module_params = {}
        for name, param in self.model.named_parameters():
            if "loranew_A" in name:
                prefix = name.split("loranew_A")[0]
                if prefix not in module_params:
                    module_params[prefix] = {}
                module_params[prefix]["loranew_A"] = param
            elif "loranew_B" in name:
                prefix = name.split("loranew_B")[0]
                if prefix not in module_params:
                    module_params[prefix] = {}
                module_params[prefix]["loranew_B"] = param
            elif "lora_A" in name:
                prefix = name.split("lora_A")[0]
                if prefix not in module_params:
                    module_params[prefix] = {}
                module_params[prefix]["lora_A"] = param
            elif "lora_B" in name:
                prefix = name.split("lora_B")[0]
                if prefix not in module_params:
                    module_params[prefix] = {}
                module_params[prefix]["lora_B"] = param

        lamda_1_list_raw = getattr(self.args, "lamda_1_list", None)
        prefix_to_segment = None
        lamda_1_by_segment = None
        if lamda_1_list_raw:
            try:
                parts = [s.strip() for s in lamda_1_list_raw.split(",")]
                if len(parts) == 3:
                    lamda_1_by_segment = [float(x) for x in parts]
                    prefix_to_segment, _ = self._get_prefix_block_segment_map(list(module_params.keys()))
                    if prefix_to_segment is None:
                        lamda_1_by_segment = None
            except (ValueError, TypeError):
                prefix_to_segment = None
                lamda_1_by_segment = None

        for prefix, params in module_params.items():
            if all(k in params for k in ["lora_A", "lora_B", "loranew_A", "loranew_B"]):
                W_past = torch.mm(params["lora_B"], params["lora_A"])
                Delta_W_t = torch.mm(params["loranew_B"], params["loranew_A"])
                module_ella = (Delta_W_t * W_past).pow(2).sum()
                if prefix_to_segment is not None and lamda_1_by_segment is not None and prefix in prefix_to_segment:
                    seg = prefix_to_segment[prefix]
                    ella_loss = ella_loss + lamda_1_by_segment[seg] * module_ella
                else:
                    ella_loss = ella_loss + module_ella

        l2_loss = 0.0
        for name, param in self.model.named_parameters():
            if "loranew_" in name:
                l2_loss += torch.norm(param, p=2)

        lamda_2 = self.args.lamda_2
        if prefix_to_segment is None or lamda_1_by_segment is None:
            lamda_1 = self.args.lamda_1
            loss = loss + ella_loss * lamda_1 + l2_loss * lamda_2
        else:
            loss = loss + ella_loss + l2_loss * lamda_2

        ella_loss_val = ella_loss.item() if isinstance(ella_loss, torch.Tensor) else float(ella_loss)
        l2_loss_val = l2_loss.item() if isinstance(l2_loss, torch.Tensor) else float(l2_loss)
        if not hasattr(self, "_ella_l2_sum"):
            self._ella_l2_sum = {"ella_loss": 0.0, "l2_loss": 0.0}
            self._ella_l2_n = 0
        self._ella_l2_sum["ella_loss"] += ella_loss_val
        self._ella_l2_sum["l2_loss"] += l2_loss_val
        self._ella_l2_n += 1
        ######################################################################

        loss_detached = loss.detach()
        self.accelerator.backward(loss)
        return loss_detached
