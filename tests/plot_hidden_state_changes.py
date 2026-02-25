#!/usr/bin/env python3
"""
Plot per layer the norm of (hidden_after_t - hidden_before_t) when learning task t.

Before task t = hidden states from model after tasks 1..t-1 (or base for t=1).
After task t = hidden states from model after tasks 1..t.
We run a fixed input through the model with the appropriate adapter(s) and compute
the L2 norm of the difference per layer (encoder and decoder separately).

Layer count vs adapter-norm plots:
  - Hidden states from the model have length 1 + num_blocks: index 0 is the
    EMBEDDING output (before any transformer block), indices 1..N are after
    each block. LoRA does not modify the embedding, so index 0 often has ~0 change.
  - Adapter norm plots one value per transformer BLOCK that has LoRA (no embedding).
  - So you see one more "layer" (24) here than "blocks" (23) there: the extra is
    the embedding (index 0). If the LAST index has 0 change for all tasks, that
    is the output of the final block; some setups only add LoRA to the first
    N-1 blocks, so the last block has no LoRA and its output does not change.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Use project PEFT (custom save/load)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel


def collect_task_folders(base_dir: Path) -> List[Tuple[int, str, Path]]:
    """Return list of (task_nr, task_name, path_to_adapter_dir) sorted by task_nr."""
    tasks = []
    for d in base_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if "-" in name:
            prefix = name.split("-", 1)[0]
            if prefix.isdigit():
                task_nr = int(prefix)
                task_name = name.split("-", 1)[1]
                adapter_dir = d / "adapter"
                if (adapter_dir / "adapter_model.bin").is_file() and (adapter_dir / "adapter_config.json").is_file():
                    tasks.append((task_nr, task_name, adapter_dir))
    tasks.sort(key=lambda x: x[0])
    return tasks


def get_base_model_path(adapter_dir: Path) -> str:
    """Read base_model_name_or_path from adapter_config.json."""
    with open(adapter_dir / "adapter_config.json") as f:
        config = json.load(f)
    return config["base_model_name_or_path"]


def run_and_get_hidden_states(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
    """Run model with output_hidden_states=True; return (encoder_hidden_states, decoder_hidden_states)."""
    model.eval()
    with torch.no_grad():
        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "decoder_input_ids": decoder_input_ids.to(device),
            "output_hidden_states": True,
        }
        outputs = model(**inputs)
    enc = outputs.encoder_hidden_states  # tuple of (1 + num_encoder_layers) tensors
    dec = outputs.decoder_hidden_states  # tuple of (1 + num_decoder_layers) tensors
    return enc, dec


def norm_of_diff(
    before: Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]],
    after: Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]],
) -> Tuple[List[float], List[float]]:
    """Compute per-layer L2 norm of (after - before). Returns (encoder_norms, decoder_norms)."""
    enc_before, dec_before = before
    enc_after, dec_after = after
    enc_norms = []
    for a, b in zip(enc_after, enc_before):
        diff = (a.float() - b.float()).to("cpu")
        enc_norms.append(diff.norm().item())
    dec_norms = []
    for a, b in zip(dec_after, dec_before):
        diff = (a.float() - b.float()).to("cpu")
        dec_norms.append(diff.norm().item())
    return enc_norms, dec_norms


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-layer norm of hidden state change (after_t - before_t) when learning task t"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="logs_and_outputs/ella/long_1000train/order_4/outputs2",
        help="Directory containing 1-taskname, 2-taskname, ... adapter folders",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model path (default: read from first adapter config; use e.g. initial_model/t5-large)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="hidden_state_changes",
        help="Output plot path prefix (<out>_encoder.png, <out>_decoder.png)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for forward pass",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Max number of tasks to process (default: all)",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        default=None,
        help="Restrict to these task numbers (default: all)",
    )
    parser.add_argument(
        "--no_embed",
        action="store_true",
        help="Drop the first layer (embedding) so x-axis aligns with transformer blocks only (same count as adapter norm)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        raise SystemExit(f"Not a directory: {base_dir}")

    tasks = collect_task_folders(base_dir)
    if not tasks:
        raise SystemExit(f"No task adapter folders found under {base_dir}")

    if args.tasks is not None:
        tasks = [t for t in tasks if t[0] in args.tasks]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
    if not tasks:
        raise SystemExit("No tasks left after filter.")

    # Resolve base model path
    base_model_path = args.base_model or get_base_model_path(tasks[0][2])
    if not Path(base_model_path).is_dir() and not base_model_path.startswith("initial_model"):
        # Try relative to project root (CL_project or repo root)
        for root in [ROOT, ROOT.parent]:
            candidate = root / base_model_path
            if candidate.is_dir():
                base_model_path = str(candidate)
                break

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)
    base_model.to(device)

    # Fixed input so all runs are comparable (same seq lengths for before/after)
    dummy_input = "translate English to German: The house is wonderful."
    dummy_target = "Das Haus ist wunderbar."
    enc = tokenizer(
        dummy_input,
        return_tensors="pt",
        padding="max_length",
        max_length=32,
        truncation=True,
    )
    dec = tokenizer(
        dummy_target,
        return_tensors="pt",
        padding="max_length",
        max_length=32,
        truncation=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    decoder_input_ids = dec["input_ids"]

    # h_before_0 = hidden states from base model (before any task)
    enc_h0, dec_h0 = run_and_get_hidden_states(
        base_model, input_ids, attention_mask, decoder_input_ids, device
    )

    # Load first adapter so we have a PeftModel and can load_adapter for the rest
    first_adapter_dir = tasks[0][2]
    peft_model = PeftModel.from_pretrained(base_model, str(first_adapter_dir))
    peft_model.to(device)
    enc_h1, dec_h1 = run_and_get_hidden_states(
        peft_model, input_ids, attention_mask, decoder_input_ids, device
    )

    # Norm of change when learning task 1
    enc_norms_1, dec_norms_1 = norm_of_diff((enc_h0, dec_h0), (enc_h1, dec_h1))
    num_enc_layers = len(enc_norms_1)
    num_dec_layers = len(dec_norms_1)

    data_enc: List[Tuple[int, str, List[float]]] = [(tasks[0][0], tasks[0][1], enc_norms_1)]
    data_dec: List[Tuple[int, str, List[float]]] = [(tasks[0][0], tasks[0][1], dec_norms_1)]

    # For tasks 2..N: before = hidden states from model after task i-1 (stored as enc_prev, dec_prev)
    enc_prev, dec_prev = enc_h1, dec_h1
    for i in range(1, len(tasks)):
        task_nr, task_name, adapter_dir = tasks[i]
        peft_model.load_adapter(str(adapter_dir), adapter_name=f"task_{task_nr}")
        peft_model.set_adapter(f"task_{task_nr}")
        enc_after, dec_after = run_and_get_hidden_states(
            peft_model, input_ids, attention_mask, decoder_input_ids, device
        )
        enc_norms, dec_norms = norm_of_diff((enc_prev, dec_prev), (enc_after, dec_after))
        enc_prev, dec_prev = enc_after, dec_after
        data_enc.append((task_nr, task_name, enc_norms))
        data_dec.append((task_nr, task_name, dec_norms))

    # Optionally drop embedding (index 0) so layer count matches adapter-norm blocks
    if args.no_embed:
        data_enc = [(t, n, norms[1:]) for t, n, norms in data_enc]
        data_dec = [(t, n, norms[1:]) for t, n, norms in data_dec]
        num_enc_layers -= 1
        num_dec_layers -= 1

    # Plot (same style as plot_w_past_per_layer / plot_adapter_norm)
    out_prefix = args.out.rstrip(".png")
    if out_prefix == args.out and not out_prefix.endswith("_"):
        out_prefix = args.out

    def do_plot(
        layer_indices: List[int],
        layer_labels: List[str],
        data: List[Tuple[int, str, List[float]]],
        title: str,
        out_path: str,
        ylabel: str = "||Δhidden|| (L2)",
    ) -> None:
        x = np.arange(len(layer_indices))
        fig, ax = plt.subplots(figsize=(12, 6))
        for task_nr, task_name, norms_list in data:
            line, = ax.plot(x, norms_list, marker="o", markersize=3, label=f"{task_nr}-{task_name}", alpha=0.8)
            ax.annotate(
                str(task_nr),
                xy=(x[-1], norms_list[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8,
                color=line.get_color(),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=line.get_color(), alpha=0.9),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels, rotation=0)
        ax.set_xlabel("Block index (0 = first transformer block)" + (" (embedding excluded)" if args.no_embed else " (0 = embed)"))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")

    # Labels: with embed we show "embed", "0", "1", ... (block indices); without embed just "0", "1", ...
    enc_layer_idx = list(range(num_enc_layers))
    dec_layer_idx = list(range(num_dec_layers))
    enc_labels = (["embed"] + [str(i) for i in range(num_enc_layers - 1)]) if not args.no_embed else [str(i) for i in range(num_enc_layers)]
    dec_labels = (["embed"] + [str(i) for i in range(num_dec_layers - 1)]) if not args.no_embed else [str(i) for i in range(num_dec_layers)]
    do_plot(
        enc_layer_idx,
        enc_labels,
        data_enc,
        "Hidden state change per layer (encoder): ||h_after_t − h_before_t||",
        f"{out_prefix}_encoder.png",
    )
    do_plot(
        dec_layer_idx,
        dec_labels,
        data_dec,
        "Hidden state change per layer (decoder): ||h_after_t − h_before_t||",
        f"{out_prefix}_decoder.png",
    )


if __name__ == "__main__":
    main()
