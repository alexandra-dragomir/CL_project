# T5 block structure and why the "last" hidden state has 0 change

## 1. T5 structure (HuggingFace) — all blocks are the same

- **Encoder/decoder stack:** `T5Stack` has `self.block = nn.ModuleList([T5Block(...) for i in range(config.num_layers)])`, so blocks are `0 .. num_layers-1` (e.g. 0..23 for 24 layers).
- **Each T5Block** (encoder and decoder) has the same layout:
  - `layer[0]` = `T5LayerSelfAttention` → contains `SelfAttention` with **q, k, v, o** (`nn.Linear`).
  - Decoder also has `layer[1]` = `T5LayerCrossAttention` → `EncDecAttention` with q, k, v, o.
  - `layer[-1]` = `T5LayerFF` (feed-forward: wi, wo) — **not** targeted by LoRA when `target_modules=["q","v"]`.

So **block 23 has the same structure as block 0**: it has `SelfAttention.q` and `SelfAttention.v`. There is no special case for the last block in the model code.

## 2. What gets LoRA in this project

- **Target modules for T5:** `["q", "v"]` (see `src/peft/utils/other.py`, `TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING`).
- **Where LoRA is applied:** `src/peft/tuners/lora.py` → `_find_and_replace()` iterates over **every** module key and replaces any module whose name **ends with** `"q"` or `"v"`. There is **no** check on block index and **no** code that skips the last block.

So by code, **every** block that has a submodule named `q` or `v` (i.e. all of them) is a candidate for LoRA. If a saved adapter has LoRA only for blocks 0..22, that comes from what is in the **checkpoint** (e.g. a run with a 23-layer model or a different config), not from the project “skipping” block 23.

## 3. What the “last” hidden state actually is (and why it’s 0)

In HuggingFace T5, `encoder_hidden_states` / `decoder_hidden_states` are built in **T5Stack.forward** roughly as:

```python
for layer_module in self.block:
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)  # before this block
    layer_outputs = layer_module(...)
    hidden_states = layer_outputs[0]
# After the loop:
hidden_states = self.final_layer_norm(hidden_states)
hidden_states = self.dropout(hidden_states)
if output_hidden_states:
    all_hidden_states = all_hidden_states + (hidden_states,)  # last element
```

So the **last** element in the tuple is **after `final_layer_norm` and dropout**, not “after block 23”. That is:

- **Not** a transformer block.
- A **T5LayerNorm** + dropout. LoRA is only applied to Linear layers (q, v, etc.), so this part is **never** replaced by LoRA — it stays the same (frozen) for all adapters.

If the **second-to-last** entry (output of block 23) also has ~0 change, then in that checkpoint **block 23 has no LoRA**: its `q` and `v` are still the original `nn.Linear` and were never replaced. So:

- **Block 23:** same structure as other blocks (SelfAttention.q, .v). It has no LoRA **in that saved adapter**, so it stays the original Linear layers (frozen in the sense “not LoRA, not updated by the adapter”).
- **Last hidden state:** output of `final_layer_norm`; no LoRA by design, so it always shows 0 change when comparing adapters.

## 4. Summary

| What                | Structure                         | LoRA? | Frozen? |
|---------------------|-----------------------------------|-------|--------|
| Block 0..22         | SelfAttention.q, .v (same each)   | Yes (in your checkpoint) | No (trainable LoRA) |
| Block 23            | Same as above                     | No (in your checkpoint)   | Yes (original Linear, not replaced) |
| Last hidden state   | After `final_layer_norm` + dropout | No (LayerNorm, not a Linear) | Yes (never has LoRA) |

So: **block 23 does not have LoRA in the adapter you inspected**; it stays the base (frozen) `nn.Linear` for q and v. The **very last** entry in the hidden-state tuple is not a block at all — it’s the stack’s final_layer_norm output, which has no LoRA and therefore always shows 0 change.
