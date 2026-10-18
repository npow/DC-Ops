This fallback keeps each compiled NPU program and its weights below the
2.5 MiB staging allocation. It requires no change to the trained architecture.

The live firmware now selects `artifacts/translator/spatial_staging`, produced
with `export_staged.py --spatial-ffn --output artifacts/translator/spatial_staging`.
This equivalent FFN uses two 1×1 convolutions to preserve the token-row layout.
The original fused linear FFN skipped rows on E8: rows 0–3 matched reference
rows 0, 8, 16 and 24. The isolated spatial FFN passed on E8 across all 8,192
outputs (maximum absolute error 0.00038624), with no firmware change or display
error during that diagnostic. Its report is in `spatial_ffn_hardware`.

The spatial pipeline preserves the pretrained weights and the four original
vocabulary heads. Its frozen host PT2E evaluation matches 8/8 baseline token
sequences. Executable TOSA reference evaluation matches 7/8 token sequences and
finishes 8/8: “Press the green button” produces “Pulse el botón verde” instead
of “Presione el botón verde”. These are distinct numerical-validation results;
neither is an end-to-end E8 accuracy claim. `live_translation_evaluation` holds
the separate physical-board results when available.

Current hardware status: the original Dedicated_Sram encoder and fused FFN
partitions execute but fail numerical comparison. Isolated attention, softmax,
LayerNorm and both FFN linear projections pass. These default artifacts are
diagnostic until the fused-graph discrepancy is resolved. The `variants`
directory contains recompiles with Shared_Sram and identical quantized weights
for checking whether the fast-scratch scheduling causes the difference. The
Shared_Sram FFN result has the same hardware error as Dedicated_Sram. The
`variants/flag_audit.json` command hashes show that disabling cascading or
weight buffering has no effect for this graph; disabling chaining or IFM reuse
changes its command stream and remains an independent hardware experiment.

Run from `alif-e8`:

```sh
translation/.venv/bin/python translation/staging/export_staged.py
translation/.venv/bin/python translation/staging/fixtures.py
translation/.venv/bin/python translation/staging/export_cpu_layer.py
```

Artifacts go to `artifacts/translator/staging`. `evaluation.json` compares all
tokens of the assembled PT2E quantized network with the original floating point
adapter on the same eight frozen phrases. This is a small parity check, not a
translation-quality evaluation or hardware validation. Model and sample hashes
are inherited from the parent translator workflow.

Execution order is encoder_0 through encoder_5 once per source phrase, then
decoder_0, decoder_1, and head_0 through head_3 once per generated token.
Encoder inputs are hidden `[1,32,256]` and source mask `[1,1,1,32]`.
Decoder inputs are hidden, encoded `[1,32,256]`, and source mask; causal masking
is constant inside each partition. Head inputs are decoder hidden and the
last-token one-hot mask `[1,32,1]`. Each head performs position selection itself.
The four vocabulary ranges are `[0,8000)`, `[8000,16000)`, `[16000,24000)`,
and `[24000,32001)`. Their float outputs can be concatenated or searched for a
global maximum directly; each output's dequantization is inside its PTE.
Suppress pad token 32000, append the largest-logit token, and stop at EOS 0.

Stage each PTE into SRAM before loading the method. Its planned buffers must
also be allocated in SRAM because they contain quantized delegate inputs and
outputs. Fast scratch must stay in SRAM. CPU-readable external RAM may hold
the source PTEs, embeddings, float boundary tensors, and state between calls.
Recreate the loader/program/method for each staged PTE, preserving the float
state elsewhere before reusing the allocation. `manifest.json` lists PTE sizes,
planned memory, delegate count and CPU boundary operators. Vela scratch and
performance estimates are in each `vela_*/output` directory; estimates are
not measurements of the E8.

`*_fixture.npz` contains each partition's `input_0` through `input_n` and
expected quantized `output`. The first encoder fixture has an additional
`hardware_fixture.npz` alias using the existing benchmark field names.
`encoder_0_cpu_fp32.pte` supplies the corresponding portable FP32 CPU timing
baseline. It is larger than the SRAM staging allocation and can be read from
external RAM by the CPU.
