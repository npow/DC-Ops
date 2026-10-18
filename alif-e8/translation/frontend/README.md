# Embedded English → Spanish frontend

`frontend.h` / `frontend.cpp` provide the CPU portion around the fixed-shape ExecuTorch encoder and decoder. There is no heap allocation, SentencePiece runtime, tokenizer Python dependency, or C++ standard-library container in the embedded implementation. Model execution remains the caller's responsibility.

Export and check with the existing translation environment:

```sh
translation/.venv/bin/python translation/frontend/validate.py
translation/.venv/bin/python translation/frontend/validate.py --reuse-assets --sanitize
```

The generated assets and evidence live in ignored `artifacts/translator/frontend/`. The exporter uses the pinned checkpoint from `prepare.py`. `manifest.json` records checkpoint revision, frozen sample hash, asset hashes, token IDs, and expected decode strings. Validation checks 32 frozen source phrases plus 8 boundary/normalization phrases, their exact padded token IDs, exact encoder/decoder embeddings and masks, 32 generated Spanish decodes plus 4 accent/punctuation cases, and error boundaries. These checks establish frontend parity, not translation quality or device performance.

## Integration

1. Map `lexicon.bin`, `embeddings_fp16.bin`, and `positions_fp32.bin` read-only, aligned to at least 4 bytes. Pass their pointers and byte lengths to `Frontend::initialize`. Assets must remain available for the lifetime of the frontend. They are little-endian and exported for this checkpoint only; check their manifest hashes before deployment.
2. `tokenizeEnglish(text, ids, count)` writes 32 IDs including EOS and PAD. It rejects input that would exceed 31 content tokens rather than silently clipping it. `encoderInputs(ids, embeddings, mask)` fills float32 `[1,32,256]` and `[1,1,1,32]` buffers for the encoder.
3. Begin the decoder prefix with ID 32000. `decoderInputs(prefix, count, embeddings, last_position)` fills float32 `[1,32,256]` and `[1,32,1]`. Invoke the decoder with these, the encoder output, and the unchanged encoder mask. `greedyToken(logits)` selects the highest logit while excluding PAD. Append the result; stop at EOS 0, or at 32 prefix tokens. Report a length limit if EOS has not arrived. Do not call the decoder with an empty prefix.
4. `detokenizeSpanish(ids, count, output, capacity)` converts generated IDs into UTF-8, removes the model's special tokens, converts SentencePiece word markers to spaces, and trims outside spaces. It supports the checkpoint's Spanish vocabulary; draw the resulting UTF-8 with a font covering Spanish characters.

Never share a single frontend between simultaneous tokenization callers; it has mutable Viterbi workspace. Input and output arrays belong to the caller, and should not overlap the immutable assets. Embedded compilation must retain floating-point NaN/Inf semantics (do not enable `-ffast-math`).

## Memory and supported input

Assets: embedding 16,384,512 bytes; lexicon 2,337,119 bytes; positions 32,768 bytes (18,754,399 total). Place the large immutable assets in external RAM or flash. Frontend object uses 11,328 bytes on the 64-bit host, with about 2 KB additional tokenizer stack; the M55 build can use fewer pointer bytes. Each float embedding buffer is another 32,768 bytes and can be reused between encoder and decoder calls. No KV cache is used by the current exported decoder.

The tokenizer implements unigram Viterbi segmentation with the checkpoint's scores and its shared vocabulary. Normalization accepts printable ASCII, ordinary whitespace, and precomposed Latin-1 letters/punctuation with their supported NFKC mappings. It collapses spaces and inserts SentencePiece's dummy word prefix. Latin-1 spacing accents, fractions, and micro sign are explicitly rejected because their NFKC result needs characters outside this implementation's supported normalization range. Combining accents, arbitrary Unicode, malformed UTF-8, and control characters return `UnsupportedUnicode`. OCR must supply precomposed supported text; do not silently drop unsupported characters. Unknown characters that the tokenizer vocabulary cannot represent retain the checkpoint's `<unk>` behavior.

`frontend.cpp` cross-compiles for Cortex-M55 with Arm GNU 15.3.1, C++17, `-Os -mthumb -mcpu=cortex-m55 -fno-exceptions -fno-rtti`. Runtime calls use only ordinary C string/math functions; no C++ runtime initialization is needed for the immutable assets.
