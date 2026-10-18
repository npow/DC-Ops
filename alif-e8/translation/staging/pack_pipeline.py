"""Pack the existing 12 staged programs into one aligned hardware transfer."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/translator/staging"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=ARTIFACTS)
    folder = parser.parse_args().directory.resolve()
    names = [f"encoder_{i}" for i in range(6)] + [f"decoder_{i}" for i in range(2)] + [f"head_{i}" for i in range(4)]
    image = bytearray()
    stages = []
    for i, name in enumerate(names):
        image.extend(b"\0"*((-len(image)) % 32))
        raw = (folder / (name+"_u85.pte")).read_bytes()
        inputs = [[1, 32, 256], [1, 1, 1, 32]] if i < 6 else (
            [[1, 32, 256], [1, 32, 256], [1, 1, 1, 32]] if i < 8 else [[1, 32, 256], [1, 32, 1]])
        output = [1, 32, 256] if i < 8 else [1, 8001 if i == 11 else 8000]
        stages.append({"index": i, "name": name, "offset": len(image), "bytes": len(raw),
                       "input_shapes": inputs, "output_shape": output,
                       "sha256": hashlib.sha256(raw).hexdigest()})
        image.extend(raw)
    image.extend(b"\0"*((-len(image)) % 32))
    (folder / "pipeline.bin").write_bytes(image)
    manifest = {"alignment": 32, "bytes": len(image), "sha256": hashlib.sha256(image).hexdigest(),
                "scalar_type": "float32", "stages": stages}
    (folder / "pipeline_index.json").write_text(json.dumps(manifest, indent=2)+"\n")
    header = ["#pragma once", "#include <stdint.h>", "#define DCOPS_PIPELINE_STAGE_COUNT 12",
              f"#define DCOPS_PIPELINE_BYTES {len(image)}u",
              "struct DcopsPackedStage { uint32_t offset; uint32_t bytes; uint32_t input_count; uint32_t output_elements; };",
              "static constexpr DcopsPackedStage dcops_packed_stages[12] = {"]
    for stage in stages:
        elements = 8192 if stage["index"] < 8 else stage["output_shape"][-1]
        header.append(f"    {{{stage['offset']}u, {stage['bytes']}u, {len(stage['input_shapes'])}u, {elements}u}}, // {stage['name']}")
    header.append("};\n")
    (folder / "pipeline_index.h").write_text("\n".join(header))
    print(json.dumps({"binary": str(folder / "pipeline.bin"), "bytes": len(image), "stages": len(stages)}))


if __name__ == "__main__":
    main()
