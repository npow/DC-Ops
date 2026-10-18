"""Compare frozen quantized partitions with executable TOSA reference output."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from executorch.backends.arm.common.pipeline_config import ArmPassPipelineConfig, SoftmaxDecompositionConfig
from executorch.backends.arm.test.runner_utils import TosaReferenceModelDispatch
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare import ARTIFACTS


def main():
    torch.set_num_threads(4)
    cases = [
        ("linear_ffn", "probes/ffn_body", "probes/ffn_body_fixture.npz"),
        ("spatial_ffn", "offline_spatial_ffn/ffn_spatial", "offline_spatial_ffn/ffn_spatial_fixture.npz"),
        ("spatial_encoder0", "spatial_translator/encoder0", "spatial_translator/encoder0_fixture.npz"),
    ]
    folder = ARTIFACTS / "staging"
    results = []
    with torch.no_grad():
        for name, stem, fixture_name in cases:
            model_path = folder / (stem+"_a16w8.pt2")
            fixture_path = folder / fixture_name
            fixture = np.load(fixture_path)
            inputs = tuple(torch.from_numpy(fixture[key]) for key in sorted(fixture.files)
                           if key.startswith("input_"))
            program = torch.export.load(model_path)
            expected = program.module()(*inputs).numpy()
            spec = TosaCompileSpec("TOSA-1.0+INT+int16+int4+cf")
            spec.set_pass_pipeline_config(ArmPassPipelineConfig(softmax=SoftmaxDecompositionConfig.STABLE))
            edge = to_edge_transform_and_lower(program, partitioner=[TOSAPartitioner(spec)],
                compile_config=EdgeCompileConfig(_check_ir_validity=False))
            with TosaReferenceModelDispatch():
                actual = edge.exported_program().module()(*inputs).numpy()
            error = np.abs(actual-expected)
            row = {"name": name, "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                   "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                   "elements": int(error.size), "nonfinite": int((~np.isfinite(actual)).sum()),
                   "mean_abs_error": float(error.mean()), "max_abs_error": float(error.max()),
                   "row_mean_abs_error": error.reshape(32, -1).mean(axis=1).tolist(),
                   "execution": "TOSA reference model on host; no Vela command stream or E8 execution"}
            np.savez(folder / (name+"_tosa_reference.npz"), actual=actual, quantized_host=expected)
            results.append(row)
            print(json.dumps(row), flush=True)
    (folder / "tosa_reference_validation.json").write_text(json.dumps({"denominator": len(results),
        "results": results}, indent=2)+"\n")


if __name__ == "__main__":
    main()
