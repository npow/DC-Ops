"""Matching portable FP32 encoder-layer baseline for hardware speed comparison."""
import json
import sys
from pathlib import Path

import torch
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.passes.constant_prop_pass import constant_prop_pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare import ARTIFACTS, load_model
from translator import Translator


def main():
    torch.set_num_threads(4)
    model, tokenizer = load_model()
    translator = Translator(model, tokenizer)
    with torch.no_grad():
        args = translator.encode_inputs("Please close the door")
        module = translator.encoder.layers[0]
        program = torch.export.export(module, args, strict=False)
        edge = to_edge_transform_and_lower(program, compile_config=EdgeCompileConfig(_check_ir_validity=False))
        constant_prop_pass(edge.exported_program())
        compiled = edge.to_executorch()
        path = ARTIFACTS / "staging/encoder_0_cpu_fp32.pte"
        path.write_bytes(bytes(compiled.buffer))
        result = {"pte_bytes": path.stat().st_size,
                  "planned_buffers": compiled.executorch_program.execution_plan[0].non_const_buffer_sizes}
        path.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
