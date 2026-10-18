"""Export the original FP32 architecture for an on-board CPU timing baseline."""
import json

import torch
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.passes.constant_prop_pass import constant_prop_pass

from prepare import ARTIFACTS, load_model
from translator import Translator


def main():
    torch.set_num_threads(4)
    model, tokenizer = load_model()
    translator = Translator(model, tokenizer)
    with torch.no_grad():
        enc_args = translator.encode_inputs("Please close the door")
        hidden = translator.encoder(*enc_args)
        dec_args = translator.decoder_inputs([model.config.decoder_start_token_id], hidden, enc_args[1])
    result = {}
    for name, module, args in [("encoder", translator.encoder, enc_args), ("decoder", translator.decoder, dec_args)]:
        program = torch.export.export(module, args, strict=False)
        edge = to_edge_transform_and_lower(program, compile_config=EdgeCompileConfig(_check_ir_validity=False))
        constant_prop_pass(edge.exported_program())
        compiled = edge.to_executorch()
        path = ARTIFACTS / (name + "_cpu_fp32.pte")
        path.write_bytes(bytes(compiled.buffer))
        operators = compiled.executorch_program.execution_plan[0].operators
        result[name] = {"pte_bytes": path.stat().st_size,
                        "planned_buffers": compiled.executorch_program.execution_plan[0].non_const_buffer_sizes,
                        "operators": [{"name": op.name, "overload": op.overload} for op in operators]}
        print(name, json.dumps(result[name]), flush=True)
    (ARTIFACTS / "cpu_compile.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
