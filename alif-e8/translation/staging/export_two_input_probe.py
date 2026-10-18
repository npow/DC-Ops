"""Add an unused mask input without recompiling the frozen spatial FFN delegate."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from executorch.exir._serialize._program import deserialize_pte_binary, serialize_pte_binary
from executorch.exir.schema import EValue


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    artifacts = Path(__file__).resolve().parents[2] / 'artifacts/translator/staging'
    source = artifacts / 'offline_spatial_ffn'
    destination = artifacts / 'offline_spatial_ffn_two_input'
    destination.mkdir(parents=True, exist_ok=True)
    original = (source / 'ffn_spatial_u85.pte').read_bytes()
    parsed = deserialize_pte_binary(original)
    plan = parsed.program.execution_plan[0]
    assert len(plan.inputs) == 1 and len(plan.delegates) == 1
    delegate_data = [bytes(buffer.data) for buffer in parsed.program.backend_delegate_data]
    original_chains = copy.deepcopy(plan.chains)
    original_memory = copy.deepcopy(plan.non_const_buffer_sizes)
    mask = copy.deepcopy(plan.values[plan.inputs[0]].val)
    mask.sizes = [1, 1, 1, 32]
    mask.dim_order = [0, 1, 2, 3]
    mask.allocation_info = None
    plan.inputs.append(len(plan.values))
    plan.values.append(EValue(mask))
    input_tree = json.loads(plan.container_meta_type.encoded_inp_str)
    positional_args = input_tree[1]['children_spec'][0]['children_spec']
    positional_args.append({'type': None, 'context': None, 'children_spec': []})
    plan.container_meta_type.encoded_inp_str = json.dumps(input_tree)
    binary = bytes(serialize_pte_binary(parsed))
    assert len(binary) <= 1038896
    checked = deserialize_pte_binary(binary + bytes(1038896 - len(binary)))
    checked_plan = checked.program.execution_plan[0]
    assert len(checked_plan.inputs) == 2 and len(checked_plan.delegates) == 1
    assert checked_plan.chains == original_chains
    assert checked_plan.non_const_buffer_sizes == original_memory
    assert [bytes(buffer.data) for buffer in checked.program.backend_delegate_data] == delegate_data
    inputs = [checked_plan.values[index].val for index in checked_plan.inputs]
    output = checked_plan.values[checked_plan.outputs[0]].val
    assert [tensor.sizes for tensor in inputs] == [[1, 32, 256], [1, 1, 1, 32]]
    assert output.sizes == [1, 32, 256]
    assert all(int(tensor.scalar_type) == 6 for tensor in inputs + [output])
    path = destination / 'ffn_spatial_two_input_u85.pte'
    path.write_bytes(binary)
    fixture = np.load(source / 'ffn_spatial_fixture.npz')
    np.savez(destination / 'fixture.npz', input_0=fixture['input_0'],
             input_1=np.zeros((1, 1, 1, 32), dtype=np.float32),
             output=fixture['output'], fp32_output=fixture['fp32_output'])
    report = {
        'program': str(path), 'pte_bytes': len(binary), 'padded_bytes': 1038896,
        'source_sha256': sha(original), 'sha256': sha(binary),
        'delegate_sha256': [sha(data) for data in delegate_data],
        'delegate_payload_unchanged': True, 'instructions_unchanged': True,
        'inputs': [tensor.sizes for tensor in inputs], 'output': output.sizes,
        'planned_buffers': checked_plan.non_const_buffer_sizes,
        'delegates': len(checked_plan.delegates),
        'scratch_bytes': 16384, 'fast_scratch_bytes': 106624,
        'mask_semantics': 'Unused external float32 input; no allocation or computation',
        'hardware_tested': False,
    }
    (destination / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
