"""Compare one frozen E8 inference with its matching host quantized graph."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from export_detector import ROOT, RawDetector, letterbox, integer_io
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e


def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    folder = ROOT / 'artifacts'
    metadata = json.loads((folder / 'export_metadata.json').read_text())
    checkpoint = folder / 'dc_ops_yolov8n_seg_v3.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == metadata['checkpoint_sha256']
    samples = []
    for entry in metadata['calibration']:
        path = folder / 'calibration' / entry['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
        samples.append(letterbox(path, metadata['input_size']))
    model = RawDetector(YOLO(checkpoint).model.float().eval().fuse()).eval()
    spec = EthosUCompileSpec('ethos-u85-256', system_config='Ethos_U85_SRAM_MRAM',
        memory_mode='Shared_Sram', config_ini=str(ROOT / '.cmsis/ensemble_vela.ini'))
    graph = torch.export.export(model, (samples[0],), strict=False).module()
    for node in list(graph.graph.nodes):
        if node.op == 'call_function' and '_assert_tensor_metadata' in str(node.target):
            graph.graph.erase_node(node)
    graph.recompile()
    quantizer = EthosUQuantizer(spec).set_global(get_symmetric_quantization_config())
    prepared = prepare_pt2e(graph, quantizer)
    with torch.no_grad():
        for image in samples: prepared(image)
    quantized = convert_pt2e(prepared)
    input_params, output_params = integer_io(quantized)
    assert input_params == metadata['input'] and output_params == metadata['outputs']
    size = metadata['input_size']
    raw = np.fromfile(folder / 'hw_input.bin', dtype=np.int8).reshape(1,size,size,3).transpose(0,3,1,2)
    with torch.no_grad():
        reference = quantized(torch.from_numpy(raw.copy()))
    report = {'input_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
              'pte_sha256': metadata['pte_sha256'],
              'calibration_count': len(samples), 'heads': [],
              'scope': 'One real camera inference; numerical comparison, not detection accuracy evaluation.'}
    for i, tensor in enumerate(reference):
        expected = tensor.numpy()
        expected.tofile(folder / f'host_output_{i}.bin')
        actual = np.fromfile(folder / f'hw_output_{i}.bin', dtype=np.int8).reshape(expected.shape)
        delta = np.abs(actual.astype(np.int16)-expected.astype(np.int16))
        report['heads'].append({'shape': list(expected.shape), 'elements': int(delta.size),
            'mean_abs_int8_error': float(delta.mean()), 'max_abs_int8_error': int(delta.max()),
            'within_2_lsb_count': int((delta <= 2).sum()),
            'hardware_sha256': hashlib.sha256(actual.tobytes()).hexdigest()})
    (folder / 'hardware_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
