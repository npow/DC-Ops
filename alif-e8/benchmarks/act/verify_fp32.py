"""Cross-check the deployment split against LeRobot's unchanged ACT forward."""
import ast
import json
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

import einops
import numpy as np
import torch
from torch import nn, Tensor
from act_components import ACTEncoder, ACTDecoder, ACTSinusoidalPositionEmbedding2d
from prepare import Vision, Head, weights, ROOT


class ReferenceVision(Vision):
    def __init__(self):
        super().__init__()
        self.fc=SimpleNamespace(in_features=512)

    def forward(self,x):
        return {'feature_map':super().forward(x)}


def main():
    torch.set_num_threads(6)
    provenance=json.loads((ROOT/'provenance.json').read_text())
    source_path=ROOT/'checkpoint/modeling_act.py'
    if not source_path.exists():
        source_path=Path(provenance['components_source'])
    source=source_path.read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='ACT')
    namespace=dict(torch=torch,nn=nn,Tensor=Tensor,einops=einops,np=np,chain=chain,
        ACTConfig=object,ACTEncoder=ACTEncoder,ACTDecoder=ACTDecoder,
        ACTSinusoidalPositionEmbedding2d=ACTSinusoidalPositionEmbedding2d,
        FrozenBatchNorm2d=object,OBS_IMAGES='images',OBS_STATE='state',OBS_ENV_STATE='env',ACTION='action',
        torchvision=SimpleNamespace(models=SimpleNamespace(resnet18=lambda **kw:ReferenceVision())),
        IntermediateLayerGetter=lambda backbone,return_layers:backbone)
    exec(compile(ast.get_source_segment(source,node),provenance['components_source'],'exec'),namespace)
    config=SimpleNamespace(**json.loads((ROOT/'checkpoint/config.json').read_text()))
    config.use_vae=False
    config.robot_state_feature=SimpleNamespace(shape=[6])
    config.env_state_feature=None
    config.image_features={'front':None,'side':None}
    config.action_feature=SimpleNamespace(shape=[6])
    ref=namespace['ACT'](config).eval()
    state={k:v for k,v in weights().items() if not k.startswith('vae_')}
    ref.load_state_dict(state,strict=True)
    v=torch.export.load(ROOT/'vision.pt2').module()
    h=torch.export.load(ROOT/'head.pt2').module()
    samples=torch.load(ROOT/'inputs.pt',weights_only=True)
    image=samples['vision'][0];joints=samples['head'][2]
    with torch.no_grad():
        expected=ref({'images':[image,1-image],'state':joints})[0]
        actual=h(v(image),v(1-image),joints)
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
    report={'passed':True,'max_absolute_difference':float((actual-expected).abs().max()),
        'output_shape':list(actual.shape),'scope':'One synthetic input pair; split deployment forward vs unchanged LeRobot ACT forward, same strictly loaded checkpoint and ResNet18 implementation.'}
    (ROOT/'fp32_equivalence.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
