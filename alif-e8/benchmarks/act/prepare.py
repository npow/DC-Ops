"""Prepare the exact ACT checkpoint for latency-only board testing, without motors."""
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from act_components import ACTEncoder, ACTDecoder, ACTSinusoidalPositionEmbedding2d

ROOT = Path(__file__).resolve().parent


class FrozenBN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        for name, value in [('weight', 1), ('bias', 0), ('running_mean', 0), ('running_var', 1)]:
            self.register_buffer(name, torch.full((channels,), float(value)))

    def forward(self, x):
        scale = self.weight * (self.running_var + 1e-5).rsqrt()
        bias = self.bias - self.running_mean * scale
        return x * scale.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)


class Block(nn.Module):
    def __init__(self, ci, co, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(ci, co, 3, stride, 1, bias=False)
        self.bn1 = FrozenBN(co)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(co, co, 3, 1, 1, bias=False)
        self.bn2 = FrozenBN(co)
        self.downsample = nn.Sequential(nn.Conv2d(ci, co, 1, stride, bias=False), FrozenBN(co)) if stride != 1 or ci != co else None

    def forward(self, x):
        skip = x if self.downsample is None else self.downsample(x)
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + skip)


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = FrozenBN(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        ci = 64
        for i, co in enumerate([64, 128, 256, 512]):
            setattr(self, f'layer{i+1}', nn.Sequential(Block(ci, co, 1 if i == 0 else 2), Block(co, co, 1)))
            ci = co

    def forward(self, image):
        x = self.maxpool(self.relu(self.bn1(self.conv1(image))))
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)
        return x


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)
        self.encoder_robot_state_input_proj = nn.Linear(6, 512)
        self.encoder_latent_input_proj = nn.Linear(32, 512)
        self.encoder_img_feat_input_proj = nn.Conv2d(512, 512, 1)
        self.encoder_1d_feature_pos_embed = nn.Embedding(2, 512)
        self.decoder_pos_embed = nn.Embedding(50, 512)
        self.action_head = nn.Linear(512, 6)
        position = ACTSinusoidalPositionEmbedding2d(256)(torch.zeros(1, 512, 15, 20))
        self.register_buffer('camera_position', position.flatten(2).permute(2, 0, 1), persistent=False)

    def forward(self, front, side, state):
        front = self.encoder_img_feat_input_proj(front).flatten(2).permute(2, 0, 1)
        side = self.encoder_img_feat_input_proj(side).flatten(2).permute(2, 0, 1)
        latent = self.encoder_latent_input_proj(torch.zeros(1, 32)).unsqueeze(0)
        state = self.encoder_robot_state_input_proj(state).unsqueeze(0)
        tokens = torch.cat([latent, state, front, side], dim=0)
        position = torch.cat([self.encoder_1d_feature_pos_embed.weight.unsqueeze(1), self.camera_position, self.camera_position], dim=0)
        encoded = self.encoder(tokens, pos_embed=position)
        decoded = self.decoder(torch.zeros(50, 1, 512), encoded,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1), encoder_pos_embed=position)
        return self.action_head(decoded.transpose(0, 1))


def weights():
    raw = (ROOT / 'checkpoint/model.safetensors').read_bytes()
    n = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8+n])
    result = {}
    for name, item in header.items():
        if name == '__metadata__':
            continue
        assert item['dtype'] == 'F32', item
        a, b = item['data_offsets']
        result[name.removeprefix('model.')] = torch.frombuffer(bytearray(raw[8+n+a:8+n+b]), dtype=torch.float32).reshape(item['shape'])
    return result


def main():
    torch.set_num_threads(6)
    torch.manual_seed(20261017)
    config = SimpleNamespace(**json.loads((ROOT / 'checkpoint/config.json').read_text()))
    state = weights()
    vision, head = Vision().eval(), Head(config).eval()
    vision.load_state_dict({k.removeprefix('backbone.'):v for k,v in state.items() if k.startswith('backbone.')}, strict=True)
    head.load_state_dict({k:v for k,v in state.items() if not k.startswith(('backbone.', 'vae_'))}, strict=True)
    inputs = {'vision': (torch.rand(1, 3, 480, 640),)}
    with torch.no_grad():
        front = vision(*inputs['vision'])
        side = vision(1-inputs['vision'][0])
        inputs['head'] = (front, side, torch.zeros(1, 6))
        reference = head(*inputs['head'])
    torch.save(inputs, ROOT/'inputs.pt')
    (ROOT/'image.bin').write_bytes(inputs['vision'][0].numpy().tobytes())
    (ROOT/'state.bin').write_bytes(inputs['head'][2].numpy().tobytes())
    torch.save(reference, ROOT/'reference.pt')
    report = json.loads((ROOT/'source.json').read_text())
    report.update({'weights_sha256': hashlib.sha256((ROOT/'checkpoint/model.safetensors').read_bytes()).hexdigest(),
        'vision_parameters':sum(p.numel() for p in vision.parameters()),
        'head_parameters':sum(p.numel() for p in head.parameters()),
        'scope':'Exact checkpoint inference architecture, two 480x640 image passes and 50x6 actions. Synthetic normalized inputs; latency smoke only, no task-quality claim.',
        'input_sha256':hashlib.sha256((ROOT/'image.bin').read_bytes()).hexdigest()})
    (ROOT/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report, indent=2),flush=True)
    for name, module in [('vision',vision),('head',head)]:
        print('export',name,flush=True)
        exported = torch.export.export(module, inputs[name], strict=False)
        torch.export.save(exported, ROOT/f'{name}.pt2')
        with torch.no_grad():
            actual=exported.module()(*inputs[name])
            expected=module(*inputs[name])
            torch.testing.assert_close(actual,expected)
        print('saved',name,flush=True)


if __name__ == '__main__':
    main()
