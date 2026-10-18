"""Fetch the pinned checkpoint and LeRobot reference, checking recorded hashes."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


def fetch(url, destination, expected):
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
        print('verified', destination.name)
        return
    temporary = destination.with_suffix(destination.suffix + '.download')
    digest = hashlib.sha256()
    with urlopen(url) as response, temporary.open('wb') as output:
        while block := response.read(1024 * 1024):
            digest.update(block)
            output.write(block)
    if digest.hexdigest() != expected:
        temporary.unlink()
        raise RuntimeError(f'Hash mismatch for {destination.name}')
    temporary.replace(destination)
    print('downloaded and verified', destination.name)


def main():
    source = json.loads((ROOT / 'source.json').read_text())
    manifest = json.loads((ROOT / 'artifact_manifest.json').read_text())
    provenance = json.loads((ROOT / 'provenance.json').read_text())
    (ROOT / 'checkpoint').mkdir(exist_ok=True)
    for name in ['config.json', 'model.safetensors']:
        fetch(f'https://huggingface.co/{source["repo"]}/resolve/{source["revision"]}/{name}',
              ROOT / 'checkpoint' / name, manifest[f'checkpoint/{name}']['sha256'])
    fetch(f'https://raw.githubusercontent.com/huggingface/lerobot/{provenance["upstream_revision"]}/{provenance["upstream_path"]}',
          ROOT / 'checkpoint/modeling_act.py', provenance['sha256'])


if __name__ == '__main__':
    main()
