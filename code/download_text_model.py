"""Download the public MiniLM ONNX files at a recorded, immutable revision."""

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
FILES = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'config.json',
         '1_Pooling/config.json', 'README.md', 'onnx/model_qint8_arm64.onnx']


def main():
    folder = ROOT / 'assets/minilm'
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1)))
    if previous:
        revision = previous['revision']
    else:
        response = session.get(f'https://huggingface.co/api/models/{MODEL_ID}', timeout=60)
        response.raise_for_status()
        metadata = response.json()
        available = {entry['rfilename'] for entry in metadata['siblings']}
        if not set(FILES).issubset(available):
            raise RuntimeError(f'Model files unavailable: {set(FILES)-available}')
        revision = metadata['sha']
        (folder / 'source_metadata.json').write_text(json.dumps(metadata, indent=2))
    records = []
    for filename in FILES:
        target = folder / filename
        known = next((item for item in previous.get('files', []) if item['name'] == filename), None)
        if target.exists() and known and hashlib.sha256(target.read_bytes()).hexdigest() == known['sha256']:
            records.append(known)
            print(f'Verified {filename}', flush=True)
            continue
        url = f'https://huggingface.co/{MODEL_ID}/resolve/{revision}/{quote(filename, safe="/")}'
        print(f'Downloading {filename}', flush=True)
        response = session.get(url, timeout=(30, 120), stream=True)
        response.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.part')
        digest = hashlib.sha256()
        with temporary.open('wb') as stream:
            for chunk in response.iter_content(1024*1024):
                digest.update(chunk)
                stream.write(chunk)
        temporary.replace(target)
        records.append({'name': filename, 'sha256': digest.hexdigest(),
                        'bytes': target.stat().st_size, 'url': url})
    license_path = folder / 'LICENSE.txt'
    license_url = 'https://www.apache.org/licenses/LICENSE-2.0.txt'
    if not license_path.exists():
        response = session.get(license_url, timeout=60)
        response.raise_for_status()
        license_path.write_text(response.text, encoding='utf-8')
    records.append({'name': 'LICENSE.txt', 'sha256': hashlib.sha256(license_path.read_bytes()).hexdigest(),
                    'bytes': license_path.stat().st_size, 'url': license_url})
    manifest = {'model_id': MODEL_ID, 'revision': revision, 'license': 'Apache-2.0',
                'model_card': f'https://huggingface.co/{MODEL_ID}', 'files': records,
                'representation': 'Frozen 6-layer MiniLM, quantised ONNX, attention-masked mean pooling, L2 norm',
                'output_dimensions': 384, 'max_tokens': 256}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'MiniLM ready at revision {revision}', flush=True)


if __name__ == '__main__':
    main()
