"""Run the saved-fold MiniLM comparison using the common exhaustive runner."""
import argparse
import hashlib
import json
from pathlib import Path

from endterm import check_continuity
from final_experiments import ROOT, OUT, MODELS, run_family
from final_text import semantic_grids, encode_texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    _, train, _, folds, _, continuity = check_continuity(ROOT)
    manifest = json.loads((ROOT / 'assets/minilm/manifest.json').read_text())
    continuity['minilm_revision'] = manifest['revision']
    continuity['semantic_code_sha256'] = hashlib.sha256((ROOT / 'code/final_text.py').read_bytes()).hexdigest()
    print('Encoding training synopses with frozen MiniLM', flush=True)
    embeddings = encode_texts(train.plot_summary.tolist())
    print(f'Training embedding matrix: {embeddings.shape}', flush=True)
    for name, (model, grid) in semantic_grids().items():
        run_family(name, model, grid, train, folds, continuity, args.workers)


if __name__ == '__main__':
    main()
