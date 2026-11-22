"""Compare sparse-category engineering, with and without MiniLM, on saved folds."""
import argparse
import hashlib
import json

from endterm import check_continuity
from final_experiments import ROOT, OUT, MODELS, run_family
from final_text import encode_texts
from target_features import target_grids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    _, train, _, folds, _, continuity = check_continuity(ROOT)
    continuity['target_encoding_code_sha256'] = hashlib.sha256((ROOT / 'code/target_features.py').read_bytes()).hexdigest()
    manifest = json.loads((ROOT / 'assets/minilm/manifest.json').read_text())
    continuity['minilm_revision'] = manifest['revision']
    continuity['semantic_code_sha256'] = hashlib.sha256((ROOT / 'code/final_text.py').read_bytes()).hexdigest()
    encode_texts(train.plot_summary.tolist())
    for name, (model, grid) in target_grids().items():
        run_family(name, model, grid, train, folds, continuity, args.workers)


if __name__ == '__main__':
    main()
