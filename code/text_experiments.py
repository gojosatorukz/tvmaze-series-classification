"""Run structured, TF-IDF and latent-text controls with identical saved folds."""
import argparse

from endterm import check_continuity
from final_experiments import ROOT, OUT, MODELS, run_family
from final_models import synopsis_grids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    _, train, _, folds, _, continuity = check_continuity(ROOT)
    for name, (model, grid) in synopsis_grids().items():
        run_family(name, model, grid, train, folds, continuity, args.workers)


if __name__ == '__main__':
    main()
