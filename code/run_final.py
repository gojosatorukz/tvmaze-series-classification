"""Run the complete Final study, or verify the included trained artefacts."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(filename, arguments=()):
    subprocess.run([sys.executable, str(ROOT / 'code' / filename), *arguments], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--fresh', action='store_true', help='Retrain in a new isolated directory; preserve supplied artefacts.')
    args = parser.parse_args()
    if args.fresh:
        destination = ROOT / '.runtime' / ('reproduce-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        destination.mkdir(parents=True)
        for folder in ['code', 'data', 'assets']:
            shutil.copytree(ROOT / folder, destination / folder,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copytree(ROOT / 'results/midterm', destination / 'results/midterm')
        (destination / 'results/endterm').mkdir(parents=True)
        shutil.copy2(ROOT / 'results/endterm/experiment_summary.json',
                     destination / 'results/endterm/experiment_summary.json')
        (destination / 'models').mkdir()
        shutil.copy2(ROOT / 'models/endterm_model.joblib', destination / 'models/endterm_model.joblib')
        print('Fresh reproduction directory:', destination, flush=True)
        subprocess.run([sys.executable, str(destination / 'code/run_final.py'),
                        '--workers', str(args.workers)], cwd=destination, check=True)
        return
    if args.verify_only:
        run('verify_final.py')
        return
    run('download_text_model.py')
    run('audit_final_data.py')
    for family in ['random_forest', 'gradient_boosting', 'mlp']:
        run('final_experiments.py', ['--family', family, '--workers', str(args.workers)])
    for filename in ['text_experiments.py', 'semantic_experiments.py', 'target_experiments.py']:
        run(filename, ['--workers', str(args.workers)])
    run('final_analysis.py')
    run('verify_final.py')


if __name__ == '__main__':
    main()
