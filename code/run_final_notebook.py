"""Execute the Final walkthrough with this Python environment and export HTML."""
import os
from pathlib import Path
import sys

os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / 'Final_TVSeries.ipynb'
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=600, kernel_name='python3',
                   resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.write(notebook, path)
    html, _ = HTMLExporter().from_notebook_node(notebook)
    (ROOT / 'Final_TVSeries.html').write_text(html, encoding='utf-8')
    print('Executed every Final walkthrough cell and exported HTML.')


if __name__ == '__main__':
    main()
