"""Reproduce the Endterm notebook and HTML from a fresh local Python kernel."""
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for name,sub in [('MPLCONFIGDIR','matplotlib'),('IPYTHONDIR','ipython'),('JUPYTER_RUNTIME_DIR','jupyter')]:
    folder=ROOT/'.runtime'/sub
    folder.mkdir(parents=True,exist_ok=True)
    os.environ[name]=str(folder)
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                  VECLIB_MAXIMUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
                  MPLBACKEND='module://matplotlib_inline.backend_inline')
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager

path=ROOT/'Endterm_TVSeries.ipynb'
nb=nbformat.read(path,as_version=4)
km=KernelManager(kernel_name='python3')
km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client=NotebookClient(nb,km=km,timeout=1800,allow_errors=False,
                      resources={'metadata':{'path':str(ROOT)}})
try:
    client.execute()
finally:
    if km.has_kernel: km.shutdown_kernel(now=True)
assert not any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
nbformat.write(nb,path)
html,_=HTMLExporter(template_name='lab').from_notebook_node(nb)
(ROOT/'Endterm_TVSeries.html').write_text(html)
print('Completed all Endterm cells. Notebook and HTML updated.')
