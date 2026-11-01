"""Выполнить ноутбук и вывести ошибки/таблицы. python tools/run_nb.py nb.ipynb out.ipynb"""
import sys, os, nbformat
from nbclient import NotebookClient
nb = nbformat.read(sys.argv[1], as_version=4)
wd = os.path.dirname(os.path.abspath(sys.argv[2]))
os.makedirs(wd, exist_ok=True)
client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": wd}})
try:
    client.execute()
finally:
    nbformat.write(nb, sys.argv[2])
for i, c in enumerate(nb.cells):
    if c.cell_type != "code" or c.source.startswith("%%writefile"):
        continue
    for o in c.get("outputs", []):
        if o.output_type == "stream":
            print(f"[{i}]", o.text[:1500])
        elif o.output_type in ("execute_result", "display_data") and "text/plain" in o.get("data", {}):
            print(f"[{i}]", o["data"]["text/plain"][:2500])
        elif o.output_type == "error":
            print(f"[{i}] ERROR", o.ename, o.evalue)
