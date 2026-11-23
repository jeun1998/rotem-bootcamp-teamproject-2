
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parents[1]
for task in ("import_uploads.py","check_data.py","train.py","evaluate.py"):
    print("[RUN]",task,flush=True)
    subprocess.run([sys.executable,str(root/"scripts"/task)],cwd=root,check=True)
print("[COMPLETE]")
