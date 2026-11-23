
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pyarrow.parquet as pq
from src.core import DATA,features

expected={"train":24,"validation":12,"test":12}
fields,_=features("experimental_tail")
good=True
for split,count in expected.items():
    folder=DATA/"ml"/("split="+split)
    paths=sorted(folder.glob("*.parquet")) if folder.exists() else []
    print(split,len(paths),"files (expected",count,")")
    if len(paths)!=count:
        good=False
    rows=0
    for path in paths:
        pf=pq.ParquetFile(path)
        rows+=pf.metadata.num_rows
        missing=set(fields+["label","sample_weight"])-set(pf.schema_arrow.names)
        if missing:
            good=False
            print("Missing columns:",path.name,sorted(missing))
    print("Rows:",rows)
if not good:
    raise SystemExit("Data missing. Copy ml.zip and split archives into data/inbox and run import_uploads.py.")
print("[OK] Data structure checked")
