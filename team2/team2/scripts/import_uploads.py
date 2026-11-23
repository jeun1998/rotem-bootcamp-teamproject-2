
from pathlib import Path
import shutil,zipfile,re,json,filecmp
ROOT=Path(__file__).resolve().parents[1]
INBOX=ROOT/"data"/"inbox"
DATA=ROOT/"data"/"processed_v3"
REF=ROOT/"data"/"reference"
imported=[]
skipped=[]

def move_stream(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        skipped.append(str(target)+" (already exists)")
        return
    with target.open("wb") as f:
        shutil.copyfileobj(source,f,1024*1024)
    imported.append(str(target.relative_to(ROOT)))

def target_for(name,archive=""):
    s=name.replace("\\","/")
    base=s.split("/")[-1]
    if not base.endswith(".parquet"):
        return None
    if "merged/" in s or "interim/" in s or "tail_sameday/" in s:
        return None
    if re.fullmatch(r"part-20(22|23|24|25)-\d\d.parquet",base):
        year=int(base[5:9])
        split="train" if year<=2023 else "validation" if year==2024 else "test"
        if archive.startswith("year=") or archive.startswith("interim"):
            return None
        return DATA/"ml"/("split="+split)/base
    if base.startswith(("hist_","monthly_outcomes_hist_","sched_counts_")) or base=="route_distance.parquet":
        return DATA/"aggregates"/base
    if base=="unmatched_flights.parquet":
        return REF/base
    return None

def main():
    INBOX.mkdir(parents=True,exist_ok=True)
    REF.mkdir(parents=True,exist_ok=True)
    for path in sorted(INBOX.iterdir()):
        if not path.is_file():
            continue
        print("[SCAN]",path.name,flush=True)
        if path.name.startswith("feature_lists") and path.suffix==".json":
            obj=json.loads(path.read_text(encoding="utf-8"))
            if len(obj.get("base_plus_candidates",[]))==70:
                (DATA/"feature_lists.json").write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
            continue
        if path.suffix==".parquet":
            target=target_for(path.name)
            if target:
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    shutil.copy2(path,target)
                    imported.append(str(target.relative_to(ROOT)))
            continue
        if path.suffix!=".zip":
            continue
        if path.name.startswith(("year=","interim","flight_weather_merged")):
            skipped.append(path.name+" (reference archive)")
            continue
        with zipfile.ZipFile(path) as z:
            for item in z.infolist():
                if item.is_dir():
                    continue
                target=target_for(item.filename,path.name)
                if target:
                    with z.open(item) as src:
                        move_stream(src,target)
    (REF/"import_manifest.json").write_text(json.dumps({"imported":imported,"skipped":skipped},ensure_ascii=False,indent=2),encoding="utf-8")
    print("[IMPORTED]",len(imported),"[SKIPPED]",len(skipped))
    for split in ("train","validation","test"):
        path=DATA/"ml"/("split="+split)
        print(split,len(list(path.glob("*.parquet"))) if path.exists() else 0)

if __name__=="__main__":
    main()
