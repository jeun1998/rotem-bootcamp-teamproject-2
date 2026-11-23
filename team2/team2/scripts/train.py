
from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,joblib
from src.core import OUT,features,load,prepare,create_model,tune,scale_prob,evaluate

p=argparse.ArgumentParser()
p.add_argument("--max-train-rows",type=int,default=200000)
p.add_argument("--feature-set",default="base_plus_candidates")
args=p.parse_args()
fields,cats=features(args.feature_set)
no_fields,no_cats=features("no_weather")
all_fields=list(dict.fromkeys(fields+no_fields))
train=load("train",all_fields,args.max_train_rows)
valid=load("validation",all_fields)
route_med={}
overall=120.
if "CRSElapsedTime" in train:
    raw=pd.to_numeric(train["CRSElapsedTime"],errors="coerce")
    overall=float(raw.median())
    if "route" in train:
        route_med=pd.DataFrame({"route":train["route"].astype(str),"value":raw}).groupby("route")["value"].median().dropna().to_dict()
OUT.mkdir(exist_ok=True)
(OUT/"models").mkdir(exist_ok=True)

def experiment(name,fields,cats):
    x=prepare(train,fields,cats,route_med,overall)
    xv=prepare(valid,fields,cats,route_med,overall)
    y=train["label"].to_numpy(dtype=int)
    yv=valid["label"].to_numpy(dtype=int)
    weights=train["sample_weight"].to_numpy(dtype=float)
    model=create_model(name,fields,cats)
    print("[TRAIN]",name,len(x),flush=True)
    model.fit(x,y,model__sample_weight=weights)
    raw=model.predict_proba(xv)
    temperature=tune(raw,yv)
    result=evaluate(yv,scale_prob(raw,temperature))
    return model,temperature,result

results=[]
temps={}
best=None
best_value=-1
for name in ("xgboost","random_forest","logistic"):
    model,t,result=experiment(name,fields,cats)
    result.update({"model":name,"temperature":t,"feature_set":args.feature_set})
    results.append(result)
    temps[name]=t
    joblib.dump(model,OUT/"models"/(name+".joblib"))
    print("[VAL]",name,result["cancel_pr_auc_ap"],result["macro_f1"],flush=True)
    if result["cancel_pr_auc_ap"]>best_value:
        best_value=result["cancel_pr_auc_ap"]
        best=name

backup,t_backup,res_backup=experiment("xgboost",no_fields,no_cats)
joblib.dump(backup,OUT/"models"/"no_weather.joblib")
pd.DataFrame([{k:v for k,v in r.items() if k!="confusion_matrix"} for r in results]).to_csv(OUT/"validation_comparison.csv",index=False)
meta={"winner":best,"temperature":temps[best],"feature_set":args.feature_set,"features":fields,"categorical":cats,"no_weather":{"temperature":t_backup,"features":no_fields,"categorical":no_cats},"elapsed_median":overall,"elapsed_route_medians":route_med,"training_correction":"sample_weight only","data_version":"processed_v3"}
(OUT/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
(OUT/"validation_results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
print("[WINNER]",best)
