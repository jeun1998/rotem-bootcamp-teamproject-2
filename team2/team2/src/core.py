
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import average_precision_score,f1_score,log_loss,confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"processed_v3"
OUT=ROOT/"outputs"

def features(name="base_plus_candidates"):
    spec=json.loads((DATA/"feature_lists.json").read_text(encoding="utf-8"))
    if name not in ("base","base_plus_candidates","no_weather","with_dest_weather","experimental_tail"):
        raise ValueError("Unknown feature set")
    fields=spec[name]
    blocked=set(["flight_id","label","sample_weight","sample_rate","label_arr"])
    blocked.update(f for f in fields if f.endswith("_oracle") or f.endswith("_sameday") or f.endswith("_proxy_v2_unavailable"))
    if set(fields)&blocked:
        raise ValueError("Forbidden feature leakage")
    return fields,[c for c in spec["categorical"] if c in fields]

def load(split,fields,limit=0):
    folder=DATA/"ml"/("split="+split)
    paths=sorted(folder.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(str(folder))
    columns=list(dict.fromkeys(["flight_id","label","sample_weight"]+list(fields)))
    pieces=[]
    for path in paths:
        frame=pd.read_parquet(path,columns=columns)
        if limit>0:
            each=max(1,int(limit/len(paths)))
            if len(frame)>each:
                frame=frame.sample(each,random_state=42)
        pieces.append(frame)
    data=pd.concat(pieces,ignore_index=True)
    if limit>0 and len(data)>limit:
        data=data.sample(limit,random_state=42).reset_index(drop=True)
    return data

def prepare(frame,fields,categories,route_medians=None,overall=None):
    x=frame[fields].copy()
    for col in fields:
        if col in categories:
            x[col]=x[col].fillna("_MISSING_").astype(str)
        else:
            x[col]=pd.to_numeric(x[col],errors="coerce").astype("float32")
    if "CRSElapsedTime" in x:
        if route_medians and "route" in x:
            replacement=x["route"].map(route_medians)
            x["CRSElapsedTime"]=x["CRSElapsedTime"].fillna(replacement)
        x["CRSElapsedTime"]=x["CRSElapsedTime"].fillna(overall if overall is not None else 120)
    return x

def create_model(name,fields,categories):
    nums=[f for f in fields if f not in categories]
    if name=="logistic":
        pre=ColumnTransformer([
            ("cats",OneHotEncoder(handle_unknown="ignore",min_frequency=30,dtype=np.float32),categories),
            ("nums",Pipeline([
                ("impute",SimpleImputer(strategy="median")),
                ("scale",StandardScaler())
            ]),nums)
        ])
        model=LogisticRegression(solver="saga",max_iter=250,random_state=42)
    else:
        cat_pipe=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1,dtype=np.float32)
        numeric="passthrough" if name=="xgboost" else SimpleImputer(strategy="median")
        pre=ColumnTransformer([("cats",cat_pipe,categories),("nums",numeric,nums)])
        if name=="xgboost":
            model=XGBClassifier(objective="multi:softprob",num_class=3,n_estimators=180,max_depth=6,learning_rate=0.07,tree_method="hist",n_jobs=4,random_state=42)
        else:
            model=RandomForestClassifier(n_estimators=90,max_depth=17,min_samples_leaf=8,n_jobs=4,random_state=42)
    return Pipeline([("pre",pre),("model",model)])

def scale_prob(p,t):
    p=np.clip(np.asarray(p,dtype=float),1e-12,1)
    return softmax(np.log(p)/t,axis=1)

def tune(p,y):
    def objective(t):
        return log_loss(y,scale_prob(p,t),labels=[0,1,2])
    result=minimize_scalar(objective,bounds=(0.6,3),method="bounded")
    return float(result.x)

def evaluate(y,p):
    y=np.asarray(y,dtype=int)
    pred=p.argmax(axis=1)
    onehot=np.eye(3)[y]
    return {
        "rows":len(y),
        "macro_f1":float(f1_score(y,pred,labels=[0,1,2],average="macro",zero_division=0)),
        "delay_pr_auc_ap":float(average_precision_score((y==1).astype(int),p[:,1])),
        "cancel_pr_auc_ap":float(average_precision_score((y==2).astype(int),p[:,2])),
        "log_loss":float(log_loss(y,p,labels=[0,1,2])),
        "brier":float(np.mean(np.sum((p-onehot)**2,axis=1))),
        "confusion_matrix":confusion_matrix(y,pred,labels=[0,1,2]).tolist()
    }
