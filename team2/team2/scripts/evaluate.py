
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd,numpy as np,joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from src.core import OUT,load,prepare,scale_prob,evaluate
meta=json.loads((OUT/"metadata.json").read_text(encoding="utf-8"))
fields=meta["features"]
fallback=meta["no_weather"]["features"]
test=load("test",list(dict.fromkeys(fields+fallback)))
y=test["label"].to_numpy(dtype=int)

def predict(name,fields,cats,temp):
    model=joblib.load(OUT/"models"/(name+".joblib"))
    x=prepare(test,fields,cats,meta["elapsed_route_medians"],meta["elapsed_median"])
    prob=scale_prob(model.predict_proba(x),temp)
    return evaluate(y,prob),prob

score,p=predict(meta["winner"],fields,meta["categorical"],meta["temperature"])
nw=meta["no_weather"]
nw_score,_=predict("no_weather",nw["features"],nw["categorical"],nw["temperature"])
(OUT/"test_metrics.json").write_text(json.dumps({"main":score,"no_weather":nw_score,"selected":meta["winner"]},ensure_ascii=False,indent=2),encoding="utf-8")
pd.DataFrame({"flight_id":test["flight_id"],"actual":y,"predicted":p.argmax(axis=1),"p_normal":p[:,0],"p_delay":p[:,1],"p_cancel":p[:,2]}).to_csv(OUT/"test_predictions.csv",index=False)
plot=OUT/"plots"
plot.mkdir(exist_ok=True)
cm=np.array(score["confusion_matrix"])
fig,ax=plt.subplots(figsize=(7,5))
ax.imshow(cm)
ax.set_xticks(range(3),["Normal","Delay","Cancel"])
ax.set_yticks(range(3),["Normal","Delay","Cancel"])
for i in range(3):
    for j in range(3):
        ax.text(j,i,str(cm[i,j]),ha="center",va="center")
fig.tight_layout()
fig.savefig(plot/"confusion_matrix.png",dpi=160)
plt.close(fig)
fig,ax=plt.subplots(figsize=(7,5))
for i,name in enumerate(("Normal","Delay","Cancel")):
    actual,pred=calibration_curve((y==i).astype(int),p[:,i],n_bins=10,strategy="quantile")
    ax.plot(pred,actual,marker="o",label=name)
ax.plot([0,1],[0,1],"--")
ax.legend()
fig.tight_layout()
fig.savefig(plot/"calibration.png",dpi=160)
plt.close(fig)
print("[TEST]",json.dumps(score,indent=2))
