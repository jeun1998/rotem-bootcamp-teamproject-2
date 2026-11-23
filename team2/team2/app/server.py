
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib,pandas as pd
from flask import Flask,render_template,request,jsonify
from src.core import OUT,prepare,scale_prob
app=Flask(__name__)
app.config["MAX_CONTENT_LENGTH"]=20*1024*1024
cache={}
def load_models():
    if not cache:
        meta=json.loads((OUT/"metadata.json").read_text(encoding="utf-8"))
        cache["meta"]=meta
        cache["main"]=joblib.load(OUT/"models"/(meta["winner"]+".joblib"))
        cache["fallback"]=joblib.load(OUT/"models"/"no_weather.joblib")
    return cache
@app.get("/")
def index():
    return render_template("index.html")
@app.get("/api/metrics")
def metrics():
    path=OUT/"test_metrics.json"
    if not path.exists():
        return jsonify(error="Run evaluation first"),404
    return jsonify(json.loads(path.read_text(encoding="utf-8")))
@app.post("/api/predict")
def predict():
    f=request.files.get("file")
    if f is None:
        return jsonify(error="CSV required"),400
    try:
        df=pd.read_csv(f.stream,nrows=100)
        obj=load_models()
        meta=obj["meta"]
        fallback=meta["no_weather"]
        if all(c in df.columns for c in meta["features"]):
            fields=meta["features"];cats=meta["categorical"]
            model=obj["main"];t=meta["temperature"];mode="weather"
        elif all(c in df.columns for c in fallback["features"]):
            fields=fallback["features"];cats=fallback["categorical"]
            model=obj["fallback"];t=fallback["temperature"];mode="no_weather"
        else:
            return jsonify(error="Required feature columns missing"),400
        x=prepare(df,fields,cats,meta["elapsed_route_medians"],meta["elapsed_median"])
        proba=scale_prob(model.predict_proba(x),t)
        return jsonify(mode=mode,results=[{"predicted":int(p.argmax()),"normal":float(p[0]),"delay":float(p[1]),"cancel":float(p[2])} for p in proba])
    except Exception as exc:
        return jsonify(error=str(exc)),400
if __name__=="__main__":
    app.run(port=5000,debug=False)
