import joblib
import pandas as pd
from .rules import ROOT, FEATURES, check

def predict(student, explain=False):
    model = joblib.load(ROOT / "models/xgboost_dte.joblib")
    X = pd.DataFrame([{key: student[key] for key in FEATURES}])[FEATURES]
    y = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])
    policy = check(student)
    result = {
        "ml_prediction": bool(y),
        "ml_probability_eligible": round(prob,5),
        "rule_engine": policy,
        "mismatch": bool(y) != policy["preliminary_eligible"],
        "disclaimer": "Demonstration prediction only; not an official scholarship decision."
    }
    if explain:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer(X)
        vals = sv.values[0]
        result["shap"] = {
            "base_value": float(sv.base_values[0]),
            "feature_contributions": {k:round(float(v),5) for k,v in zip(FEATURES,vals)},
            "note": "SHAP values explain model scores, not policy/legal reasons."
        }
    return result
