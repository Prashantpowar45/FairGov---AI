import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from .rules import ROOT, FEATURES

def train():
    df = pd.read_csv(ROOT / "data/synthetic_applicants.csv")
    X = df[FEATURES]
    y = df["eligible"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    model = XGBClassifier(n_estimators=170,max_depth=4,learning_rate=.08,subsample=.9,colsample_bytree=.95,random_state=42,n_jobs=2,eval_metric="logloss")
    model.fit(X_train,y_train)
    p = model.predict(X_test)
    result = {
        "dataset_type": "synthetic-policy-derived",
        "train_records": len(X_train),
        "test_records": len(X_test),
        "accuracy": accuracy_score(y_test,p),
        "precision": precision_score(y_test,p,zero_division=0),
        "recall": recall_score(y_test,p,zero_division=0),
        "f1": f1_score(y_test,p,zero_division=0),
        "confusion_matrix": confusion_matrix(y_test,p).tolist(),
        "warning": "Synthetic-only test metrics cannot establish real-world scholarship performance."
    }
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models/xgboost_dte.joblib")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/model_evaluation.json").write_text(json.dumps(result,indent=2))
    return result
