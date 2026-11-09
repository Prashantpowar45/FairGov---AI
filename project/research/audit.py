import pandas as pd
from src.rules import check,FEATURES

def audit_cases(model, inputs):
    X=pd.DataFrame(inputs)[FEATURES]
    ml=model.predict(X)
    rows=[]
    for record,pred in zip(inputs,ml):
        rules=check(record)
        rows.append({'ml_prediction':bool(pred),'rule_preliminary_eligible':rules['preliminary_eligible'],'mismatch':bool(pred)!=rules['preliminary_eligible'],'failed_rules':', '.join(rules['failed_rules'])})
    return pd.DataFrame(rows)

def group_metric_demo(df,model,group='maharashtra_domicile'):
    # Not a real-world fairness conclusion; labels and subgroup were synthetically generated.
    X=df[FEATURES]
    out=df[[group,'eligible']].copy()
    out['predicted']=model.predict(X)
    return out.groupby(group).agg(n=('eligible','size'),positive_prediction_rate=('predicted','mean'),reference_positive_rate=('eligible','mean')).reset_index()
