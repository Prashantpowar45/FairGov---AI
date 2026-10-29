import json
import pandas as pd
from .rules import ROOT, FEATURES, check

def validate():
    df = pd.read_csv(ROOT / "data/synthetic_applicants.csv")
    invalid = []
    for _, r in df.iterrows():
        label = int(check(r.to_dict())["preliminary_eligible"])
        if label != r["eligible"]:
            invalid.append(str(r["application_id"]))
    result = {
        "records": len(df),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_ids": int(df.application_id.duplicated().sum()),
        "invalid_labels": len(invalid),
        "class_counts": {str(k): int(v) for k,v in df.eligible.value_counts().items()},
        "income_range": [int(df.annual_income.min()), int(df.annual_income.max())],
        "note": "Consistency with implemented rules, not real-world or complete legal validity."
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/dataset_validation.json").write_text(json.dumps(result, indent=2))
    assert result["missing_values"] == 0 and result["duplicate_ids"] == 0 and result["invalid_labels"] == 0
    assert set(df.eligible) == {0,1}, "Both classes required"
    return result
