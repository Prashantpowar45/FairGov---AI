import numpy as np
import pandas as pd
from .rules import FEATURES, check, ROOT

def make_dataset(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        valid = i % 2 == 0
        r = {
            "annual_income": int(rng.integers(60000, 780000 if valid else 1100000)),
            "maharashtra_domicile": 1,
            "indian_national": 1,
            "cap_admission": 1,
            "bonafide": 1,
            "professional_course": 1,
            "private_or_deemed_university": 0,
            "other_scholarship": 0,
            "family_beneficiaries": int(rng.choice([0,1,2])),
            "attendance_pct": int(rng.integers(50,101)),
            "first_semester": int(rng.integers(0,2)),
            "gap_years": int(rng.choice([0,1]))
        }
        if not valid:
            f = rng.choice(["annual_income","maharashtra_domicile","indian_national","cap_admission","bonafide","professional_course","private_or_deemed_university","other_scholarship","family_beneficiaries","attendance_pct","gap_years"])
            r[f] = {
                "annual_income": int(rng.integers(800001,1500000)),
                "maharashtra_domicile": 0, "indian_national": 0, "cap_admission": 0,
                "bonafide": 0, "professional_course": 0, "private_or_deemed_university": 1,
                "other_scholarship": 1, "family_beneficiaries": 3,
                "attendance_pct": int(rng.integers(10,50)), "gap_years": 2
            }[f]
            if f == "attendance_pct":
                r["first_semester"] = 0
        r["application_id"] = f"SYN{i+1:05d}"
        r["eligible"] = int(check(r)["preliminary_eligible"])
        rows.append(r)
    df = pd.DataFrame(rows)[["application_id"] + FEATURES + ["eligible"]]
    (ROOT / "data").mkdir(exist_ok=True)
    df.to_csv(ROOT / "data/synthetic_applicants.csv", index=False)
    return df

if __name__ == "__main__":
    print(make_dataset()["eligible"].value_counts().to_dict())
