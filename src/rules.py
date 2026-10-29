import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = json.loads((ROOT / "policies/dte_rules.json").read_text(encoding="utf-8"))
FEATURES = ["annual_income","maharashtra_domicile","indian_national","cap_admission","bonafide","professional_course","private_or_deemed_university","other_scholarship","family_beneficiaries","attendance_pct","first_semester","gap_years"]

def check(record):
    checks = {
      "Indian nationality": bool(record["indian_national"]),
      "Maharashtra domicile": bool(record["maharashtra_domicile"]),
      "Family income <= INR 800,000": float(record["annual_income"]) <= 800000,
      "CAP admission": bool(record["cap_admission"]),
      "Bonafide student": bool(record["bonafide"]),
      "Technical/professional course": bool(record["professional_course"]),
      "Not private/deemed university": not bool(record["private_or_deemed_university"]),
      "No other scholarship": not bool(record["other_scholarship"]),
      "At most 2 family beneficiaries": int(record["family_beneficiaries"]) <= 2,
      "Attendance >=50%, or first semester": bool(record["first_semester"]) or float(record["attendance_pct"]) >= 50,
      "Gap less than two years": int(record["gap_years"]) < 2,
    }
    return {"preliminary_eligible": all(checks.values()), "checks": checks, "failed_rules": [key for key,v in checks.items() if not v], "warning": "Demo checks only; incomplete scheme/GR exceptions and document verification. Not an official approval."}
