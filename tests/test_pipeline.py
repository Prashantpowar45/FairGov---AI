from src.rules import check
from src.rag import retrieve

def student():
    return dict(annual_income=600000,maharashtra_domicile=1,indian_national=1,cap_admission=1,bonafide=1,professional_course=1,private_or_deemed_university=0,other_scholarship=0,family_beneficiaries=1,attendance_pct=75,first_semester=0,gap_years=0)

def test_rules_and_boundaries():
    s=student()
    assert check(s)["preliminary_eligible"]
    s["annual_income"]=800001
    assert not check(s)["preliminary_eligible"]
    s=student()
    s["attendance_pct"]=49
    assert not check(s)["preliminary_eligible"]
    s["first_semester"]=1
    assert check(s)["preliminary_eligible"]

def test_retrieve():
    assert retrieve("income limit family 800000")["results"]
