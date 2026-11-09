import pandas as pd
from src.rules import ROOT,check,FEATURES

def independently_specified_cases():
    base=dict(annual_income=600000,maharashtra_domicile=1,indian_national=1,cap_admission=1,bonafide=1,professional_course=1,private_or_deemed_university=0,other_scholarship=0,family_beneficiaries=1,attendance_pct=80,first_semester=0,gap_years=0)
    specs=[('baseline',{},True),('income_at_cap',{'annual_income':800000},True),('income_exceeds',{'annual_income':800001},False),('cap_missing',{'cap_admission':0},False),('attendance_50',{'attendance_pct':50},True),('attendance_49',{'attendance_pct':49},False),('first_semester',{'attendance_pct':49,'first_semester':1},True),('gap_2',{'gap_years':2},False),('other_scholarship',{'other_scholarship':1},False)]
    return pd.DataFrame([{'case':name,'expected':expected,'actual':check({**base,**changes})['preliminary_eligible']} for name,changes,expected in specs]).assign(pass_check=lambda x:x.expected==x.actual)

def evaluate_dataset():
    df=pd.read_csv(ROOT/'data/synthetic_applicants.csv')
    assert len(df)>0 and set(df.eligible)=={0,1}
    assert df[FEATURES].notna().all().all()
    assert df.application_id.is_unique
    assert df.annual_income.ge(0).all() and df.attendance_pct.between(0,100).all()
    report=independently_specified_cases()
    assert report.pass_check.all()
    return report
