import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.rules import ROOT, FEATURES,check

def overview():
    df=pd.read_csv(ROOT/'data/synthetic_applicants.csv')
    assert all(c in df.columns for c in FEATURES+['eligible'])
    report={'rows':len(df),'features':len(FEATURES),'nulls':int(df.isna().sum().sum()),'duplicates':int(df.application_id.duplicated().sum()),'labels':df.eligible.value_counts().to_dict()}
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    df.eligible.value_counts().sort_index().plot.bar(ax=axes[0],title='Synthetic label balance')
    df.groupby('eligible').annual_income.plot.hist(ax=axes[1],alpha=.55,legend=True,bins=22,title='Income by synthetic label')
    fig.tight_layout()
    return df,report,fig
