import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,roc_curve,ConfusionMatrixDisplay
from xgboost import XGBClassifier
from src.rules import ROOT,FEATURES

def tune():
    df=pd.read_csv(ROOT/'data/synthetic_applicants.csv')
    X=df[FEATURES]; y=df.eligible
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    grid=GridSearchCV(XGBClassifier(n_estimators=80,random_state=42,n_jobs=1,eval_metric='logloss'),{'max_depth':[2,4],'learning_rate':[.05,.15]},scoring='f1',cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42),n_jobs=1)
    grid.fit(X_train,y_train)
    model=grid.best_estimator_
    p=model.predict(X_test); prob=model.predict_proba(X_test)[:,1]
    info={'best_params':grid.best_params_,'cv_f1':float(grid.best_score_),'holdout_auc':float(roc_auc_score(y_test,prob)),'classification_report':classification_report(y_test,p,output_dict=True),'confusion_matrix':confusion_matrix(y_test,p).tolist()}
    return model,X_test,y_test,info
