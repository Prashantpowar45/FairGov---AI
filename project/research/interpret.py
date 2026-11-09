import shap
import matplotlib.pyplot as plt

def explain(model, X):
    sample=X.head(min(50,len(X)))
    exp=shap.TreeExplainer(model)(sample)
    shap.plots.beeswarm(exp,show=False)
    plt.tight_layout();plt.show()
    shap.plots.waterfall(exp[0],show=False)
    plt.tight_layout();plt.show()
    return exp
