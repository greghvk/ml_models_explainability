# XAI-metrics 
A package for analysis and evaluating metrics for machine learning models explainability.

## Installation
Install from PyPI:
```
pip install xai-metrics
```

## Usage
Examples of usage:

* Perturbation based on permutation importances

```
from xai_metrics import examine_interpretation

X_train.columns = ['0','1','2','3']
X_test.columns = ['0','1','2','3']
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
perm = PermutationImportance(xgb_model, random_state=1).fit(X_test, y_test)
perm_importances = perm.feature_importances_

examine_interpretation(xgb_model, X_test, y_test, perm_importances, epsilon=4, resolution=50, proportionality_mode=0)
```
![Perturbation based on permutation importances](https://raw.githubusercontent.com/hubertsiuzdak/ml_models_explainability/master/examples/img/perturbation.png)

* Perturbation based on local importances
```
from xai_metrics import examine_local_fidelity

examine_local_fidelity(xgb_model, X_test, y_test, epsilon=3)
```
![Perturbation based on permutation importances](https://raw.githubusercontent.com/hubertsiuzdak/ml_models_explainability/master/examples/img/local_fidelity.png)

* Gradual elimination
```
from xai_metrics import gradual_elimination

gradual_elimination(f_forest, f_X_test, f_y_test, f_shap)
```
![Perturbation based on permutation importances](https://raw.githubusercontent.com/hubertsiuzdak/ml_models_explainability/master/examples/img/gradual_elimination.png)

See [here](examples) for notebooks with full examples of usage.

