# Customer Churn Prediction API
 
A REST API built with **FastAPI** and **scikit-learn** that predicts whether a SaaS customer is likely to churn, based on subscription tenure, usage, and plan type. The API selects its production model through cross-validated comparison of three classic ML algorithms rather than committing to a single one upfront.
 
## Features
 
- Categorical feature encoding (plan type) with a consistent, reusable encoding function — new incoming requests are guaranteed to match the exact column structure the model was trained on
- Model selection via **5-fold cross-validation** (not a single train/test split) across three algorithms: Logistic Regression, Decision Tree, and Random Forest, compared on F1 score
- Batch prediction endpoint — score multiple customers in a single request
- A transparency endpoint that exposes the full cross-validation comparison, not just the winning model
## Tech Stack
 
- Python 3
- FastAPI + Pydantic (typed request validation)
- scikit-learn (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier)
- Pandas
## Installation & Running Locally
 
```bash
pip install fastapi uvicorn pandas scikit-learn
uvicorn CustomerChurnPredictionAPI:app --reload
```
 
Visit `http://127.0.0.1:8000/docs` for interactive Swagger documentation.
 
## API Reference
 
### `GET /model-comparison`
 
Returns the full 5-fold cross-validation results for all three candidate models, plus which one was selected for serving predictions.
 
**Response**
 
```json
{
  "cv_folds": 5,
  "scoring_metric": "f1",
  "result_per_model": {
    "LogisticRegression": {"f1_per_fold": [1.0, 1.0, 1.0, 1.0, 0.0], "f1_mean": 0.8, "f1_std": 0.4},
    "DecisionClassifier": {"f1_per_fold": [1.0, 1.0, 1.0, 1.0, 0.667], "f1_mean": 0.933, "f1_std": 0.133},
    "ForestClassifier": {"f1_per_fold": [1.0, 1.0, 1.0, 1.0, 0.667], "f1_mean": 0.933, "f1_std": 0.133}
  },
  "selected_model": "DecisionClassifier",
  "reason_selection": "rata-rata F1 cross-validation tertinggi"
}
```
 
### `POST /predict-churn`
 
Accepts a list of one or more customers and returns a churn prediction with probability for each.
 
**Request body**
 
```json
[
  {"lama_berlangganan_bulan": 8, "rata_pemakaian_jam": 19, "jenis_paket": "Premium"},
  {"lama_berlangganan_bulan": 2, "rata_pemakaian_jam": 5, "jenis_paket": "Basic"}
]
```
 
**Response**
 
```json
{
  "status": "success",
  "total_data": 2,
  "results": [
    {
      "data_input": {"lama_berlangganan_bulan": 8, "rata_pemakaian_jam": 19, "jenis_paket": "Premium"},
      "churn_prediction": 0,
      "probability_no_churn": 1.0,
      "probability_churn": 0.0
    },
    {
      "data_input": {"lama_berlangganan_bulan": 2, "rata_pemakaian_jam": 5, "jenis_paket": "Basic"},
      "churn_prediction": 1,
      "probability_no_churn": 0.0,
      "probability_churn": 1.0
    }
  ]
}
```
 
## Lessons Learned
 
- **Consistent encoding between training and inference is a real production hazard.** One-hot encoding a categorical column during training produces a fixed set of columns in a fixed order. If new incoming data is encoded independently at request time, there's no guarantee the resulting columns match — a subtle bug that can silently corrupt predictions rather than throw an error. The fix here was to store the training-time column list once and `reindex()` every new request against it, regardless of which categories happen to be present in that particular request.
- **A tie in cross-validation scores isn't a "winner."** The Decision Tree and Random Forest models scored an identical mean F1 (0.933) across 5 folds. The model-selection code (`max()` over a dict) still had to pick one, but the honest conclusion is that both performed equivalently on this data — not that one algorithm proved superior. Any downstream claim about "why this model was chosen" needs to reflect that nuance.
- **High-confidence predictions on out-of-distribution inputs deserve skepticism, not trust.** Every "Enterprise" customer in the training data had both high tenure and high usage, always paired with `churn=0`. When queried with an "Enterprise" customer with unusually low tenure and usage — a combination never seen in training — the model returned a 100% churn probability. This is a model extrapolating with false confidence outside the pattern it actually learned, not a considered judgment about that specific edge case.
- **Small training data (16 rows) limits how much any of the above can be trusted at face value.** The evaluation methodology (cross-validation, consistent encoding) is sound, but the sample size itself is the binding constraint on how confidently these results generalize.
## Possible Improvements
 
- Persist the trained model with `joblib` instead of retraining on every server startup — important once training data grows large or training time becomes non-trivial
- Surface the training sample size and a confidence caveat directly in the `/model-comparison` response
- Add automated tests (`pytest`) covering the encoding consistency logic and both endpoints
- Add a confidence threshold or "low-confidence" flag for predictions on inputs that fall far outside the training data's distribution
