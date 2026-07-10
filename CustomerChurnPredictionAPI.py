import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal



data_pelanggan = {
    "lama_berlangganan_bulan": [2, 24, 3, 36, 1, 18, 4, 30, 2, 20, 5, 15, 1, 28, 6, 10],
    "rata_pemakaian_jam": [5, 45, 8, 60, 3, 40, 10, 55, 4, 42, 12, 35, 2, 50, 15, 25],
    "jenis_paket": ["Basic","Premium","Basic","Enterprise","Basic","Premium","Basic","Enterprise",
                    "Basic","Premium","Basic","Premium","Basic","Enterprise","Basic","Premium"],
    "churn": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data_pelanggan)
y = df["churn"]

x_encoded = pd.get_dummies(df.drop(columns=["churn"]), columns=["jenis_paket"], drop_first=True)
TRAINS_COLUMNS = x_encoded.columns.tolist()

def encode_input(data: pd.DataFrame) -> pd.DataFrame:
    encoded = data.copy()
    encoded["jenis_paket_Enterprise"] = (encoded["jenis_paket"] == "Enterprise").astype(int)
    encoded["jenis_paket_Premium"] = (encoded["jenis_paket"] == "Premium").astype(int)    
    encoded = encoded.drop(columns=["jenis_paket"])
    encoded = encoded.reindex(columns=TRAINS_COLUMNS, fill_value=0)
    return encoded

candidate_models = {
  "LogisticRegression": LogisticRegression(random_state=30),
  "DecisionClassifier": DecisionTreeClassifier(random_state=30),
  "ForestClassifier": RandomForestClassifier(n_estimators=100, random_state=30)
}

model_comparison_results = {}
for name, model in candidate_models.items():
  score = cross_val_score(model, x_encoded, y, cv=5, scoring="f1")
  model_comparison_results[name] = {
    "f1_per_fold" : [round(float(s), 3) for s in score],
    "f1_mean" : round(float(score.mean()), 3),
    "f1_std" : round(float(score.std()), 3)
  }

BEST_MODEL_NAME = max(model_comparison_results, key=lambda k: model_comparison_results[k]["f1_mean"])

best_model = candidate_models[BEST_MODEL_NAME]
best_model.fit(x_encoded, y)

app = FastAPI(title="Customer Churn Prediction API")

class PelangganBaru(BaseModel):
  lama_berlangganan_bulan: int = Field(..., ge=0, description="How many months user been subscribed")
  rata_pemakaian_jam: int = Field(..., ge=0, description="Average monthly usage (hours)")
  jenis_paket: Literal["Basic", "Premium", "Enterprise"]

@app.post("/predict-churn")
def predict_chun(data_input: list[PelangganBaru]):
  data = [pelanggan.model_dump() for pelanggan in data_input]
  df_encoded = encode_input(pd.DataFrame(data))
  predictions = best_model.predict(df_encoded)
  probabilities = best_model.predict_proba(df_encoded)
  results = []
  for i, pelanggan in enumerate(data_input):
      results.append({
          "data_input": pelanggan.model_dump(),
          "churn_prediction": int(predictions[i]),  # Konversi numpy int ke python int
          "probability_no_churn": round(float(probabilities[i][0]), 4),
          "probability_churn": round(float(probabilities[i][1]), 4)
      })
        
  return {"status": "success", "total_data": len(results), "results": results}

@app.get("/model-comparison")
def model_comparison():
    return {
        "cv_folds": 5,
        "scoring_metric": "f1",
        "result_per_model": model_comparison_results,
        "selected_model": BEST_MODEL_NAME,
        "reason_selection": "rata-rata F1 cross-validation tertinggi",
    }
