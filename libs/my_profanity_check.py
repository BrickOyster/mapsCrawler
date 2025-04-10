import numpy as np
import joblib
import os

vectorizer_path = os.path.join(os.path.dirname(__file__), 'data/vectorizer.joblib')
model_path = os.path.join(os.path.dirname(__file__), 'data/model.joblib')
vectorizer = joblib.load(vectorizer_path)
model = joblib.load(model_path)

def _get_profane_prob(prob):
  return prob[1]

def predict(texts):
  return model.predict(vectorizer.transform(texts))

def predict_prob(texts):
  return np.apply_along_axis(_get_profane_prob, 1, model.predict_proba(vectorizer.transform(texts)))
