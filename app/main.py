from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Charger le modèle et le vecteur
model = pickle.load(open("svm_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

app = FastAPI()

class Tweet(BaseModel):
    text: str

@app.post("/predict/")
def predict(tweet: Tweet):
    X_input = vectorizer.transform([tweet.text])
    prediction = model.predict(X_input)
    result = "Suspect" if prediction[0] == 0 else "Non Suspect"
    return {"text": tweet.text, "prediction": result}
