from fastapi import FastAPI, File
from pydantic import BaseModel
from data_model_class import DataPipeline
from fastapi.responses import JSONResponse
import requests
import pickle
import numpy as np
import pandas as pd
from io import StringIO

app = FastAPI()


class InputData(BaseModel):
    reviewText: str


# Load the tokenizer
tokenizer = pickle.load(open('./tokenizer.pickle', 'rb'))

# Create an instance of the data papeline:
pipeline = DataPipeline(tokenizer=tokenizer)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analyzer application!"}


@app.post('/predict')
def predict_sentiment(input_data: InputData):
    text = input_data.reviewText
    text = text.lower()
    text = pipeline.tokenization_punct(text)
    text = pipeline.remove_punctuation(text)
    text = pipeline.remove_stopwords(text)
    text = pipeline.lemmatizer(text)
    text = pipeline.remove_numbers(text)

    textlist = pipeline.tokenize_pad_sentences(pd.Series(text))

    # Model serving predictions:
    payload = {"instances": textlist.tolist()}
    endpoint = "http://127.0.0.1:1234/invocations"
    response = requests.post(endpoint, json=payload)
    probs = eval(response.text)['predictions'][0][0]
    predicted_class = 'POSITIVE' if probs >= 0.5 else 'NEGATIVE'

    # Create the response JSON
    response_data = {"text": input_data.reviewText,
                     "predicted_class": predicted_class,
                     "predicted_probability": probs}

    return JSONResponse(content=response_data)


@app.post("/files/")
async def batch_prediction(file: bytes = File(...)):
    s = str(file, 'utf-8')
    data = StringIO(s)
    movie_reviews = pd.read_csv(data)
    backup_texts = movie_reviews.copy()
    backup_texts = backup_texts.values.tolist()
    backup_texts = [i[0] for i in backup_texts]

    movie_reviews['reviewText'] = movie_reviews['reviewText'].apply(
        lambda x: x.lower())
    movie_reviews['reviewText'] = movie_reviews['reviewText'].apply(
        lambda x: pipeline.tokenization_punct(x))
    movie_reviews['reviewText'] = movie_reviews['reviewText'].apply(
        lambda x: pipeline.remove_punctuation(x))
    movie_reviews['reviewText'] = movie_reviews['reviewText'].apply(
        lambda x: pipeline.remove_stopwords(x))
    movie_reviews['reviewText'] = movie_reviews['reviewText'].apply(
        lambda x: pipeline.lemmatizer(x))
    movie_reviews['reviewText'] = movie_reviews['reviewText'].apply(
        lambda x: pipeline.remove_numbers(x))

    # Tokenize and pad sentences:
    movie_reviews = pipeline.tokenize_pad_sentences(
        pd.Series(movie_reviews['reviewText']))

    # Model serving predictions:
    payload = {"instances": movie_reviews.tolist()}
    endpoint = "http://127.0.0.1:1234/invocations"
    response = requests.post(endpoint, json=payload)
    response = eval(response.text)['predictions']
    probs = [i[0] for i in response]
    predicted_class = ['POSITIVE' if i >= 0.5 else 'NEGATIVE' for i in probs]

    # Create the response JSON
    response_data = [{"text": backup_text,
                      "predicted_class": predicted_class,
                      "predicted_probability": probs} for backup_text, predicted_class, probs in zip(backup_texts, predicted_class, probs)]

    return JSONResponse(content=response_data)
