import os
import uvicorn
from fastapi import FastAPI
from transformers import pipeline
import gradio as gr

# Specify the cache directory
cache_dir = "Sentiment_Analysis/__pycache__"

# Set the environment variable TRANSFORMERS_CACHE
os.environ["TRANSFORMERS_CACHE"] = cache_dir

# Load the pre-trained sentiment analysis model
sentiment_analysis_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", revision="af0f99b", cache_dir=cache_dir)

app = FastAPI()

# Define the function for sentiment analysis
def predict_sentiment(text):
    # Perform sentiment analysis on the input text
    result = sentiment_analysis_pipeline(text)
    # Combine sentiment label and score into a single string
    sentiment_output = f"Sentiment: {result[0]['label']}, Score: {result[0]['score']}"
    return sentiment_output

# Create Gradio interface
iface = gr.Interface(
    fn=predict_sentiment, 
    inputs="text", 
    outputs="text",
    title="Sentiment Analysis",
    description="Enter some text to analyze its sentiment."
)

# Provide examples for the interface
iface.examples = [
    ["I love coding!"],
    ["I hate Mondays..."]
]

# Launch the FastAPI server
@app.get("/")
def read_root():
    return {"message": "Welcome to Sentiment Analysis API!"}

@app.post("/predict")
async def predict(text: str):
    sentiment = predict_sentiment(text)
    return {"sentiment": sentiment}

# Launch the FastAPI server using uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)





