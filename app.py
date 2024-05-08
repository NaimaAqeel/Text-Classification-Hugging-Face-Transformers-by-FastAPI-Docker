import gradio as gr
from transformers import pipeline

# Load the sentiment analysis model
sentiment_analysis_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

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

# Launch the interface
iface.launch()
