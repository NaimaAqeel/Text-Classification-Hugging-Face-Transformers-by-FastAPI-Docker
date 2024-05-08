import gradio as gr

def sentiment_analysis(text):
    # Your sentiment analysis logic here
    return "Positive" if text.lower().count("happy") > text.lower().count("sad") else "Negative"

iface = gr.Interface(fn=sentiment_analysis, inputs="text", outputs="text")
iface.launch()