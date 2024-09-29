import gradio as gr
import pickle
import numpy as np

# Load your trained model
with open("app/model.pkl", "rb") as f:  # Adjusted the path to model.pkl
    model = pickle.load(f)

# Define a function to make predictions
def predict(feature1, feature2, feature3, feature4, feature5):
    features = np.array([feature1, feature2, feature3, feature4, feature5]).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.inputs.Number(label="Feature 1"),
        gr.inputs.Number(label="Feature 2"),
        gr.inputs.Number(label="Feature 3"),
        gr.inputs.Number(label="Feature 4"),
        gr.inputs.Number(label="Feature 5")
    ],
    outputs="text",
    title="Penguin Classifier"
)

if __name__ == "__main__":
    interface.launch()


