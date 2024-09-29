import gradio as gr
import pickle
import numpy as np

# Load your trained model
with open("model.pkl", "rb") as f:  # Adjusted the path to model.pkl
    model = pickle.load(f)

# Define a function to make predictions
def predict(feature1, feature2, feature3, feature4, feature5):
    features = np.array([feature1, feature2, feature3, feature4, feature5]).reshape(1, -1)
    prediction = model.predict(features)
    #add here conversion to penguin type
    return prediction[0]

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.inputs.Number(label="island"),
        gr.inputs.Number(label="bill_length_mm"),
        gr.inputs.Number(label="bill_depth_mm"),
        gr.inputs.Number(label="flipper_length_mm"),
        gr.inputs.Number(label="body_mass_g")
    ],
    outputs="text",
    title="Penguin Classifier"
)

if __name__ == "__main__":
    interface.launch()
