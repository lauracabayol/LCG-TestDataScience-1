import pickle
import numpy as np
import gradio as gr

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define a prediction function
def predict_penguin(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# Create a Gradio interface
interface = gr.Interface(
    fn=predict_penguin,
    inputs=[
        gr.inputs.Number(label="island"),
        gr.inputs.Number(label="bill_length_mm"),
        gr.inputs.Number(label="bill_depth_mm"),
        gr.inputs.Number(label="flipper_length_mm")
        gr.inputs.Number(label="body_mass_g")
    ],
    outputs="text",
    title="Penguin Classifier",
    description="Enter features to predict the penguin species."
)

# Launch the interface
interface.launch()
