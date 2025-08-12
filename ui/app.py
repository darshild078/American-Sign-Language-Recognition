import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Load MobileNetV2 model from checkpoints folder once when app starts
@st.cache_resource
def load_model():
    model_path = os.path.join('..', 'checkpoints', 'best_mobilenet.h5')
    return tf.keras.models.load_model(model_path)

model = load_model()

# Classes (must match your training class indices order)
class_names = [
  'A','B','C','D','E','F','G','H','I','J','K','L','M',
  'N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space'
]

st.title("ASL Alphabet Sentence Builder")

st.write("Upload images of ASL alphabets, one by one. Predictions will be appended to form sentences.")

# Initialize session state for sentence storage
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload ASL Image (jpg/png)", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", width=300)

        # Preprocess image for MobileNetV2 (128x128 RGB)
        image = image.resize((128, 128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict
        preds = model.predict(img_array)
        pred_class = class_names[np.argmax(preds)]
        st.success(f"Predicted alphabet: {pred_class}")

        # Handle special class "del" and "space"
        if pred_class == "space":
            st.session_state.sentence += " "
        elif pred_class == "del":
            if len(st.session_state.sentence) > 0:
                st.session_state.sentence = st.session_state.sentence[:-1]
        elif pred_class == "nothing":
            # Do not append anything if 'nothing' predicted
            pass
        else:
            st.session_state.sentence += pred_class

with col2:
    st.text_area("Sentence:", value=st.session_state.sentence, height=200)

clear_col, backspace_col = st.columns([1,1])
with clear_col:
    if st.button("Clear Sentence"):
        st.session_state.sentence = ""

with backspace_col:
    if st.button("Backspace"):
        if len(st.session_state.sentence) > 0:
            st.session_state.sentence = st.session_state.sentence[:-1]
