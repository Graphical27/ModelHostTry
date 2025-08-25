# Streamlit client to test the FastAPI model
# Run API first: uvicorn app:app --host 0.0.0.0 --port 8000
# Then run: streamlit run streamlit_app.py

import io
import json
import requests
from PIL import Image
import streamlit as st

API_URL = " http://127.0.0.1:8000"


st.set_page_config(page_title="Plant Disease Classifier", layout="centered")
st.title("🌿 Plant Disease Classifier")
st.caption("Upload an image to get a prediction from the FastAPI service.")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API base URL", API_URL)
    health_btn = st.button("Check API Health")
    if health_btn:
        try:
            r = requests.get(f"{api_url}/health", timeout=10)
            st.write(r.json())
        except Exception as e:
            st.error(f"Health check failed: {e}")

uploaded = st.file_uploader("Choose a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    # Preview
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        try:
            uploaded.seek(0)
            files = {"file": (uploaded.name, uploaded.read(), uploaded.type or "image/jpeg")}
            r = requests.post(f"{api_url}/predict", files=files, timeout=60)
            if r.status_code != 200:
                st.error(f"API error {r.status_code}: {r.text}")
            else:
                data = r.json()
                st.subheader("Top-1 Prediction")
                st.write(f"Label: {data['top1']['label']}")
                st.write(f"Confidence: {data['top1']['confidence']:.4f}")

                st.subheader("Top-5")
                for i, item in enumerate(data.get("top5", []), start=1):
                    st.write(f"{i}. {item['label']} — {item['confidence']:.4f}")

                if st.checkbox("Show raw probabilities"):
                    st.json(json.loads(json.dumps(data.get("raw", []))))
        except Exception as e:
            st.error(f"Prediction failed: {e}")