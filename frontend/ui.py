import streamlit as st
import requests
from PIL import Image
import io

# Config
API_URL = "http://localhost:8000/detect"

st.set_page_config(page_title="SafetyLens MLOps", layout="wide")

st.title("🛡️ SafetyLens: Industrial Inspection System")
st.markdown("Upload an image. If the AI is unsure (<40% confidence), it sends the data to the **Active Learning Pipeline**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Original Image", use_column_width=True)
    
    # Button to trigger prediction
    if st.button("Run Inspection"):
        with st.spinner("Processing via FastAPI..."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                # Display Result
                image_bytes = response.content
                image = Image.open(io.BytesIO(image_bytes))
                with col2:
                    st.image(image, caption="AI Detection Output", use_column_width=True)
                st.success("Processing Complete.")
            else:
                st.error(f"Error: {response.text}")