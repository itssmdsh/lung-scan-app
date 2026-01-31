import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time
import os
import gdown
import matplotlib.cm as cm

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="LungScan AI | Pro",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for Medical UI
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button {
        width: 100%;
        background-color: #2ecc71;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        border: none;
        font-size: 18px;
    }
    .stButton>button:hover { background-color: #27ae60; }
    .report-view {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING (Google Drive Bypass)
# ==========================================
CLASSES = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis']

@st.cache_resource
def load_models():
    # -----------------------------------------------------------
    # ⚠️ IMPORTANT: REPLACE THESE WITH YOUR GOOGLE DRIVE IDs
    # -----------------------------------------------------------
    id_dense = '1aWtU79Xk1Vmrg8BsBL9VgwwZxk6eY4oz' 
    id_res   = '176xn7ZUy1iRllmtPxdcpeWplQ2nJ40sW'   
    
    # 1. Download DenseNet if missing
    if not os.path.exists("Final_DenseNet.keras"):
        with st.spinner("📥 Downloading DenseNet Model (1/2)..."):
            gdown.download(id=id_dense, output="Final_DenseNet.keras", quiet=False)

    # 2. Download ResNet if missing
    if not os.path.exists("Final_ResNet.keras"):
        with st.spinner("📥 Downloading ResNet Model (2/2)..."):
            gdown.download(id=id_res, output="Final_ResNet.keras", quiet=False)

    # 3. Load into Memory
    m1 = tf.keras.models.load_model("Final_DenseNet.keras")
    m2 = tf.keras.models.load_model("Final_ResNet.keras")
    return m1, m2

with st.spinner("🧠 Initializing AI System..."):
    try:
        model_dense, model_res = load_models()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}\n\nMake sure you updated the Google Drive IDs in app.py!")
        st.stop()

# ==========================================
# 3. GRAD-CAM & DEEP SCAN ENGINES
# ==========================================
def generate_120_views(image_pil):
    """Generates 120 augmented views for Deep Scan"""
    img = np.array(image_pil.convert('RGB'))
    img = cv2.resize(img, (224, 224))
    views = []
    h, w = img.shape[:2]
    center = (w//2, h//2)
    
    for angle in range(-14, 15, 2): # 15 angles
        for scale in [1.0, 1.05, 1.10, 1.15]: # 4 scales
            M = cv2.getRotationMatrix2D(center, angle, scale)
            aug = cv2.warpAffine(img, M, (w, h))
            views.append(aug)
            views.append(cv2.flip(aug, 1)) # + Flips
            
    return np.array(views)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_gradcam_overlay(img_pil, model):
    img_array = np.array(img_pil.resize((224, 224))).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Find last conv layer dynamically
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
    heatmap = np.uint8(255 * heatmap)
    jet_heatmap = cm.get_cmap("jet")(np.arange(256))[:, :3]
    jet_heatmap = jet_heatmap[heatmap]
    jet_heatmap = cv2.resize(jet_heatmap, (224, 224))
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap).resize(img_pil.size)
    
    # Superimpose
    original_img = np.array(img_pil)
    superimposed_img = np.array(jet_heatmap) * 0.4 + original_img * 0.6
    return np.uint8(superimposed_img)

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
def run_prediction(image, deep_scan_mode):
    start_time = time.time()
    
    if deep_scan_mode:
        # --- DEEP SCAN (120 Views) ---
        status_text = st.empty()
        bar = st.progress(0)
        status_text.info("🧬 Generating 120 views...")
        
        batch = generate_120_views(image).astype('float32') / 255.0
        chunk_size = 32
        preds = []
        
        for i in range(0, len(batch), chunk_size):
            chunk = batch[i:i+chunk_size]
            p1 = model_dense.predict(chunk, verbose=0)
            p2 = model_res.predict(chunk, verbose=0)
            preds.append((p1 + p2) / 2.0)
            bar.progress(min((i + chunk_size) / 120, 1.0))
            
        final_probs = np.mean(np.vstack(preds), axis=0)
        status_text.empty(); bar.empty()
        
    else:
        # --- FAST SCAN (1 View) ---
        img = np.array(image.resize((224, 224))).astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        p1 = model_dense.predict(img, verbose=0)[0]
        p2 = model_res.predict(img, verbose=0)[0]
        final_probs = (p1 * 0.5) + (p2 * 0.5)

    return final_probs, time.time() - start_time

# ==========================================
# 5. UI LAYOUT
# ==========================================
# Sidebar Settings
with st.sidebar:
    st.title("⚙️ Settings")
    deep_mode = st.toggle("🧬 Deep Scan (120 Views)", value=False)
    explain_ai = st.toggle("🔥 Explain AI (Grad-CAM)", value=True)
    st.info("Deep Scan increases accuracy but takes longer.")

# Main Page
st.title("🫁 LungScan AI")
st.markdown("### Medical Diagnostic System")

uploaded_file = st.file_uploader("Upload X-Ray", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Patient X-Ray", use_column_width=True)
    
    with col2:
        st.write("#### Diagnostics")
        btn_label = "Run Deep Scan" if deep_mode else "Run Analysis"
        
        if st.button(btn_label):
            with st.spinner("⚡ analyzing patterns..."):
                probs, time_taken = run_prediction(image, deep_mode)
                
                # Results
                idx = np.argmax(probs)
                label = CLASSES[idx]
                conf = probs[idx] * 100
                color = "#27ae60" if label == "Normal" else "#e74c3c"
                
                st.markdown(f"""
                <div class="report-view" style="border-top: 5px solid {color};">
                    <h2 style="color: {color}; margin:0;">{label.replace('_', ' ')}</h2>
                    <p style="font-size: 24px; font-weight: bold;">{conf:.2f}% Confidence</p>
                    <p style="color:gray; font-size:12px;">⏱️ {time_taken:.2f}s | Scans: {120 if deep_mode else 1}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Grad-CAM
                if explain_ai and label != "Normal":
                    st.markdown("##### 🔥 AI Attention Map")
                    heatmap = generate_gradcam_overlay(image, model_res)
                    st.image(heatmap, caption="Red = Infected Region", use_column_width=True)
                
                # Bar Chart

                st.bar_chart(dict(zip(CLASSES, probs)))
