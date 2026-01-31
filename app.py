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
# 1. PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="LungScan AI | Medical Diagnostic System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PROFESSIONAL MEDICAL THEME CSS
st.markdown("""
    <style>
    .stApp { background: linear-gradient(to bottom right, #f8f9fa, #e3f2fd); }
    h1 { color: #1565C0; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; text-align: center; }
    h3 { color: #455A64; text-align: center; font-weight: 300; }
    .stButton>button {
        width: 100%; background-color: #1976D2; color: white; font-weight: bold;
        border-radius: 8px; height: 55px; border: none; font-size: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: 0.3s;
    }
    .stButton>button:hover { background-color: #1565C0; transform: translateY(-2px); }
    .report-view {
        background-color: white; padding: 30px; border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08); text-align: center;
        margin-bottom: 20px; border: 1px solid #e0e0e0;
    }
    .info-box {
        background-color: #E3F2FD; padding: 15px; border-radius: 10px;
        border-left: 5px solid #1976D2; margin-bottom: 15px;
    }
    .symptom-box {
        background-color: #fff3e0; border-left: 5px solid #ff9800;
        padding: 15px; border-radius: 5px; margin-top: 10px; text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. KNOWLEDGE BASE (Symptoms)
# ==========================================
CLASSES = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis']

SYMPTOMS = {
    'Bacterial Pneumonia': [
        "High fever and chills",
        "Cough with thick yellow or green phlegm",
        "Sharp chest pain when breathing deeply",
        "Shortness of breath during mild activity"
    ],
    'Corona Virus Disease': [
        "Fever or chills",
        "New loss of taste or smell",
        "Dry, persistent cough",
        "Difficulty breathing (dyspnea)"
    ],
    'Tuberculosis': [
        "Persistent cough lasting 3+ weeks",
        "Coughing up blood (hemoptysis)",
        "Unintentional weight loss",
        "Night sweats and fever"
    ],
    'Normal': [
        "No radiological abnormalities detected.",
        "Lungs appear clear and healthy.",
        "Maintain regular check-ups."
    ]
}

# ==========================================
# 3. MODEL LOADING (Google Drive Bypass)
# ==========================================
@st.cache_resource
def load_models():
    # ⚠️ GOOGLE DRIVE IDs
    id_dense = '1aWtU79Xk1Vmrg8BsBL9VgwwZxk6eY4oz' 
    id_res   = '176xn7ZUy1iRllmtPxdcpeWplQ2nJ40sW'   
    
    if not os.path.exists("Final_DenseNet.keras"):
        with st.spinner("📥 System Initializing: Downloading DenseNet Core..."):
            gdown.download(id=id_dense, output="Final_DenseNet.keras", quiet=False)

    if not os.path.exists("Final_ResNet.keras"):
        with st.spinner("📥 System Initializing: Downloading ResNet Core..."):
            gdown.download(id=id_res, output="Final_ResNet.keras", quiet=False)

    m1 = tf.keras.models.load_model("Final_DenseNet.keras")
    m2 = tf.keras.models.load_model("Final_ResNet.keras")
    return m1, m2

with st.spinner("🏥 Booting Diagnostic Engine..."):
    try:
        model_dense, model_res = load_models()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

# ==========================================
# 4. CORE AI ENGINES
# ==========================================
def generate_120_views(image_pil):
    img = np.array(image_pil.convert('RGB'))
    img = cv2.resize(img, (224, 224))
    views = []
    h, w = img.shape[:2]
    center = (w//2, h//2)
    for angle in range(-14, 15, 2): 
        for scale in [1.0, 1.05, 1.10, 1.15]: 
            M = cv2.getRotationMatrix2D(center, angle, scale)
            aug = cv2.warpAffine(img, M, (w, h))
            views.append(aug)
            views.append(cv2.flip(aug, 1)) 
    return np.array(views)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        # ---------------------------------------------------------
        # 🛠️ NUCLEAR FIX: Handle Scalar vs Array conversion
        # ---------------------------------------------------------
        if isinstance(pred_index, tf.Tensor):
            pred_index = pred_index.numpy()
        
        if isinstance(pred_index, np.ndarray):
            # If it's an array like [2], flatten to get 2
            pred_index = pred_index.flatten()[0]
            
        # Final safety cast to plain Python integer
        pred_index = int(pred_index)
            
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
    
    original_img = np.array(img_pil)
    superimposed_img = np.array(jet_heatmap) * 0.4 + original_img * 0.6
    return np.uint8(superimposed_img)

def run_prediction(image, deep_scan_mode):
    start_time = time.time()
    
    if deep_scan_mode:
        status_text = st.empty()
        bar = st.progress(0)
        status_text.info("🧬 Performing Deep Scan (120 Angles)...")
        
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
        img = np.array(image.resize((224, 224))).astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        p1 = model_dense.predict(img, verbose=0)[0]
        p2 = model_res.predict(img, verbose=0)[0]
        final_probs = (p1 * 0.5) + (p2 * 0.5)

    return final_probs, time.time() - start_time

# ==========================================
# 5. UI LAYOUT
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
    st.title("Settings")
    st.markdown("""
    <div class="info-box">
        <b>Supported Detections:</b><br>
        • Bacterial Pneumonia<br>• COVID-19<br>• Tuberculosis (TB)<br>• Normal (Healthy)
    </div>
    """, unsafe_allow_html=True)
    
    deep_mode = st.toggle("🧬 Deep Scan (High Accuracy)", value=False)
    explain_ai = st.toggle("🔥 Explain AI (Heatmap)", value=True)
    st.markdown("---")
    st.caption("v1.0.7 | ResNet50V2 + DenseNet121")

st.title("🫁 LungScan AI")
st.markdown("### Advanced Chest X-Ray Diagnostic System")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown("#### 1. Upload Patient Scan")
    uploaded_file = st.file_uploader("Upload X-Ray (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded X-Ray", use_container_width=True)

with col2:
    if uploaded_file:
        st.markdown("#### 2. Clinical Analysis")
        btn_label = "🔍 Run Deep Scan Analysis" if deep_mode else "⚡ Run Fast Analysis"
        
        if st.button(btn_label):
            with st.spinner("🤖 Analyzing pulmonary patterns..."):
                probs, time_taken = run_prediction(image, deep_mode)
                
                idx = np.argmax(probs)
                label = CLASSES[idx]
                conf = probs[idx] * 100
                
                if label == "Normal":
                    color = "#2e7d32"
                    status = "✅ Healthy Lung Tissue Detected"
                else:
                    color = "#c62828"
                    status = f"⚠️ Abnormality Detected: {label.replace('_', ' ')}"
                
                st.markdown(f"""
                <div class="report-view" style="border-top: 6px solid {color};">
                    <h3 style="color: {color}; margin:0; font-weight:bold;">{status}</h3>
                    <h1 style="font-size: 45px; margin: 10px 0; color: #333;">{conf:.1f}%</h1>
                    <p style="color:gray; font-size:14px;">Confidence Score</p>
                    <hr>
                    <p style="font-size:12px;">⏱️ Time: {time_taken:.3f}s | 🧠 Scans: {120 if deep_mode else 1}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"##### 🩺 Typical Symptoms ({label.replace('_', ' ')})")
                symptoms_list = SYMPTOMS.get(label, [])
                symptoms_html = "".join([f"<li>{s}</li>" for s in symptoms_list])
                st.markdown(f"<div class='symptom-box'><ul>{symptoms_html}</ul></div>", unsafe_allow_html=True)

                if explain_ai and label != "Normal":
                    st.markdown("##### 🔥 AI Attention Map (Lesion Localization)")
                    heatmap = generate_gradcam_overlay(image, model_res)
                    st.image(heatmap, caption="Red Areas Indicate Disease Pattern", use_container_width=True)
                
                st.markdown("##### Detailed Probability Distribution")
                st.bar_chart(dict(zip(CLASSES, probs)), color=color)

    else:
        st.info("👈 Please upload a Chest X-Ray from the left panel to begin analysis.")
        st.markdown("""
        **System Capabilities:**
        * **97%+ Accuracy** using Ensemble Learning
        * **120-View Deep Scan** for edge cases
        * **Grad-CAM** visualization for interpretability
        * **Automated Symptom Checker**
        """)
