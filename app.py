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
    page_title="LungScan AI | Medical Diagnostic System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. THEME
# ==========================================
st.markdown("""
<style>
.stApp { background: linear-gradient(to bottom right, #f8f9fa, #e3f2fd); }
h1 { color: #1565C0; text-align: center; }
.stButton>button {
    width: 100%;
    background-color: #1976D2;
    color: white;
    font-weight: bold;
    height: 55px;
    border-radius: 8px;
    font-size: 18px;
}
.report-view {
    background: white;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
}
.symptom-box {
    background: #fff3e0;
    padding: 15px;
    border-left: 5px solid #ff9800;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LABELS & SYMPTOMS
# ==========================================
CLASSES = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis']

SYMPTOMS = {
    'Bacterial Pneumonia': [
        "High fever", "Productive cough", "Chest pain", "Shortness of breath"
    ],
    'Corona Virus Disease': [
        "Fever", "Dry cough", "Loss of smell", "Breathing difficulty"
    ],
    'Tuberculosis': [
        "Chronic cough", "Weight loss", "Night sweats", "Blood in sputum"
    ],
    'Normal': [
        "No abnormalities detected", "Healthy lung fields"
    ]
}

# ==========================================
# 4. MODEL LOADING
# ==========================================
@st.cache_resource
def load_models():
    dense_id = "1aWtU79Xk1Vmrg8BsBL9VgwwZxk6eY4oz"
    res_id   = "176xn7ZUy1iRllmtPxdcpeWplQ2nJ40sW"

    if not os.path.exists("Final_DenseNet.keras"):
        gdown.download(id=dense_id, output="Final_DenseNet.keras", quiet=False)

    if not os.path.exists("Final_ResNet.keras"):
        gdown.download(id=res_id, output="Final_ResNet.keras", quiet=False)

    return (
        tf.keras.models.load_model("Final_DenseNet.keras"),
        tf.keras.models.load_model("Final_ResNet.keras")
    )

model_dense, model_res = load_models()

# ==========================================
# 5. IMAGE AUGMENTATION
# ==========================================
def generate_120_views(image_pil):
    img = cv2.resize(np.array(image_pil), (224, 224))
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    views = []

    for angle in range(-14, 15, 2):
        for scale in [1.0, 1.05, 1.10, 1.15]:
            M = cv2.getRotationMatrix2D(center, angle, scale)
            aug = cv2.warpAffine(img, M, (w, h))
            views.append(aug)
            views.append(cv2.flip(aug, 1))

    return np.array(views)

# ==========================================
# 6. 🔥 FIXED GRAD-CAM (CRASH-PROOF)
# ==========================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)

        # FIX 1: Handle list outputs (Keras 3)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        # FIX 2: Safe index extraction
        if pred_index is None:
            pred_index = tf.argmax(preds, axis=-1)

        if isinstance(pred_index, tf.Tensor):
            pred_index = pred_index.numpy()

        pred_index = int(np.squeeze(pred_index))

        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

def generate_gradcam_overlay(img_pil, model):
    img_array = np.expand_dims(
        np.array(img_pil.resize((224, 224))).astype("float32") / 255.0,
        axis=0
    )

    last_conv = next(
        layer.name for layer in reversed(model.layers)
        if isinstance(layer, tf.keras.layers.Conv2D)
    )

    heatmap = make_gradcam_heatmap(img_array, model, last_conv)
    heatmap = cv2.resize(heatmap, img_pil.size)

    heatmap = cm.jet(heatmap)[..., :3] * 255
    overlay = 0.6 * np.array(img_pil) + 0.4 * heatmap

    return overlay.astype(np.uint8)

# ==========================================
# 7. PREDICTION ENGINE
# ==========================================
def run_prediction(image, deep_scan):
    start = time.time()

    if deep_scan:
        batch = generate_120_views(image).astype("float32") / 255.0
        p1 = model_dense.predict(batch, verbose=0)
        p2 = model_res.predict(batch, verbose=0)
        probs = np.mean((p1 + p2) / 2.0, axis=0)
    else:
        img = np.expand_dims(
            np.array(image.resize((224, 224))).astype("float32") / 255.0,
            axis=0
        )
        probs = (model_dense.predict(img, verbose=0)[0] +
                 model_res.predict(img, verbose=0)[0]) / 2.0

    return probs, time.time() - start

# ==========================================
# 8. UI
# ==========================================
st.title("🫁 LungScan AI")
st.markdown("### Advanced Chest X-Ray Diagnostic System")

deep_mode = st.toggle("🧬 Deep Scan Mode", False)
explain_ai = st.toggle("🔥 Explain AI (Grad-CAM)", True)

uploaded = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, width=350)

    if st.button("🔍 Analyze"):
        probs, t = run_prediction(image, deep_mode)
        idx = np.argmax(probs)
        label = CLASSES[idx]

        st.markdown(f"""
        <div class="report-view">
            <h3>{label}</h3>
            <h1>{probs[idx]*100:.1f}%</h1>
            <p>⏱ {t:.2f}s</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Symptoms")
        st.markdown(
            "<div class='symptom-box'><ul>" +
            "".join(f"<li>{s}</li>" for s in SYMPTOMS[label]) +
            "</ul></div>",
            unsafe_allow_html=True
        )

        if explain_ai and label != "Normal":
            st.markdown("#### 🔥 AI Attention Map")
            st.image(generate_gradcam_overlay(image, model_res))

        st.markdown("#### Probability Distribution")
        st.bar_chart(dict(zip(CLASSES, probs)))
