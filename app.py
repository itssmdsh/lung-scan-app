import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time
import os
import gdown
import matplotlib.cm as cm

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="LungScan AI",
    page_icon="🫁",
    layout="wide"
)

# =====================================================
# UI STYLE
# =====================================================
st.markdown("""
<style>
.stApp { background: linear-gradient(to bottom right, #f8f9fa, #e3f2fd); }
h1 { text-align: center; color: #1565C0; }
.stButton>button {
    width: 100%;
    height: 55px;
    font-size: 18px;
    background-color: #1976D2;
    color: white;
    border-radius: 8px;
}
.report {
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 10px 20px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LABELS
# =====================================================
CLASSES = [
    "Bacterial Pneumonia",
    "Corona Virus Disease",
    "Normal",
    "Tuberculosis"
]

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    dense_id = "1aWtU79Xk1Vmrg8BsBL9VgwwZxk6eY4oz"
    res_id   = "176xn7ZUy1iRllmtPxdcpeWplQ2nJ40sW"

    if not os.path.exists("Final_DenseNet.keras"):
        gdown.download(id=dense_id, output="Final_DenseNet.keras", quiet=False)

    if not os.path.exists("Final_ResNet.keras"):
        gdown.download(id=res_id, output="Final_ResNet.keras", quiet=False)

    dense = tf.keras.models.load_model("Final_DenseNet.keras")
    res   = tf.keras.models.load_model("Final_ResNet.keras")
    return dense, res

model_dense, model_res = load_models()

# =====================================================
# IMAGE AUGMENTATION (DEEP SCAN)
# =====================================================
def generate_120_views(img_pil):
    img = cv2.resize(np.array(img_pil), (224, 224))
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    views = []

    for angle in range(-14, 15, 2):
        for scale in [1.0, 1.05, 1.1, 1.15]:
            M = cv2.getRotationMatrix2D(center, angle, scale)
            aug = cv2.warpAffine(img, M, (w, h))
            views.append(aug)
            views.append(cv2.flip(aug, 1))

    return np.array(views)

# =====================================================
# 🔥 GRAD-CAM (CRASH-PROOF, KERAS 3 SAFE)
# =====================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # 🔒 Keras 3 safety
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        class_idx = tf.argmax(predictions, axis=-1)
        class_score = tf.gather(predictions[0], class_idx)

    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

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

# =====================================================
# PREDICTION
# =====================================================
def predict(image, deep_scan):
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

# =====================================================
# UI
# =====================================================
st.title("🫁 LungScan AI")

deep_scan = st.toggle("🧬 Deep Scan (120 Views)", False)
explain_ai = st.toggle("🔥 Explain AI (Grad-CAM)", True)

file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, width=350)

    if st.button("🔍 Analyze"):
        probs, t = predict(image, deep_scan)
        idx = int(np.argmax(probs))
        label = CLASSES[idx]

        st.markdown(f"""
        <div class="report">
            <h3>{label}</h3>
            <h1>{probs[idx]*100:.1f}%</h1>
            <p>⏱ {t:.2f}s</p>
        </div>
        """, unsafe_allow_html=True)

        if explain_ai and label != "Normal":
            st.subheader("🔥 AI Attention Map")
            st.image(generate_gradcam_overlay(image, model_res))

        st.subheader("Class Probabilities")
        st.bar_chart(dict(zip(CLASSES, probs)))
