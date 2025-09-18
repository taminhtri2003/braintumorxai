import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import io

# --------------------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------------------
# Set the layout to wide mode for a better view and provide a title for the browser tab.
st.set_page_config(layout="wide", page_title="AI Explainability: Grad-CAM")


# --------------------------------------------------------------------------
# Model Loading
# --------------------------------------------------------------------------
# It's good practice to inform the user about the model loading status.
# We'll simulate a dummy model if the actual file is not found.
@st.cache_resource
def load_model():
    """Load the trained model or create a dummy model for demonstration."""
    try:
        # --- IMPORTANT ---
        # Replace "my_model.keras" with the actual path to your trained model file.
        model = keras.models.load_model("my_model.keras")
        return model
    except (IOError, ImportError):
        st.warning("Model file 'my_model.keras' not found. A dummy model will be used for demonstration purposes. Please upload your model.", icon="⚠️")
        # Create a simple dummy model for UI demonstration if the real one isn't available
        dummy_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(8, 3, activation='relu', name='conv1'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(16, 3, activation='relu', name='last_conv_layer'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        return dummy_model

model = load_model()

# --- Placeholder for class names ---
# For a real application, you should replace this with your actual class labels.
CLASS_NAMES = [f"Class {i}" for i in range(model.output_shape[1])]


# --------------------------------------------------------------------------
# Core Grad-CAM Functions (Unchanged from original logic)
# --------------------------------------------------------------------------
def get_img_array(uploaded_file, size):
    """Convert uploaded image to a NumPy array."""
    img = Image.open(uploaded_file).convert("RGB").resize(size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array, img

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate the Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(
        [model.input], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), int(pred_index)

def overlay_heatmap(img, heatmap, alpha=0.4, cmap="jet"):
    """Overlay the heatmap on the original image."""
    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap(cmap)
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.size[0], img.size[1]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    img_array = keras.utils.img_to_array(img)
    superimposed_img = jet_heatmap * alpha + img_array
    return keras.utils.array_to_img(superimposed_img)


# --------------------------------------------------------------------------
# Streamlit App UI
# --------------------------------------------------------------------------
st.title("🧠 Brain Tumor Image Classification with Grad-CAM Explainability")
st.markdown("""
Welcome! This application demonstrates how **Grad-CAM** (Gradient-weighted Class Activation Mapping)
can provide visual explanations for decisions made by Convolutional Neural Networks (CNNs) .

Used for **Biomedical Image Processing** Lab 

**How it works:**
1.  **Upload an image** using the controls on the left.
2.  The model will **classify the image** and show its top prediction.
3.  A **heatmap** will be generated and overlaid on the image, highlighting the regions the model focused on for its prediction.
""")

# --- Sidebar for User Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Adding interactive controls for the heatmap visualization
    st.header("🎨 Heatmap Options")
    alpha = st.slider("Intensity (Alpha)", 0.2, 1.0, 0.5, 0.05, help="Controls the transparency of the heatmap overlay.")
    cmap = st.selectbox("Color Map", ["jet", "viridis", "plasma", "inferno", "magma"], help="The color scheme for the heatmap.")


# --- Main Content Area ---
if uploaded_file is None:
    # Display a welcome message and a sample image if no file is uploaded.
    st.info("Upload an image using the sidebar to get started.", icon="👈")
    try:
        st.image("https://placehold.co/600x400/2F343D/FFFFFF?text=Upload+an+Image", caption="Awaiting image upload")
    except Exception as e:
        st.error(f"Could not load placeholder image. Error: {e}")

else:
    # --- Image Processing and Prediction ---
    img_size = model.input_shape[1:3]
    img_array, orig_img = get_img_array(uploaded_file, size=img_size)

    # Find the last convolutional layer automatically
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name is None:
        st.error("❌ No Conv2D layer found in the model. Cannot apply Grad-CAM.", icon="🔥")
    else:
        # --- Run Grad-CAM and get predictions ---
        with st.spinner('Analyzing the image...'):
            heatmap, pred_class_index = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
            superimposed_img = overlay_heatmap(orig_img, heatmap, alpha=alpha, cmap=cmap)
            preds = model.predict(img_array)[0]
            confidence = np.max(preds)
            predicted_class_name = CLASS_NAMES[pred_class_index]

        st.header("🔍 Analysis Results")
        st.divider()

        # --- Display Images ---
        col1, col2 = st.columns(2)
        with col1:
            st.image(orig_img, caption="Original Uploaded Image", use_container_width=True)
        with col2:
            st.image(superimposed_img, caption=f"Grad-CAM Visualization (Intensity: {alpha})", use_container_width=True)

        st.divider()

        # --- Display Prediction Metrics & Charts ---
        st.subheader("📊 Prediction Details")
        col1, col2 = st.columns([1, 2])

        with col1:
            # Using st.metric to highlight the top prediction in a clear, card-like format.
            confidence_percent = confidence * 100
            st.metric(
                label="🏆 Top Prediction",
                value=predicted_class_name,
                delta=f"{confidence_percent:.2f}% Confidence",
                delta_color="normal"
            )

        with col2:
            # Show the top 5 predictions in a more detailed bar chart.
            top_5_indices = np.argsort(preds)[::-1][:5]
            top_5_confidences = preds[top_5_indices]
            top_5_class_names = [CLASS_NAMES[i] for i in top_5_indices]

            df = pd.DataFrame({
                "Confidence": top_5_confidences,
                "Class": top_5_class_names,
            })
            
            st.write("**Top 5 Predictions:**")
            st.bar_chart(df.set_index("Class"))

        # --- Explanation Expander ---
        with st.expander("💡 How to interpret this?"):
            st.markdown(f"""
                The model is **{confidence_percent:.2f}% confident** that this image belongs to the class **'{predicted_class_name}'**.

                The **Grad-CAM visualization** on the right shows the areas the model found most important for this classification.
                - **Warmer colors (like red and yellow)** indicate regions that strongly influenced the prediction.
                - **Cooler colors (like blue)** indicate less important regions.
                This helps us understand if the model is "looking" at the right parts of the image.
            """)

