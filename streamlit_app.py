# streamlit_app.py
import streamlit as st
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# Import utility functions
from model_utils import load_model, predict_image, FASHION_CLASSES

# --- Page Configuration ---
st.set_page_config(
    page_title="Fashion-MNIST Classifier - Transfer Learning",
    page_icon="👕",
    layout="wide"
)

# --- Custom CSS for better styling ---
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Prediction card styling */
    .prediction-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        transition: transform 0.2s ease;
    }
    
    .prediction-card:hover {
        transform: translateX(8px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }
    
    .prediction-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    
    .prediction-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d3748;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .prediction-percentage {
        font-size: 1.2rem;
        font-weight: 700;
        padding: 0.2rem 1rem;
        border-radius: 20px;
        background: #f0f0f0;
        min-width: 80px;
        text-align: center;
    }
    
    .percentage-high {
        color: #28a745;
        background: #d4edda;
    }
    
    .percentage-medium {
        color: #856404;
        background: #fff3cd;
    }
    
    .percentage-low {
        color: #721c24;
        background: #f8d7da;
    }
    
    .confidence-bar-container {
        background: #f0f0f0;
        border-radius: 8px;
        height: 8px;
        overflow: hidden;
        margin-top: 0.3rem;
    }
    
    .confidence-bar {
        height: 100%;
        border-radius: 8px;
        transition: width 1s ease-in-out;
    }
    
    .rank-badge {
        background: #667eea;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    
    /* Top prediction special styling */
    .top-prediction {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-left: 4px solid #ffd700;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .top-prediction .prediction-label {
        font-size: 1.5rem;
        justify-content: center;
    }
    
    .top-prediction .prediction-percentage {
        font-size: 2rem;
        padding: 0.3rem 2rem;
        background: #ffd700;
        color: #333;
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem 0 1rem 0;
        margin-top: 2rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# --- Header with styling ---
st.markdown("""
<div class="main-header">
    <h1>👕 Fashion-MNIST Classifier</h1>
    <p>Transfer Learning with VGG16 • 10 Fashion Categories</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Model path input
    model_path = st.text_input(
        "📂 Model Path",
        value="./saved_models/transfer_learning_model.pth",
        help="Path to your trained .pth model file"
    )
    
    # Model loading button
    if st.button("🚀 Load Model", use_container_width=True):
        with st.spinner("Loading model..."):
            try:
                if not os.path.exists(model_path):
                    st.error(f"❌ Model file not found at: {model_path}")
                else:
                    model = load_model(model_path)
                    st.session_state['model'] = model
                    st.success("✅ Model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
    
    # Model status
    st.markdown("---")
    st.markdown("### 📊 Model Status")
    if 'model' in st.session_state:
        st.success("✅ Model is ready for predictions")
        st.info("🟢 VGG16 with custom classifier")
    else:
        st.warning("⚠️ Please load a model first")
    
    st.markdown("---")
    st.markdown("### 📝 Class List")
    # Display class names in a nice grid
    class_cols = st.columns(2)
    for i, class_name in enumerate(FASHION_CLASSES):
        col_idx = i % 2
        with class_cols[col_idx]:
            st.write(f"{i+1}. {class_name}")
    
    # Clear results button
    if st.button("🗑️ Clear Results", use_container_width=True):
        if 'predictions' in st.session_state:
            del st.session_state['predictions']
        if 'image' in st.session_state:
            del st.session_state['image']
        st.rerun()

# --- Main Content ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Image")
    
    # File uploader with custom styling
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload any fashion item image"
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.session_state['image'] = image
            
            # Prediction button with styling
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("🔍 Predict", type="primary", use_container_width=True):
                    if 'model' in st.session_state:
                        with st.spinner("🧠 Classifying..."):
                            try:
                                predictions, indices, probs = predict_image(image, st.session_state['model'])
                                st.session_state['predictions'] = predictions
                                st.session_state['probs'] = probs
                                st.success("✅ Prediction complete!")
                            except Exception as e:
                                st.error(f"❌ Prediction failed: {e}")
                    else:
                        st.warning("⚠️ Please load a model first")
        except Exception as e:
            st.error(f"❌ Error loading image: {e}")

with col2:
    st.markdown("### 📊 Prediction Results")
    
    if 'predictions' in st.session_state:
        predictions = st.session_state['predictions']
        
        # Get top prediction
        top_class, top_conf = predictions[0]
        
        # Display top prediction with special styling
        st.markdown(f"""
        <div class="top-prediction">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">🏆 Top Prediction</div>
            <div class="prediction-label" style="font-size: 1.8rem; justify-content: center; gap: 1rem;">
                <span>{top_class}</span>
                <span class="prediction-percentage" style="font-size: 2rem; background: #ffd700; color: #333;">
                    {top_conf*100:.1f}%
                </span>
            </div>
            <div style="margin-top: 0.5rem; color: #666; font-size: 0.9rem;">
                Confidence Score
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display all predictions with combined labels
        st.markdown("#### All Predictions")
        
        for i, (class_name, confidence) in enumerate(predictions):
            # Determine color class for percentage
            if confidence >= 0.8:
                pct_class = "percentage-high"
                bar_color = "#28a745"
            elif confidence >= 0.5:
                pct_class = "percentage-medium"
                bar_color = "#ffc107"
            else:
                pct_class = "percentage-low"
                bar_color = "#dc3545"
            
            # Display prediction card with combined label and percentage
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-header">
                    <div class="prediction-label">
                        <span class="rank-badge">{i+1}</span>
                        <span>{class_name}</span>
                    </div>
                    <div class="prediction-percentage {pct_class}">
                        {confidence*100:.1f}%
                    </div>
                </div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar" style="width: {confidence*100:.1f}%; background: {bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Show detailed probabilities
        with st.expander("📈 Detailed View", expanded=False):
            fig, ax = plt.subplots(figsize=(10, 4))
            classes = [pred[0] for pred in predictions]
            probs = [pred[1] for pred in predictions]
            
            # Create horizontal bar chart with gradient colors
            colors = ['#28a745' if p >= 0.8 else '#ffc107' if p >= 0.5 else '#dc3545' for p in probs]
            bars = ax.barh(classes, probs, color=colors)
            ax.set_xlabel('Confidence')
            ax.set_xlim(0, 1)
            ax.set_title('Top-5 Prediction Probabilities')
            
            # Add percentage labels on bars
            for bar, prob in zip(bars, probs):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                       f'{prob*100:.1f}%', va='center', fontweight='bold')
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig)
            
            # Show raw probabilities table
            st.markdown("#### Raw Probabilities")
            prob_data = {
                'Rank': [f"#{i+1}" for i in range(len(predictions))],
                'Class': [pred[0] for pred in predictions],
                'Confidence': [f"{pred[1]*100:.2f}%" for pred in predictions],
                'Score': [f"{pred[1]:.4f}" for pred in predictions]
            }
            st.dataframe(prob_data, use_container_width=True, hide_index=True)

# --- Footer ---
st.markdown("""
<div class="footer">
    <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
        <span>🧠 Built with PyTorch</span>
        <span>📊 Powered by Streamlit</span>
        <span>🎯 VGG16 Transfer Learning</span>
    </div>
    <div style="margin-top: 0.5rem;">
        <small>Fashion-MNIST • 10 Classes • 28x28 Grayscale Images</small>
    </div>
</div>
""", unsafe_allow_html=True)