import streamlit as st, torch, torch.nn as nn, torchvision.models as models
from torchvision.transforms import transforms
from PIL import Image
import numpy as np, pandas as pd, plotly.express as px, os

C = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
E = ['👕','👖','🧥','👗','🧣','👡','👔','👟','👜','👢']
Colors = ['#FF6B6B','#4ECDC4','#45B7D1','#FFD93D','#FF8A5C','#DDA0DD','#A8E6CF','#FFB3B3','#B5B8C3','#6C5CE7']

st.set_page_config(page_title="Fashion-MNIST", page_icon="👗", layout="wide")
d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom styling
st.markdown("""
<style>.main-header{background:linear-gradient(135deg,#667eea,#764ba2);padding:1.5rem 2rem;border-radius:15px;margin-bottom:2rem}
.main-header h1{color:white!important;font-size:2.5rem!important;margin:0!important}
.pred-card{padding:1.5rem;border-radius:15px;background:white;box-shadow:0 4px 20px rgba(0,0,0,0.08);margin:0.5rem 0}
.progress-bar{height:25px;border-radius:30px;background:#f0f0f0;overflow:hidden;margin:0.4rem 0}
.progress-fill{height:100%;border-radius:30px;display:flex;align-items:center;padding:0 12px;color:white;font-weight:600;font-size:0.85rem}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load():
    v = models.vgg16(weights=None)
    v.classifier = nn.Sequential(
        nn.Linear(25088, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 10)
    )
    if not os.path.exists('models/transfer_learning_model.pth'):
        return None
    v.load_state_dict(torch.load('models/transfer_learning_model.pth', map_location=d))
    return v.to(d).eval()
t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

st.markdown('<div class="main-header"><h1>👗 Fashion-MNIST Classifier</h1></div>', unsafe_allow_html=True)
model = load()
if not model: st.error("Model not found!"); st.stop()

u = st.file_uploader("", type=['png','jpg','jpeg','bmp'], label_visibility="collapsed")
if u:
    img = Image.open(u)
    tensor = t(img.convert('RGB')).unsqueeze(0).to(d)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())
        conf = float(probs[pred])
        top3 = [(C[i], float(probs[i]), Colors[i]) for i in np.argsort(probs)[-3:][::-1]]
    
    col1,col2 = st.columns([1,1])
    with col1:
        st.image(img, use_column_width=True)
        st.caption(f"Image: {u.name}")
    with col2:
        color = Colors[pred]
        st.markdown(f"""
        <div class='pred-card' style='border-left:6px solid {color};'>
            <div style='display:flex;align-items:center;gap:1rem;'>
                <span style='font-size:3rem;'>{E[pred]}</span>
                <div>
                    <h2 style='margin:0;color:{color};font-size:1.8rem;'>{C[pred]}</h2>
                    <p style='margin:0;font-size:1.1rem;'>Confidence: <strong style='color:{color};'>{conf:.1%}</strong></p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Top 3")
        for rank, (name, conf, color) in enumerate(top3, 1):
            st.markdown(f"""
            <div>
                <div style='display:flex;justify-content:space-between;'>
                    <span>{['🥇','🥈','🥉'][rank-1]} <b style='color:{color};'>{E[C.index(name)]} {name}</b></span>
                    <span style='color:{color};font-weight:600;'>{conf:.1%}</span>
                </div>
                <div class='progress-bar'>
                    <div class='progress-fill' style='width:{conf*100:.1f}%;background:{color};'>{conf:.1%}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Bar chart only - pie chart removed
    st.markdown("---")
    df = pd.DataFrame({'Class': C, 'Confidence': [float(p) for p in probs]}).sort_values('Confidence')
    fig = px.bar(df, x='Confidence', y='Class', orientation='h', color='Confidence',
                 color_continuous_scale='Viridis', height=350,
                 text=df['Confidence'].apply(lambda x: f'{x:.1%}'))
    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis={'range':[0,1]}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)