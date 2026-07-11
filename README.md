#  Fashion-MNIST Image Classification using Transfer Learning

A deep learning project that classifies fashion products into one of 10 clothing categories using **Transfer Learning (VGG16)**. The application is deployed with **Streamlit**, allowing users to upload an image and instantly receive predictions along with confidence scores and interactive visualizations.

---

##  Features

- Transfer Learning using VGG16
- Image classification into 10 Fashion-MNIST classes
- Interactive Streamlit dashboard
- Upload custom images for prediction
- Top-3 / Top-5 prediction probabilities
- Confidence score visualization
- Probability bar charts
- GPU support (CUDA) when available

---

##  Technologies Used

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pandas
- Plotly
- Pillow

---

##  Project Structure

```
Fashion-MNIST-Classifier/
│
├── app.py
├── streamlit_app.py
├── model_utils.py
├── requirements.txt
├── models/
│   └── transfer_learning_model.pth
│
├── images/
│
└── README.md
```

---
##  Dataset

This project uses the **Fashion-MNIST** dataset, containing **70,000 grayscale images** across **10 clothing categories**.

### Classes

- T-shirt / Top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle Boot

---

##  Model Architecture

- Pre-trained VGG16
- Transfer Learning
- Frozen feature extractor
- Custom fully connected classifier
- Dropout for regularization
- Softmax output layer

---

##  Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Fashion-MNIST-Classifier.git
cd Fashion-MNIST-Classifier
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

or

```bash
streamlit run streamlit_app.py
```

---

## How It Works

1. Upload an image.
2. The image is preprocessed.
3. The trained VGG16 model performs inference.
4. Softmax probabilities are calculated.
5. The predicted class and confidence scores are displayed.

---

##  Output

The application displays:

- Uploaded Image
- Predicted Class
- Confidence Score
- Top Predictions
- Probability Distribution
- Interactive Charts

---

##  Requirements

- Python 3.9+
- PyTorch
- Torchvision
- Streamlit
- Plotly
- Pandas
- Pillow
- NumPy

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

##  Future Improvements

- Support multiple CNN architectures (ResNet, EfficientNet, DenseNet)
- Model comparison dashboard
- Batch image prediction
- Grad-CAM visualization
- Deploy on Streamlit Cloud
- REST API using FastAPI

---

## 👨‍💻 Author

Prabesh Kyapchhayakee

---

## ⭐ If you found this project useful, don't forget to star the repository!
