# model_utils.py
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import numpy as np

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fashion-MNIST class names
FASHION_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def load_model(model_path):
    """
    Load the VGG16 model with the same architecture used in training
    """
    # Create the same VGG16 architecture
    model = models.vgg16(pretrained=False)
    
    # Freeze feature extraction layers (as in training)
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Replace classifier with the same architecture
    model.classifier = nn.Sequential(
        nn.Linear(25088, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 10)  # 10 classes for Fashion-MNIST
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    return model

# Transformation pipeline matching the training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    """
    Preprocess the uploaded image to match the training data format
    Fashion-MNIST images are 28x28 grayscale, but the model expects 224x224 RGB
    """
    # Convert to grayscale if image is not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 28x28 (Fashion-MNIST size)
    image = image.resize((28, 28))
    
    # Convert to numpy array
    image_array = np.array(image, dtype=np.uint8)
    
    # Stack to create 3 channels (RGB) - same as training
    image_array = np.stack([image_array] * 3, axis=-1)
    
    # Convert back to PIL Image
    image = Image.fromarray(image_array)
    
    # Apply the same transformations as training
    image_tensor = transform(image)
    
    # Add batch dimension
    return image_tensor.unsqueeze(0).to(device)

def predict_image(image, model):
    """
    Make predictions on the preprocessed image
    """
    # Preprocess the image
    input_tensor = preprocess_image(image)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
    
    # Get top-5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    top5_prob = top5_prob.squeeze().cpu().numpy()
    top5_indices = top5_indices.squeeze().cpu().numpy()
    
    predictions = []
    for i in range(len(top5_indices)):
        pred_class = FASHION_CLASSES[top5_indices[i]]
        pred_confidence = top5_prob[i]
        predictions.append((pred_class, pred_confidence))
    
    return predictions, top5_indices, top5_prob