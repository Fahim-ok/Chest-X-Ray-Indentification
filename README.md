# Chest X-ray Classification: COVID-19, Viral Pneumonia, and Normal

This project classifies chest X-rays into three categories: Normal, COVID-19, and Viral Pneumonia using a deep learning model. The deployment is done through two applications: a Flask web application and a Gradio interface.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Deployment](#deployment)
  - [Flask Application](#flask-application)
  - [Gradio Application](#gradio-application)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project aims to classify chest X-ray images into three categories: Normal, COVID-19, and Viral Pneumonia. The model is trained using PyTorch and deployed via Flask and Gradio for easy accessibility.

## Dataset
The dataset used in this project is the [COVID-QU-Ex Dataset](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu) from Kaggle, which contains X-ray images categorized as Normal, COVID-19, and Viral Pneumonia.

## Model Training
The model training is done using an ensemble of ResNet18, VGG16, and InceptionV3 architectures. The training process is documented in the Jupyter Notebook file `Detecting_COVID_19_and_Viral_Pneumonia_Notebook_Ensamble_ResNet18,VGG16,InceptionV3.ipynb`.

## Deployment
The trained model is deployed using two different approaches: Flask and Gradio.

### Flask Application
The Flask application provides a web interface for users to upload chest X-ray images and get predictions.

#### `flask_app.py`
```python
from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained model
model = torch.load('model.pth')
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            img_bytes = file.read()
            class_id = predict(img_bytes)
            return f"Predicted Class: {class_id}"
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

#### `gradio_app.py`
Gradio Application
The Gradio application provides an interactive interface for users to upload chest X-ray images and get predictions.

gradio_app.py
python

import gradio as gr
import torch
from torchvision import transforms
from PIL import Image

# Load the pre-trained model
model = torch.load('model.pth')
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image):
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Define the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(shape=(224, 224)),
    outputs="label",
    description="Upload a chest X-ray image to predict the class (Normal, COVID-19, Viral Pneumonia)"
)

if __name__ == '__main__':
    iface.launch()


Installation
Follow these steps to set up the project locally:

Clone the repository:
git clone https://github.com/yourusername/chest-xray-classification.git
cd chest-xray-classification
Set up a virtual environment:


python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages:


pip install -r requirements.txt
Download the pre-trained model and place it in the project directory:


# Assuming the model file is named 'model.pth'
Usage
Running the Flask Application
Navigate to the project directory and run the Flask application:


python flask_app.py
Open a web browser and go to http://127.0.0.1:5000/ to access the application.

Running the Gradio Application
Navigate to the project directory and run the Gradio application:


python gradio_app.py
Follow the link provided in the terminal to access the Gradio interface.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure that your code adheres to the existing style and includes appropriate tests.
