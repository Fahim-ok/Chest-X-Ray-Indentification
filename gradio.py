import gradio as gr
import torch
from torchvision import transforms
from PIL import Image

# Load the pre-trained model
# Assuming the model is saved as 'model.pth'
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
