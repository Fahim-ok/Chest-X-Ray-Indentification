from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained model
# Assuming the model is saved as 'model.pth'
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
