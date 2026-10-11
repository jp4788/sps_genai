from fastapi import FastAPI, UploadFile, File, HTTPException
from app.embedding_model import calculate_embedding
from app.cnn_model import CNN

import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
from pathlib import Path

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Word Embedding API is running"}


@app.get("/embedding/{word}")
def get_embedding(word: str):
    embedding = calculate_embedding(word)

    return {
        "word": word,
        "embedding": embedding.tolist()
    }


# Load the trained CNN model
model = CNN()

model_path = Path(__file__).with_name("cnn_model.pth")
model.load_state_dict(
    torch.load(model_path, map_location="cpu", weights_only=True)
)
model.eval()

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()

        return {"class": classes[predicted_class]}

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )
