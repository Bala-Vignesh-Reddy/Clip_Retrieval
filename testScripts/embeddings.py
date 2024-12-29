from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import pandas as pd
import numpy as np
import os

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("model loaded")

IMAGE_DIR = r"..\images"
metadata = pd.read_csv(r"../metadata.csv")
metadata['image'] = metadata['image'].apply(lambda x: os.path.join(IMAGE_DIR, x))

def image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs.detach().numpy().flatten()

image_embeddings = []
for index, row in metadata.iterrows():
    print(index, row['image'])
    img_embedding = image_embedding(row['image'])
    image_embeddings.append(img_embedding)

image_embeddings = np.array(image_embeddings)
np.save("image_embeddings.npy", image_embeddings) # saving the embeddings 
print("Embeddings saved")
