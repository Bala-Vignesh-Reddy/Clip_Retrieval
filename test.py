from transformers import CLIPProcessor, CLIPModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
import numpy as np
import pandas as pd

# Load the fine-tuned CLIP model and processor from the saved directories
fine_tuned_model_path = "fine_tuned_clip"
fine_tuned_processor_path = "fine_tuned_clip_processor"

model = CLIPModel.from_pretrained(fine_tuned_model_path)
processor = CLIPProcessor.from_pretrained(fine_tuned_processor_path)

valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
image_paths = [
    os.path.join('images', f) 
    for f in os.listdir('images') 
    if f.lower().endswith(valid_extensions)
]

IMAGE_DIR = "images"
metadata = pd.read_csv("metadata.csv")
metadata['image'] = metadata['image'].apply(lambda x: os.path.join(IMAGE_DIR, x))
# Function to embed images and save embeddings
# def generate_and_save_embeddings(image_paths, save_path='image_embeddings.npy'):
#     image = Image.open(image_paths).convert("RGB")
#     inputs = processor(images=image, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         outputs = model.get_image_features(**inputs)
#     return outputs.detach().numpy().flatten()

# image_embeddings = []
# for index, row in metadata.iterrows():
#     print(index, row['image'])
#     img_embedding = generate_and_save_embeddings(row['image'])
#     image_embeddings.append(img_embedding)

# image_embeddings = np.array(image_embeddings)
# np.save("fine_tuned_image_embeddings.npy", image_embeddings) # saving the embeddings 
# print("Embeddings saved")

# Function to perform semantic search using the fine-tuned model
# def semantic_search(query_text, top_k=3):
#     # Embed the query text using the fine-tuned model
#     inputs = processor(text=query_text, return_tensors="pt", padding=True).to(device)
    
#     with torch.no_grad():
#         text_features = model.get_text_features(**inputs)
    
#     # Normalize the text embeddings
#     text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

#     # Compute cosine similarities between the query and image embeddings
#     similarities = cosine_similarity(text_features.squeeze().cpu().numpy().reshape(1, -1), image_embeddings)

#     # Get the top-k most similar images
#     top_k_indices = similarities.argsort()[0][-top_k:][::-1]
    
#     # Retrieve the top-k images and their similarity scores
#     results = [(image_paths[i], similarities[0][i]) for i in top_k_indices]

#     return results

# # Example usage for a query
# query_text = "soft drink"
# top_results = semantic_search(query_text)

# # Print the results
# for idx, (image_path, score) in enumerate(top_results):
#     print(f"Rank {idx + 1}: {image_path} (Similarity: {score:.4f})")

image_embeddings = np.load("fine_tuned_image_embeddings.npy")
def semantic_search(query_text, top_k=3):
    # Embed the query text using the fine-tuned model
    inputs = processor(text=query_text, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    
    # Normalize the text embeddings
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    # Compute cosine similarities between the query and image embeddings
    similarities = cosine_similarity(text_features.cpu().numpy(), image_embeddings)

    # Get the top-k most similar images
    top_k_indices = similarities[0].argsort()[-top_k:][::-1]
    
    # Retrieve the top-k images and their similarity scores
    results = [(image_paths[i], similarities[0][i]) for i in top_k_indices]
    return results

# Example usage
query_text = "dabur toothpaste"
top_results = semantic_search(query_text, top_k=5)

# Print the results
for idx, (image_path, score) in enumerate(top_results):
    print(f"Rank {idx + 1}: {image_path} (Similarity: {score:.4f})")