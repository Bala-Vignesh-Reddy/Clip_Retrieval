from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt

def generate_embeddings(query, model, processor, is_image_query=False):
    """
    Generating an embedding for the query (image or text).
    - is_image_query: is True, query is treated as image.. else text.
    """

    if is_image_query:
        # Process image query
        inputs = processor(images=query, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
    else:
        # Process text query
        inputs = processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            features = model.get_text_features(**inputs)
    
    # Normalize features
    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().numpy()

def qdrant_query(qdrant_client, collection_name, query_embedding, top_k=10):
    """
    Querying qdrant for most similar vectors for given embedding.

    para:
        - qdrant_client: qdrant client object
        - collection_name: name of the collection to query
        - query_embedding: embedding of the query
        - top_k: number of most similar vectors to return
    """

    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=top_k,
    )

    return search_result

def display_query_results(search_results):
    """
    Displaying the search results... showing images and metadata..
    """
    # for result in search_results:
    #     print(f"Image ID: {result.id}")
    #     print(f"Score: {result.score}")

    #displaying image
    num_results = len(search_results)
    fig, axes = plt.subplots(1, num_results, figsize=(15, 5))
    if num_results == 1:
        axes = [axes]
    for idx, result in enumerate(search_results):
        id = result.id
        score = result.score
        print(f"Image ID: {id}")
        print(f"Score: {score}")
        metadata = load_metadata_for_image(result.id)
        print("Metadata:", metadata)
    
        img_path = f"../images/{metadata['image_path']}"
        if os.path.exists(img_path):
            image = Image.open(img_path)
            axes[idx].imshow(image)
            axes[idx].axis("off")
            axes[idx].set_title(f"Image {idx+1}\nScore: {score:.2f}")
            # axes[idx].set_xlabel(f"Name: {metadata['name']}\nDescription: {metadata['description']}\nPrice: {metadata['price']}\nQuantity: {metadata['quantity']}") 
    plt.show()


def load_metadata_for_image(image_id):
    """
    loading metadata for a given image based on id.
    """
    IMAGE_DIR = r"..\images"
    metadata = pd.read_csv(r"../metadata.csv")
    # metadata['image'] = metadata['image'].apply(lambda x: os.path.join(IMAGE_DIR, x))

    image_metadata = metadata[metadata['id'] == image_id].iloc[0]

    data = {
        "name": image_metadata['name'],
        "description": image_metadata['description'],
        "price": image_metadata['price'],
        "quantity": image_metadata['Qty'],
        "image_path": image_metadata['image']
    }

    return data

if __name__ == "__main__":
    qdrant_client = QdrantClient(url="http://localhost:6333")
    collection_name = "image_search"

    query = input("Enter query:") # give input here.. 
    is_image_query = False

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    query_embedding = generate_embeddings(query, model, processor, is_image_query)

    search_results = qdrant_query(qdrant_client, collection_name, query_embedding)

    display_query_results(search_results)
