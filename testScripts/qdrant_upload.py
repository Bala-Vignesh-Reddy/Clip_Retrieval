import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct

# loading embeddings and metadata
embeddings = np.load("image_embeddings.npy")
metadata = pd.read_csv("../metadata.csv")

# embedding dimension is 512
embedding_dim = embeddings.shape[1]
# print(f"Embedding dimension: {embedding_dim}")

#connecting to qdrant
qdrant_client = QdrantClient(host="localhost", port=6333)

#creating qdrant collection
#recreate collection is used instead of create collection because it checks whether the collection exists or not...
collection_name = "image_search"

# qdrant_client.delete_collection(collection_name)  #used to delete the collection.. to overcome we will use collection.exists()

if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name, 
        vectors_config=VectorParams(size=embedding_dim, distance="Cosine")
    )
    print(f"collection {collection_name} created successfully..")
else:
    print(f"collection {collection_name} already exists..")
    refresh = input("Do you want to refresh the collection? (y/n)").strip().lower()
    if refresh == "y":
        qdrant_client.delete_collection(collection_name)
        qdrant_client.create_collection(
            collection_name=collection_name, 
            vectors_config=VectorParams(size=embedding_dim, distance="Cosine")
        )
        print(f"collection {collection_name} refreshed successfully..")
    else:
        print("Collection refresh cancelled.")

#uploading the data
points = []
for index, row in metadata.iterrows():
    vector = embeddings[index].tolist()
    payload = {
        'id': int(row['id']),
        'name': row['name'],
        'description': row['description'],
        'price': float(row['price']),
        'quantity': int(row['Qty']),
        'image_name': row['image']
    }
    points.append(PointStruct(id=index, vector=vector, payload=payload))

qdrant_client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"Data uploaded - {len(points)} points to the qdrant collection '{collection_name}'")