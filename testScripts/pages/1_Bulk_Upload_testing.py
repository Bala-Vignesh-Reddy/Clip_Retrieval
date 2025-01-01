import streamlit as st
from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
import numpy as np

@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def generate_embeddings(image, model, processor):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_image_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().numpy()

def main():
    st.title("Bulk Image Upload and Processing")

    model, processor = load_model()
    qdrant_client = QdrantClient(host="localhost", port=6333)
    collection_name = "test_collection"

    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=512, distance="Cosine"),
            metadata={
                "image_directory": "./testImages" # deafult path for image directory
            }
        )
        st.info(f"Created new Collection: {collection_name}")
    else:
        if st.checkbox("Recreate collection? (This will delete existing data)"):
            qdrant_client.delete_collection(collection_name)
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=512, distance="Cosine")
            )
            st.info(f"Recreated collection: {collection_name}")


    uploaded_files = st.file_uploader(
        "Choose images to upload",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"Number of images uploaded: {len(uploaded_files)}")

        col1, col2 = st.columns(2)
        with col1:
            save_path = st.text_input("Save folder path:", "./testImages")
        with col2:
            prefix = st.text_input("Filename prefix:", "uploaded_") 

        if st.button("Process and save images"):
            with st.spinner("Processing images..."):
                try:
                    qdrant_client.update_collection(
                        collection_name=collection_name,
                        metadata={
                            "image_directory": save_path
                        }   
                    )
                    points = []
                    # progress_bar = st.progress(0)
                    # status_text = st.empty()

                    for idx, file in enumerate(uploaded_files):

                        # status_text(f"Processing image {idx+1}/{len(uploaded_files)}")
                        st.write(f"Processing image {idx+1}/{len(uploaded_files)}")

                        image = Image.open(file)

                        filename = f"{prefix}{idx}_{file.name}"
                        filepath = os.path.join(save_path, filename)
                        print(filepath)

                        os.makedirs(save_path, exist_ok=True)
                        image.save(filepath)

                        embedding = generate_embeddings(image, model, processor)

                        point = PointStruct(
                            id=idx + 10000, #using offset to avoid conflicts with existing ids
                            vector=embedding.tolist(),
                            payload={
                                "name":filename,
                                'price':0.0, #default price
                                'quantity':1, #default quantity
                                'image_name': filename
                            }
                        )
                        points.append(point)

                        # progress_bar.progress(idx+1/len(uploaded_files))

                    st.write("Uploading to Qdrant...")
                    # status_text.text("Uploading to Qdrant...")
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=points
                    )

                    st.success(f"Successfully processed and uploaded {len(uploaded_files)} images")
                    collection_info = qdrant_client.get_collection(collection_name)
                    st.info(f"Collection '{collection_name}' now has {collection_info.points_count} points")

                    # progress_bar.empty()
                    # status_text.empty()

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()  
