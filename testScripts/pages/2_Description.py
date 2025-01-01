import streamlit as st

st.title("Description")

st.write("This is the description page..")
st.markdown("---")

st.header("Overview")
st.write("""
This projcet is a semantic search system using CLIP(Contrastive Language Image Processing) model 
that allows users to search through text or image query. This system uses Qdrant as a vector database
to store and retrieve the embeddings of the images.
""")

st.header("Key Components")
st.write("""
    1. Uses OpenAi's CLIP model(clip-vit-base-patch32)
    2. Generates 512 dimensional embeddings for both images and text
    3. Provides a multi-modal search system that allows users to search through text or image query
    4. Uses Qdrant as a vector database to store and retrieve the embeddings of the images
    5. Currently the metadata (name, description, price, quantity, image_path) is stored in the vector database
""")    

st.header("Workflow")
st.write("""
    1. Text-Based Search:
        - Enter the query in the text box and click search it will display the result with metadata.
    2. Image-Based Search:
        - Upload the image in the sidebar and the system will display the most similar images from the database.

    3. For Bulk Image Upload and processing:
        - upload multiple images
        - it will generate embeddings and save the images to the selected folder and store the embeddings in the test_collection.
        - then in the app page.. you can select the test_collection and search accordingly.. This make it a real-time search system.
""")

st.header("Scripts")
st.write("""
    1. app.py - streamlit app
         - code for the streamlit app
         - Features:
            - sidebar having search options
            - collection selector
            - choosing search result
    2. pages/1_bulk_upload.py - bulk upload of images
         - code for bulk upload of images
         - Features:
            - upload multiple images
            - generate embeddings
            - save images to specified folder
            - store embeddings in Qdrant
    3. qdrant_query.py - query the vector database
         - handles the query to the vector database
         - processes the both text and image inputs
         - displaying the search results
         - loading metadata for a selected image 

        - parameters:
            - qdrant_client: qdrant client object
            - collection_name: name of the collection to query
            - query_embedding: embedding of the query
            - top_k: number of most similar vectors to return
""")

