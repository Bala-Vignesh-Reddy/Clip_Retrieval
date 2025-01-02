import streamlit as st
from qdrant_client import QdrantClient
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
from qdrant_query import generate_embeddings, qdrant_query
from dotenv import load_dotenv

load_dotenv()
QDRANT_HOST = os.getenv('QDRANT_HOST')
QDRANT_PORT = os.getenv('QDRANT_PORT')

st.set_page_config(page_title="Semantic Search", layout="wide")

@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

@st.cache_resource
def qdrant_init():
    # try:
    #     client = QdrantClient(
    #         url=f"http://{os.getenv('QDRANT_HOST', 'qdrant')}:{os.getenv('QDRANT_PORT', '6333')}"
    #     )
    #     # Test the connection
    #     client.get_collections()
    #     return client
    # except Exception as e:
    #     st.error(f"Failed to connect to Qdrant: {str(e)}")
    #     st.error("Make sure Qdrant is running and accessible")
    #     return None
    qdrant_client = QdrantClient(host="localhost", port=6333)
    return qdrant_client

def do_search(query, model, processor, qdrant_client, collection_name, is_image_query=False):
    if query:
            with st.spinner("Searching..."):
                try:
                    image_directory = './images' # default path for image directory

                    if is_image_query:
                        if query.mode != 'RGB':
                            query = query.convert('RGB')
                    else:
                        query = str(query)
            
                    query_embedding = generate_embeddings(
                        query, model, processor, is_image_query=is_image_query
                    )
                    
                    search_results = qdrant_query(qdrant_client, collection_name, query_embedding)

                    if not search_results:
                        st.warning("No results found")
                        return
                    
                    cols = st.columns(3)
                    for idx, result in enumerate(search_results):
                        col_idx = idx % 3
                        with cols[col_idx]:
                            metadata = result.payload
                            score = result.score

                            if collection_name == "image_search":
                                img_path = os.path.join(image_directory, metadata["image_name"])
                            else:
                                img_path = os.path.join("./testimages", metadata["image_name"])
                            # print(img_path)

                            if os.path.exists(img_path):
                                result_image = Image.open(img_path)
                                
                                if not is_image_query:
                                    result_image.thumbnail((200, 200))
                                    st.image(result_image, width=200)
                                else:
                                    result_image.thumbnail((200, 200)) 
                                    st.image(result_image, use_container_width=True)

                                # Make metadata display more compact
                                st.markdown(f"""
                                    <div style='font-size: 0.9em'>
                                        <b>{metadata['name']}</b><br>
                                        Price: ₹{metadata['price']:.2f}<br>
                                        Qty: {metadata['quantity']}<br>
                                        <small>Score: {score:.4f}</small>
                                    </div>
                                """, unsafe_allow_html=True)
                                st.write("---")
                            else:
                                # st.error("Image not found")
                                alt_path = os.path.join("./images", metadata["image_name"])
                                if os.path.exists(alt_path):
                                    result_image = Image.open(alt_path)
                                    result_image.thumbnail((200, 200))
                                    st.image(result_image, width=200)
                                else:
                                    result_image.thumbnail((200, 200))
                                    st.image(result_image, use_container_width=True)
                                st.markdown(f"""
                                    <div style='font-size: 0.9em'>
                                        <b>{metadata['name']}</b><br>
                                        Price: ₹{metadata['price']:.2f}<br>
                                        Qty: {metadata['quantity']}<br>
                                        <small>Score: {score:.4f}</small>
                                    </div>
                                """, unsafe_allow_html=True)
                                st.write("---")
                except Exception as e:
                    st.error(f"An error occured during search: {str(e)}")
    else:
        st.warning("Please enter a query to search.")


def main():
    st.title("Semantic Search")

    model, processor = load_model()
    qdrant_client = qdrant_init()

    with st.sidebar:
        st.title("Search Options")

        # collection selector
        collections = qdrant_client.get_collections().collections
        collection_name = [col.name for col in collections]
        # print(collection_name)
        selected_collection = st.selectbox("Select Collection", collection_name, index=0)

        search_type = st.radio("Choose search type", ["Text", "Image"])

        # st.markdown("---")
        # if st.button("Upload New Images", type="secondary"):
        #     st.switch_page("1_bulk_upload")

        st.markdown("---")
        st.markdown("""
        ### How to use:
        - **Text Search**: Enter text
        - **Image Search**: Upload a similar image    
        """)
        st.markdown("---")

    if search_type == "Text":
        st.subheader("Text-Based Search")
        query = st.text_input("Enter the query:", key="query_input")

        if st.button("Search") or query:
            do_search(query, model, processor, qdrant_client, selected_collection, is_image_query=False)
    else:
        st.subheader("Image-Based Search")
        uploaded_file = st.file_uploader("Upload an image to find similar items", 
                                         type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1,2])
            with col1:
                image = Image.open(uploaded_file)
                image.thumbnail((200,200))
                st.image(image, caption="Uploaded Image", use_container_width=True)

            with col2:
                st.markdown("""Image uploaded successfully
                            Click the button below to search""")
                if st.button("Search", type="primary"):
                    do_search(image, model, processor, qdrant_client, selected_collection, is_image_query=True)


if __name__ == "__main__":
    main()