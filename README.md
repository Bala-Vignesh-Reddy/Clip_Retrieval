# Semantic Image Search with CLIP and Qdrant

This project implements a semantic search system using CLIP embeddings and Qdrant vector database. It allows users to search through image collections using either text descriptions or similar images.

## Installation 

### Prerequisites
- Docker
- Git
- Python

### Windows
1. Download or clone this repo
```cmd
git clone https://github.com/Bala-Vignesh-Reddy/Clip_Retrieval.git
cd Clip_Retrieval
```
2. Run the setup script
```cmd
setup.bat
```
Or double click `setup.bat` in file explorer.

### Linux
1. Download or clone this repo
```bash
git clone https://github.com/Bala-Vignesh-Reddy/Clip_Retrieval.git
cd Clip_Retrieval
```
2. Run the setup script
```bash
./setup.sh
```
Or double click `setup.sh` in file explorer.

## Usage
### Access the application:
    - Web UI: http://localhost:8501
    - Qdrant: http://localhost:6333

## Features
- Text-to-Image search
- Image-to-Image search
- Multiple collection support
- Bulk image upload
- Semantic similarity search using CLIP embeddings
- Vector storage with Qdrant

## Tech Stack
- **Model**: OpenAI CLIP (clip-vit-base-patch32)
- **Vector Database**: Qdrant
- **Framework**: Streamlit
- **Image Processing**: Pillow
- **Deep Learning**: PyTorch, Transformers

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Bala-Vignesh-Reddy/ClipRetrieval.git
cd ClipRetrieval
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate # for windows use venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run testScripts/app.py
```

### 5. Qdrant Setup (Docker is necessary) - for linux..  for windows change the pwd to the path for the folder you want to store the data
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

## Project Structure

```
└── ClipRetrieval/
    ├── images
    ├── qdrant_storage
    ├── testScripts/
    │   ├── pages/
    │   │   ├── 1_Bulk_Upload_testing.py
    │   │   └── 2_Description.py
    │   ├── app.py
    │   ├── embeddings.py
    │   ├── image_check.py
    │   ├── qdrant_query.py
    │   └── qdrant_upload.py
    ├── .gitignore
    ├── README.md
    ├── requirements.txt
    └── test.py
```

## Usage Instructions

### 1. Collections
- **image_search**: Default collection for main images
- **test_collection**: For testing and development

### 2. Bulk Upload
1. Navigate to "Bulk Upload Testing"
2. Choose images to upload
3. Set save directory and prefix
4. Process and upload images

### 3. Searching
- **Text Search**: Enter descriptive text
- **Image Search**: Upload a similar image
- Switch collections using the dropdown

### Key Files
- `Home.py`: Main application and search interface
- `qdrant_query.py`: Query processing and embedding generation
- `1_bulk_upload.py`: Bulk image upload functionality

### Key Functions
- `generate_embeddings()`: Creates CLIP embeddings
- `do_search()`: Handles search queries
- `qdrant_query()`: Interfaces with Qdrant


