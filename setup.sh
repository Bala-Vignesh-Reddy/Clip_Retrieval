#! /bin/bash

echo "setting up Semantic search app..."

#creating directories
echo 'creating directories...'
mkdir -p images testimages qdrant_storage

#crreate network if it doesn't exist
echo "creating network..."
docker network create semantic_search_network 2>/dev/null || true

echo "stopping and removing existing containers..."
docker stop clip_search-app qdrant-server 2>/dev/null || true
docker rm clip_search-app qdrant-server 2>/dev/null || true

#pulling the latest image
echo 'pulling the latest image...'
docker pull qdrant/qdrant
docker pull balavignesh26/clip-search:latest-1.0

#running qdrant
echo 'starting qdrant...'
docker run -d \
    --name qdrant-server \
    --network semantic_search_network \
    -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant

#waiting for qdrant to start
echo "waiting for qdrant to start..."
sleep 10

# running clip_search-app
echo "starting clip search app.."
docker run -d \
    --name clip_search-app \
    --network semantic_search_network \
    -p 8501:8501 \
    -v $(pwd)/images:/app/images \
    -v $(pwd)/testimages:/app/testimages \
    -e QDRANT_HOST=qdrant-server \
    -e QDRANT_PORT=6333 \
    balavignesh26/clip-search:latest-1.0

echo "services are starting..."
echo "Access streamlit app at http://localhost:8501"
echo "Qdrant is running at http://localhost:6333"

echo "showing logs (Ctrl+C to stop)"
docker logs -f clip_search-app
