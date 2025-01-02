@echo off
echo Setting up semantic search app...

REM Create directories
echo Creating directories...
if not exist "images" mkdir images
if not exist "testimages" mkdir testimages

REM create network is doesn't exist
echo Creating network...
docker network create semantic_search_network 2>nul || ver > nul

REM stop and remove existing containers
echo Stopping and removing existing containers...
docker stop clip_search-app qdrant-server 2>nul || ver > nul
docker rm clip_search-app qdrant-server 2>nul || ver > nul

REM pulling the latest image
echo Pulling the latest image...
docker pull qdrant/qdrant
docker pull balavignesh26/clip-search:latest-1.0

REM running qdrant
echo Starting Qdrant...
docker run -d ^
    --name qdrant-server ^
    --network semantic_search_network ^
    -p 6333:6333 -p 6334:6334 ^
    -v "%cd%\qdrant_storage:/qdrant/storage" ^
    qdrant/qdrant

REM waiting for qdrant to start
echo Waiting for qdrant to start...
timeout /t 10 /nobreak > null

REM Running clip_search-app
echo Starting clip search app...
docker run -d ^
    --name clip_search-app ^
    --network semantic_search_network ^
    -p 8501:8501 ^
    -v "%cd%\images:/app/images" ^
    -v "%cd%\testimages:/app/testimages" ^
    -e QDRANT_HOST=qdrant-server ^
    -e QDRANT_PORT=6333 ^
    balavignesh26/clip-search:latest-1.0

echo Services are starting...
echo Access the Streamlit app at http://localhost:8501
echo Qdrant is running at http://localhost:6333

echo Showing logs (Ctrl+C to stop)...
docker logs -f clip_search-app

