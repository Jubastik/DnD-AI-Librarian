import os
import json
import hashlib
import time
from loguru import logger
from tqdm import tqdm
import uuid
from src.core.config import settings

# 1. Импорты Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

from langchain_text_splitters import RecursiveCharacterTextSplitter

QDRANT_PATH = os.path.join(os.getcwd(), "DND_db")
COLLECTION_NAME = "dnd_collection"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def main():
    logger.info("Loading AI Model...")
    dense_model = SentenceTransformer('intfloat/multilingual-e5-small')
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    client = QdrantClient(path=QDRANT_PATH)

    if client.collection_exists(collection_name=COLLECTION_NAME):
        logger.warning("Deleting old collection...")
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                )
            )
        }
    )

    logger.info(f"Connected to Qdrant at {QDRANT_PATH}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    files = [f for f in os.listdir(settings.DATA_DIR) if f.endswith('.json')]
    logger.info(f"Found {len(files)} files.")

    for filename in tqdm(files, desc="Processing"):
        filepath = os.path.join(settings.DATA_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue

        title = str(data.get("title", "Unknown"))
        content = data.get("content", "")
        category = str(data.get("category", "General"))
        url = str(data.get("url", f"https://dnd.su/{filename.replace('_', '/')}"))

        if not content.strip(): continue

        full_text = f"{content}"
        chunks = [full_text] if len(full_text) < 1200 else splitter.split_text(full_text)
        points = []
        for i, chunk in enumerate(chunks):
            text_to_embed = f"Документ: {filename}\nНазвание: {title}\nСодержание: {chunk}"

            dense_input = f"passage: {text_to_embed}"
            sparse_input = text_to_embed

            dense_vec = dense_model.encode(dense_input)

            sparse_result = list(sparse_model.embed([sparse_input]))[0]
            sparse_vec = models.SparseVector(
                indices=sparse_result.indices.tolist(),
                values=sparse_result.values.tolist()
            )

            point_id = str(uuid.uuid4())

            points.append(models.PointStruct(
                id=point_id,
                vector={
                    DENSE_VECTOR_NAME: dense_vec.tolist(),
                    SPARSE_VECTOR_NAME: sparse_vec
                },
                payload={
                    "title": title,
                    "text": text_to_embed,
                    "chunk_id": i,
                    "category": category,
                    "url": url
                }
            ))

        client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.success(f"Ingestion Complete! Check folder: {QDRANT_PATH}")


if __name__ == "__main__":
    main()
