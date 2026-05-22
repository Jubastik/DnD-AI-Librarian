import uuid
from typing import List, Dict, Optional, Any
from qdrant_client.http import models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.vector.qdrant.qdrant_db import QdrantAdapter
from src.vector.interfaces import BaseEmbeder, VectorMode


class QdrantEngine:
    def __init__(self,
                 collection_name: str,
                 db_path: str,
                 embedder: BaseEmbeder,
                 mode: VectorMode,
                 dense_vec_name: str = None,
                 sparse_vec_name: str = None,
                 force_recreate: bool = False):
        self.embedder = embedder
        embed_size = self.embedder.get_embedding_dimension() if dense_vec_name is not None else None
        self.db = QdrantAdapter(collection_name, db_path, dense_vec_name, embed_size, sparse_vec_name, force_recreate)

        self.mode = mode

        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "]
        )

    def add_document(
            self,
            text: str,
            metadata: Dict[str, Any],
            batch_size: int = 50,
            custom_splitter=None
    ):
        """
        Метод добавления документа.
        1. Режет на чанки.
        2. Векторизует (Dense/Sparse/Hybrid).
        3. Формирует payload.
        4. Отправляет в БД батчами.
        """
        if not text.strip():
            return

        splitter = custom_splitter or self.default_splitter
        chunks = [text] if len(text) < splitter._chunk_size else splitter.split_text(text)

        points_batch = []

        doc_title = metadata.get("title", "")

        for i, chunk in enumerate(chunks):
            text_for_embed = f"{doc_title}. {chunk}" if doc_title else chunk

            vectors_dict = self.embedder.compute_vectors(text_for_embed, mode=self.mode, is_query=False)

            vectors_dict[self.db.vector_sparse_name] = models.SparseVector(
                indices=vectors_dict[self.db.vector_sparse_name]["indices"],
                values=vectors_dict[self.db.vector_sparse_name]["values"]
            )

            chunk_payload = metadata.copy()
            chunk_payload["text"] = chunk
            chunk_payload["text_for_embed"] = text_for_embed
            chunk_payload["chunk_id"] = i

            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors_dict,
                payload=chunk_payload
            )
            points_batch.append(point)

            if len(points_batch) >= batch_size:
                self.db.upsert(points_batch)
                points_batch = []

        if points_batch:
            self.db.upsert(points_batch)

    def find(self, query: str, filters: dict = None, limit: int = 5):
        vectors = self.embedder.compute_vectors(query, mode=self.mode, is_query=True)

        dense_vec = vectors.get(self.db.vector_dense_name)
        sparse_vec = vectors.get(self.db.vector_sparse_name)

        results = self.db.search(
            dense_vec=dense_vec,
            sparse_vec=sparse_vec,
            limit=limit,
            filter_dict=filters
        )

        return [
            {
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in results
        ]

    def close(self):
        if self.db:
            self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
