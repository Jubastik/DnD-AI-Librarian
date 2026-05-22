from qdrant_client import QdrantClient, models
from typing import List, Dict
from src.vector.interfaces import BaseVectorDB, VectorMode


class QdrantAdapter(BaseVectorDB):
    def __init__(self, collection_name: str,
                 path: str,
                 vector_dense_name: str = None,
                 embedding_size: int = None,
                 vector_sparse_name: str = None,
                 recreate_collection: bool = False,):
        self.client = QdrantClient(path=path)

        self.collection_name = collection_name
        self.vector_dense_name = vector_dense_name
        self.vector_sparse_name = vector_sparse_name

        exists = self.client.collection_exists(collection_name)
        if recreate_collection or not exists:
            self.create_collection(embedding_size)

    def create_collection(self,
                          embedding_size: int = None):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        if self.vector_dense_name is None and self.vector_sparse_name is None:
            raise ValueError("One of vectors should be used")
        if self.vector_dense_name is not None and embedding_size is None:
            raise ValueError("You're using dense vectors, embedding_size could not be None")

        dense_config = {
            self.vector_dense_name: models.VectorParams(
                size=embedding_size,
                distance=models.Distance.COSINE,
            )
        } if self.vector_dense_name is not None else None

        sparse_config = {
            self.vector_sparse_name: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        } if self.vector_sparse_name is not None else None

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=dense_config,
            sparse_vectors_config=sparse_config
        )

    def upsert(self, points: list):
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self,
               dense_vec: List = None,
               sparse_vec: models.SparseVector = None,
               limit: int = 5,
               filter_dict: Dict = None):

        q_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )
            q_filter = models.Filter(must=conditions)
        if dense_vec is not None and sparse_vec is not None:
            return self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_vec,
                        using=self.vector_dense_name,
                        filter=q_filter,
                        limit=limit * 2,
                    ),
                    models.Prefetch(
                        query=sparse_vec,
                        using=self.vector_sparse_name,
                        filter=q_filter,
                        limit=limit * 2,
                    ),
                ],

                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
            ).points
        elif dense_vec is not None:
            return self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vec,
                using=self.vector_dense_name,
                filter=q_filter,
                limit=limit,
            ).points
        elif sparse_vec is not None:
            return self.client.query_points(
                collection_name=self.collection_name,
                query=sparse_vec,
                using=self.vector_sparse_name,
                filter=q_filter,
                limit=limit,
            ).points
        else:
            raise ValueError("Не передан ни один вектор для поиска!")

    def close(self):
        """Явное закрытие соединения"""
        self.client.close()
