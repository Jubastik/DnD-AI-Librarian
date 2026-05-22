from typing import Dict, Any, Literal
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client.http import models
from src.vector.interfaces import BaseEmbeder, VectorMode


class DenseMultilingualE5SmallSparseBm25Embeder(BaseEmbeder):
    def __init__(self):
        self.dense_model = SentenceTransformer("intfloat/multilingual-e5-small")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        self.supported_modes = ["hybrid", "dense", "sparse"]

    def _get_dense(self, text: str, is_query: bool) -> list:
        prefix = "query: " if is_query else "passage: "
        return self.dense_model.encode(f"{prefix}{text}").tolist()

    def _get_sparse(self, text: str) -> dict[str, object]:
        result = list(self.sparse_model.embed([text]))[0]
        return {
            "indices": result.indices.tolist(),
            "values": result.values.tolist()
        }
        # return models.SparseVector(
        #     indices=result.indices.tolist(),
        #     values=result.values.tolist()
        # )

    def get_embedding_dimension(self):
        return self.dense_model.get_sentence_embedding_dimension()

    def compute_vectors(self,
                        text: str,
                        mode: VectorMode = "hybrid",
                        dense_vec_name: str = "dense",
                        sparse_vec_name: str = "sparse",
                        is_query: bool = False) -> Dict[str, Any]:
        """
        Универсальный метод: возвращает словарь {имя_вектора: вектор}
        в зависимости от выбранного режима.
        """
        vectors = {}

        if mode.lower() not in self.supported_modes:
            raise ValueError(f"Mode {mode} is not supported")

        mode = mode.lower()

        if mode in ["hybrid", "dense"]:
            vectors[dense_vec_name] = self._get_dense(text, is_query)

        if mode in ["hybrid", "sparse"]:
            vectors[sparse_vec_name] = self._get_sparse(text)

        return vectors
