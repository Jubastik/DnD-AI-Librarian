import os
from qdrant_client import QdrantClient

from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models

QDRANT_PATH = os.path.join(os.getcwd(), "DND_db")
COLLECTION_NAME = "dnd_collection"
MODEL_NAME = "intfloat/multilingual-e5-small"

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def search(query_text, category_filter=None, limit=5):
    client = QdrantClient(path=QDRANT_PATH)
    dense_model = SentenceTransformer("intfloat/multilingual-e5-small")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    print(f"\n❓ Вопрос: {query_text}")

    dense_query_input = f"query: {query_text}"
    dense_vector = dense_model.encode(dense_query_input).tolist()

    sparse_result = list(sparse_model.embed([query_text]))[0]

    sparse_vector = models.SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist()
    )

    query_filter = None
    if category_filter:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=category_filter)
                )
            ]
        )

    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                filter=query_filter,
                limit=limit * 2,
            ),
            models.Prefetch(
                query=sparse_vector,
                using=SPARSE_VECTOR_NAME,
                filter=query_filter,
                limit=limit * 2,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
    )

    for hit in search_result.points:
        print(hit)
        score = hit.score
        payload = hit.payload
        print("-" * 40)
        print(f"📄 {payload['title']} ({payload['category']})")
        print(f"🔗 {payload['url']}")
        print(f"🎯 Сходство: {score:.4f}")
        print(f"📝 Отрывок:\n{payload['text'][:200]}...")
