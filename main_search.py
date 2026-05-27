from src.vector.qdrant.qdrant_engine import QdrantEngine
from src.vector.embedders import DenseMultilingualE5SmallSparseBm25Embeder
from src.core.config import settings

embeder = DenseMultilingualE5SmallSparseBm25Embeder()
with QdrantEngine(settings.COLLECTION_NAME,
                  settings.QDRANT_PATH,
                  embeder,
                  "hybrid",
                  "dense",
                  "sparse") as engine:
    query = "Заклинание Огненный шар"

    results = engine.find(query)

    for hit in results:
        score = hit["score"]
        payload = hit["payload"]
        print("-" * 40)
        print(hit)
        print(f"📄 {payload['title']} ({payload['category']})")
        print(f"🔗 {payload['url']}")
        print(f"🎯 Сходство: {score:.4f}")
        print(f"📝 Отрывок:\n{payload['text'][:200]}...")
