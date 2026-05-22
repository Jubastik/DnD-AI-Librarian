import os
import json
from src.core.config import settings
from src.vector.qdrant.qdrant_engine import QdrantEngine
from src.vector.embedders import DenseMultilingualE5SmallSparseBm25Embeder
from tqdm.auto import tqdm


def main():
    print("Initializing...")
    embeder = DenseMultilingualE5SmallSparseBm25Embeder()
    engine = QdrantEngine(settings.COLLECTION_NAME,
                          settings.QDRANT_PATH,
                          embeder,
                          "hybrid",
                          "dense",
                          "sparse")

    print("Initializing finished")
    files = [f for f in os.listdir(settings.DATA_DIR) if f.endswith('.json')]

    for filename in tqdm(files):
        filepath = os.path.join(settings.DATA_DIR, filename)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue

        engine.add_document(
            text=data.get("content", ""),
            metadata={
                "title": data.get("title", "Unknown"),
                "url": data.get("url"),
                "category": data.get("category", "General"),
                "source": "official"
            },
        )

    print("Загрузка завершена!")


if __name__ == "__main__":
    main()
