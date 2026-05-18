import os


class Settings:
    BASE_URL = "https://dnd.su"
    SITEMAP_URL = "https://dnd.su/sitemap.xml"
    QDRANT_PATH = os.path.join(os.getcwd(), "DND_db")
    COLLECTION_NAME = "dnd_collection"


    DATA_DIR = os.path.join(os.getcwd(), "data", "raw")

    # Фильтры URL
    # Ключ = категория для БД, Значение = подстрока в URL
    URL_FILTERS = {
        # "spells": "/spells/",
        # "classes": "/class/",
        "articles": "/articles/newbie/",
        "articles_mechanics": "/articles/mechanics/",
        "articles_inventory": "/articles/inventory/",
    }
    EXTRA_HUBS = [
        {
            "url": "/articles/newbie/",
            "category": "articles",
            "selector": "a.item-link"
        },

        {
            "url": "/articles/mechanics/",
            "category": "articles",
            "selector": "a.item-link"
        },

        {
            "url": "/articles/inventory/",
            "category": "articles",
            "selector": "a.item-link"
        },
    ]


settings = Settings()
os.makedirs(settings.DATA_DIR, exist_ok=True)