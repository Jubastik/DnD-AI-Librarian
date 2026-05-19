from .spells import SpellParser
from .classes import ClassParser
from .common import CommonParser
from src.core.config import settings


class ParserFactory:
    @staticmethod
    def get_parser(url: str):
        # if settings.URL_FILTERS["spells"] in url:
        #     return SpellParser()
        # elif settings.URL_FILTERS["classes"] in url:
        #     return ClassParser()
        if settings.URL_FILTERS["articles"] in url or settings.URL_FILTERS["articles_mechanics"] in url or settings.URL_FILTERS["articles_inventory"] in url:
            return CommonParser(category="articles")
        else:
            # Дефолтный парсер, если добавишь новые категории в конфиг, но забудешь тут
            return CommonParser(category="misc")
