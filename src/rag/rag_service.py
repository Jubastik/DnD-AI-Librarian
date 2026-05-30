from typing import List, Generator, Optional
from src.vector.qdrant.qdrant_engine import QdrantEngine
from src.llm.interfaces import BaseLLM
from loguru import logger

DEFAULT_SYSTEM_PROMPT_TEMPLATE = """
Ты — Мастер Подземелий. Отвечай только на основе контекста, не выдумывай ничего сам.

Контекст:
{context}
"""

REWRITER_PROMPT = """
Ты — модуль предварительной обработки поисковых запросов для базы знаний D&D 5e.
Твоя задача — проанализировать ТЕКУЩИЙ вопрос пользователя и историю диалога.

АЛГОРИТМ ДЕЙСТВИЙ:
1. Проанализируй: Является ли текущий вопрос продолжением предыдущей темы (содержит "он", "это", "у него", "а какой радиус" и т.д.)?
2. ЕСЛИ ЭТО ПРОДОЛЖЕНИЕ: Переформулируй вопрос, заменив местоимения на конкретные названия из истории, чтобы вопрос стал самодостаточным.
3. ЕСЛИ ЭТО НОВАЯ ТЕМА (пользователь спрашивает про что-то другое): Верни текущий вопрос БЕЗ ИЗМЕНЕНИЙ. Не пытайся притянуть старый контекст.

ПРИМЕРЫ:
История: [User: Расскажи про Гоблина.]
Текущий: Какой у него КД?
Результат: Какой КД у Гоблина?

История: [User: Расскажи про Гоблина.]
Текущий: Расскажи про заклинание Огненный шар.
Результат: Расскажи про заклинание Огненный шар. (История проигнорирована)

ВЕРНИ ТОЛЬКО ИТОГОВЫЙ ТЕКСТ ВОПРОСА.
"""


class DndRagService:
    def __init__(self,
                 engine: QdrantEngine,
                 main_llm: BaseLLM,
                 rewriter_llm: BaseLLM = None,
                 system_prompt_template: str = DEFAULT_SYSTEM_PROMPT_TEMPLATE,
                 rewriter_system_prompt: str = REWRITER_PROMPT
                 ):

        self.engine = engine
        self.main_llm = main_llm
        self.rewriter_llm = rewriter_llm or main_llm

        self.system_prompt_template = system_prompt_template
        self.rewriter_system_prompt = rewriter_system_prompt

    def _rewrite_query(self, query: str, history: List[dict]) -> str:
        """Переписывает запрос, используя быструю модель"""
        if not history:
            return query

        messages = [
            {"role": "system", "content": self.rewriter_system_prompt},
        ]
        messages.extend(history[-2:])
        messages.append({"role": "user", "content": query})

        return self.rewriter_llm.chat_complete(messages)

    def find_documents(self, query: str, chat_history: list = None):
        search_query = query
        if chat_history:
            search_query = self._rewrite_query(query, chat_history)
        logger.info(f"🔎 Поисковый запрос: {search_query}")

        results = self.engine.find(search_query, limit=5)

        return results

    def generate_answer(self, query: str, chat_history: list = None, return_sources: bool = False) -> Generator[
        str, None, None]:
        results = self.find_documents(query, chat_history)

        if return_sources:
            sources = [{"title": r['payload'].get("title", ''), "url": r['payload'].get("url", '')} for r in results]
            yield {"type": "sources", "sources": sources}

        context_str = "\n".join([f"- {h['payload'].get('text', '')}" for h in results])

        system_msg = self.system_prompt_template.format(context=context_str)
        messages = [{"role": "system", "content": system_msg}]
        if chat_history:
            messages.extend(chat_history[-4:])
        messages.append({"role": "user", "content": query})

        yield from self.main_llm.chat_stream(messages)
