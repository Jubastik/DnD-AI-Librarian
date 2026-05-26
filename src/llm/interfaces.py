from abc import ABC, abstractmethod
from typing import List, Dict, Generator, Optional

Message = Dict[str, str]


class BaseLLM(ABC):
    @abstractmethod
    def chat_stream(self, messages: List[Message]) -> Generator[str, None, None]:
        """
        Метод должен принимать историю сообщений и возвращать генератор токенов (строк).
        """
        pass

    @abstractmethod
    def chat_complete(self, messages: List[Message]) -> str:
        """
        Метод для получения полного ответа сразу (удобно для внутренних задач типа рерайта).
        """
        pass
