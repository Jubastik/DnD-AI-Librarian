from typing import List, Generator
from groq import Groq, RateLimitError, APIError
from src.llm.interfaces import BaseLLM, Message


class ModelRateLimitError(Exception):
    def __init__(self, model_name):
        self.message = f"Лимит запросов исчерпан для модели {model_name}."
        super().__init__(self.message)


class GroqLLM(BaseLLM):
    def __init__(self, api_key, model_name: str = "qwen-2.5-32b", temperature: float = 0.3):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature

    def chat_stream(self, messages: List[Message]) -> Generator[str, None, None]:
        try:
            stream = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=2048,
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        except RateLimitError:
            raise ModelRateLimitError(self.model_name)

        except APIError as e:
            raise Exception(f"Ошибка API Groq: {str(e)}")

    def chat_complete(self, messages: List[Message]) -> str:
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=1024,
                stream=False
            )
            return response.choices[0].message.content
        except RateLimitError:
            raise ModelRateLimitError(self.model_name)

        except APIError as e:
            raise Exception(f"Ошибка API Groq: {str(e)}")
