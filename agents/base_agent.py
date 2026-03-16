import logging

from ollama import chat
from openai import OpenAI


logging.basicConfig(level=logging.INFO)


class BaseAgent:
    def __init__(self, model_name: str, system_prompt: str):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.logger = logging.getLogger(self.__class__.__name__)
        self.provider = "ollama" if model_name.startswith("ollama:") else "openai"
        self.client = OpenAI() if self.provider == "openai" else None

    def run(self, input_text: str) -> str:
        self.logger.info("Input: %s", input_text)
        output = self._run_ollama(input_text) if self.provider == "ollama" else self._run_openai(input_text)
        self.logger.info("Output: %s", output)
        return output

    def _messages(self, input_text: str):
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text},
        ]

    def _run_openai(self, input_text: str) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input=self._messages(input_text),
        )
        return response.output_text

    def _run_ollama(self, input_text: str) -> str:
        response = chat(
            model=self.model_name.split("ollama:", 1)[1],
            messages=self._messages(input_text),
        )
        return response["message"]["content"]
