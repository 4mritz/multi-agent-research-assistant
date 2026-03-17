import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ollama import chat
from openai import OpenAI


logging.basicConfig(level=logging.INFO)


class BaseAgent:
    def __init__(self, model_name: str, system_prompt: str, expect_json: bool = False, temperature: float = 0.2):
        self.model_name = model_name
        self.expect_json = expect_json
        self.temperature = temperature
        self.provider, self.resolved_model = self._resolve_model(model_name)
        self.system_prompt = system_prompt + ("\nReturn ONLY valid JSON. No explanations." if expect_json else "")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = None
        if self.provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is required for OpenAI models.")
            self.client = OpenAI()

    def safe_json_parse(self, text: str) -> dict:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            self.logger.error("JSON parsing failed for %s: %s", self.__class__.__name__, exc)
            raise
        if not isinstance(data, dict):
            self.logger.error("JSON parsing failed for %s: top-level value must be an object", self.__class__.__name__)
            raise ValueError("Top-level JSON value must be an object.")
        return data

    def run(self, input_text: str) -> Any:
        self.logger.info("Input: %s", input_text)
        output = self._generate(input_text)
        self.logger.info("Output: %s", output)
        self._append_agent_log(input_text, output)
        if not self.expect_json:
            return output
        try:
            return self.safe_json_parse(output)
        except (json.JSONDecodeError, ValueError):
            retry_input = f"{input_text}\n\nYour previous response was invalid. Return ONLY valid JSON."
            retry_output = self._generate(retry_input)
            self.logger.info("Retry Output: %s", retry_output)
            self._append_agent_log(retry_input, retry_output)
            try:
                return self.safe_json_parse(retry_output)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError("Model failed to return valid JSON after retry.") from exc

    def _resolve_model(self, model_name: str) -> tuple[str, str]:
        if model_name.startswith("openai:"):
            return "openai", model_name.split("openai:", 1)[1]
        if model_name.startswith("ollama:"):
            return "ollama", model_name.split("ollama:", 1)[1]
        return "ollama", model_name

    def _messages(self, input_text: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": input_text}]

    def _generate(self, input_text: str) -> str:
        return self._run_ollama(input_text) if self.provider == "ollama" else self._run_openai(input_text)

    def _run_openai(self, input_text: str) -> str:
        response = self.client.responses.create(model=self.resolved_model, input=self._messages(input_text), temperature=self.temperature)
        return response.output_text

    def _run_ollama(self, input_text: str) -> str:
        response = chat(model=self.resolved_model, messages=self._messages(input_text), options={"temperature": self.temperature})
        return response["message"]["content"]

    def _append_agent_log(self, input_text: str, output: str) -> None:
        path = Path("logs/agent_logs.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "agent": self.__class__.__name__,
            "input": input_text,
            "output": output,
            "model": self.model_name,
            "provider": self.provider,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        except json.JSONDecodeError:
            data = []
        if not isinstance(data, list):
            data = []
        data.append(entry)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
