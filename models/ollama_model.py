import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama


class OllamaLLM:
    def __init__(self, model):
        load_dotenv()
        self._model = model

    def get_llm(self):
        """Initialize and return the Ollama Embedding model."""
        try:
            if not self._model:
                raise ValueError("LLM_MODEL environment variable is not set.")

            return {
                "llm_provider": "ollama",
                "llm_model": ChatOllama(
                    model=self._model,
                    temperature=0.1,
                    num_predict=256,
                    timeout=10,
                    # other params ...
                ),
            }

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Ollama Embedding model: {e}")
