import os
import boto3
from langchain_aws import ChatBedrockConverse
from dotenv import load_dotenv


class BedrockLLM:
    def __init__(self, model):
        load_dotenv()
        self._model = model
        self._access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self._secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self._session_token = os.getenv("AWS_SESSION_TOKEN")
        self._region = os.getenv("AWS_REGION", "eu-west-1")

    def get_llm(self):
        try:
            if not self._access_key or not self._secret_key:
                raise ValueError("AWS credentials are not set.")
            if not self._model:
                raise ValueError("LLM_MODEL is not set.")

            bedrock_client = boto3.client(
                "bedrock-runtime",
                aws_access_key_id=self._access_key,
                aws_secret_access_key=self._secret_key,
                aws_session_token=self._session_token,
                region_name=self._region,
            )

            return {
                "llm_provider": "aws",
                "llm_model": ChatBedrockConverse(
                    model_id=self._model,
                    client=bedrock_client,
                    temperature=0.1,
                    timeout=10,
                ),
            }

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Bedrock LLM: {e}")
