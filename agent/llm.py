import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock

load_dotenv()

# Default model — can be overridden via .env
DEFAULT_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
)
DEFAULT_REGION = os.getenv("AWS_REGION", "us-east-1")


def get_llm(
    model_id: str = DEFAULT_MODEL_ID,
    region: str = DEFAULT_REGION,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> ChatBedrock:
    """
    Create and return a ChatBedrock LLM client.
    """
    try:
        llm = ChatBedrock(
            model_id=model_id,
            region_name=region,
            temperature= temperature,
            max_tokens=max_tokens,
        
        )
        return llm
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize Bedrock LLM. "
            f"Check AWS credentials/region in .env: {e}"
        ) from e


if __name__ == "__main__":
    llm = get_llm()
    response = llm.invoke("Say 'EDA agent LLM connection successful' in 5 words or less.")
    print(response.content)