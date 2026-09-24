from openai import AsyncAzureOpenAI
from app.core.config import settings

client = AsyncAzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    api_version=settings.AZURE_OPENAI_API_VERSION,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
)

async def get_embedding(text: str) -> list[float]:
    response = await client.embeddings.create(
        input=text,
        model=settings.AZURE_EMBEDDING_DEPLOYMENT
    )
    embedding = response.data[0].embedding
    if len(embedding) != settings.EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Embedding dimension mismatch: expected {settings.EMBEDDING_DIMENSIONS}, "
            f"got {len(embedding)}"
        )
    return embedding
