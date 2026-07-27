from pydantic_settings import BaseSettings
import dotenv
import os
dotenv.load_dotenv()
class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://specwars_user:specwars_password@localhost:5433/specwars_db"
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_EMBEDDING_DEPLOYMENT: str = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

settings = Settings()
