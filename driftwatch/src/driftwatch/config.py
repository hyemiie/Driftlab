from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    jwt_secret: str  # required, from .env, no default
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    max_upload_bytes: int = 20 * 1024 * 1024
    data_dir: str = "data"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()