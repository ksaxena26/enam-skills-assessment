"""
Azure Key Vault secret retrieval.
Source: ### Secrets Retrieval / ## Retrieving Secrets sections.

Usage:
    from utils.secrets import get_secret
    api_key = get_secret("eodhd-api-key")

Requires AZURE_KEYVAULT_URL set as an environment variable (or in a .env file).
"""
# import os

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


def _get_client() -> SecretClient:
    # vault_url = os.environ.get("AZURE_KEYVAULT_URL")
    # if not vault_url:
    #     raise EnvironmentError("AZURE_KEYVAULT_URL is not set.")
    credential = DefaultAzureCredential()
    return SecretClient(
        vault_url  = "https://kv-markets-dev.vault.azure.net/",
        credential = credential
    )


def get_secret(secret_name: str) -> str:
    """Retrieve a secret value from Azure Key Vault by name."""
    client = _get_client()
    return client.get_secret(secret_name).value

tb_pass = get_secret("enam-tradebook")
openai_api_key = get_secret('openai-api-key')

if __name__ == "__main__":
    print(get_secret("enam-tradebook")[-4:])
    print(get_secret('openai-api-key')[-4:])