import os
import sys
from openai import AzureOpenAI

DATA_DIR = os.environ.get('DATA_DIR')
if DATA_DIR is None:
    raise EnvironmentError("The environment variable 'DATA_DIR' is not set.")

AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
if AZURE_OPENAI_API_KEY is None:
    raise EnvironmentError("The environment variable 'AZURE_OPENAI_API_KEY' is not set.")

AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
if AZURE_OPENAI_ENDPOINT is None:
    raise EnvironmentError("The environment variable 'AZURE_OPENAI_ENDPOINT' is not set.")

CLIENT = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),   # e.g. "https://<your-resource-name>.openai.azure.com"
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),           # Your Azure OpenAI key
        api_version="2024-10-01"                             # Example API version (use the one you have)
    )
