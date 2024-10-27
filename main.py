print("Hello World")


import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a .env file
load_dotenv()

# Retrieve Novita API key, and base URL from environment variables
novita_api_key = os.getenv("NOVITA_API_KEY")
base_url = os.getenv("BASE_URL")
