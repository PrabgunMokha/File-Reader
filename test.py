from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

try:
    # FIX: Add 'models/' prefix to the model name
    response = client.models.generate_content(
        model="models/gemini-1.5-pro", 
        contents="Hello"
    )
    print(response.text)
except Exception as e:
    print(f"Error: {e}")
