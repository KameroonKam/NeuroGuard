import os, requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

r = requests.get(
    f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
    timeout=30
)
print("status:", r.status_code)
print(r.text[:1000])
