import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

print("MINIMAX_KEY =", os.getenv("MINIMAX_KEY"))
print("OPENAI_BASE_URL =", os.getenv("OPENAI_BASE_URL"))
print("OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))

# 🟡 Assure-toi qu'OPENAI_API_KEY et OPENAI_BASE_URL sont bien dans l’environnement
client = OpenAI()

resp = client.chat.completions.create(
    model="MiniMax-M2.5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Test message"}
    ],
    max_tokens=200
)

print(resp.choices[0].message.content)