from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="123321",   
)

completion = client.chat.completions.create(
  model="Qwen/Qwen3-4B",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)