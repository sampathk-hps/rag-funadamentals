# ====================
# 1. get response from OpenAI directly

# from openai import OpenAI
# client = OpenAI(api_key=openai_key)
# resp = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "What is human life expectancy in the United States?",
#         },
#     ],
# )
# print(resp.choices[0].message.content)

# ====================
# 2. get response from Local LLM using OpenAI

# from openai import OpenAI
# client = OpenAI(
#     base_url='http://localhost:11434/v1',  # Ollama's endpoint
#     api_key='ollama'  # Required but unused by Ollama
# )
# resp = client.chat.completions.create(
#     model="llama3.2:3b",  # Use any Ollama model you have installed
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "What is human life expectancy in the United States?"
#         },
#     ],
# )
# print(resp.choices[0].message.content)

# ====================
# 3. get response from Local LLM using ollama

# import ollama
# response = ollama.chat(
#     model='llama3.2:3b',
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "What is human life expectancy in the United States?"
#         },
#     ]
# )
# print(response.message.content)
