import requests

URL = "https://medicalbotapi-production.up.railway.app/chat"

data = {
    "chatInput": "Hello world",
    "sessionId": "test123"
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(URL, json=data, headers=headers)

try:
    result = response.json()
    print("Reply:", result.get("reply"))
except Exception as e:
    print("Error:", e)
    print("Response text:", response.text)
