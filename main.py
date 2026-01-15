import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pinecone import Pinecone
from huggingface_hub import InferenceClient
from openai import OpenAI

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Pinecone
# --------------------
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("medicalbot")

# --------------------
# HF embeddings (NO TORCH)
# --------------------
hf_embed_client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

def get_embedding(text: str):
    embedding = hf_embed_client.feature_extraction(
        text,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 🔥 FORCE correct format for Pinecone
    if isinstance(embedding[0], list):
        embedding = embedding[0]

    return list(map(float, embedding))

# --------------------
# LLM (HF Router)
# --------------------
llm_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_API_KEY"]
)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SYSTEM_PROMPT = "You are a helpful medical assistant. You will answer the question to user according to context, it the question is not accoring to context just say 'Its not my field of expertese'"

sessions = {}

def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return sessions[session_id]

def chat(query: str, session_id: str):
    messages = get_session(session_id)

    if len(messages) == 1:
        query_vector = get_embedding(query)
        results = index.query(
            vector=query_vector,
            top_k=4,
            include_metadata=True
        )

        context = "\n\n".join(
            m["metadata"].get("text", "") for m in results["matches"]
        )

        query = f"Context:\n{context}\n\nQuestion:\n{query}"

    messages.append({"role": "user", "content": query})

    response = llm_client.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return reply

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    if not data.get("chatInput") or not data.get("sessionId"):
        return {"reply": "Invalid input"}

    return {"reply": chat(data["chatInput"], data["sessionId"])}

@app.get("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    if not data.get("chatInput") or not data.get("sessionId"):
        return {"reply": "Invalid input"}

    return {"reply": chat(data["chatInput"], data["sessionId"])}
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )