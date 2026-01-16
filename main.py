import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pinecone import Pinecone
from huggingface_hub import InferenceClient
from openai import OpenAI

load_dotenv()

# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# PINECONE SETUP
# -----------------------------
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("medicalbot")  # your Pinecone index

# -----------------------------
# EMBEDDINGS (HF Inference API)
# -----------------------------
HF_TOKEN = os.environ["HF_TOKEN"]
hf_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

def get_embedding(text: str):
    # Returns a vector list
    return hf_client.feature_extraction(text, model="sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# LLM (HF Router) SETUP
# -----------------------------
HF_API_KEY = os.environ["HF_API_KEY"]
llm_client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_API_KEY)
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SYSTEM_PROMPT = "You are a helpful medical assistant. You will short answer the questions if the question is according to context , if its not according to context just say 'its not my field of expertise' At start of your reply you will say 'According to Gale Encyclopedia of medicine'"

# -----------------------------
# SESSION MEMORY
# -----------------------------
sessions = {}

def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return sessions[session_id]

# -----------------------------
# CHAT LOGIC
# -----------------------------
def chat(query: str, session_id: str):
    messages = get_session(session_id)

    if len(messages) == 1:  # first query → add context from Pinecone
        vector = get_embedding(query)          # NumPy array
        vector = vector.tolist()               # convert to plain Python list
        results = index.query(vector=vector, top_k=4, include_metadata=True)
        context = "\n\n".join(m["metadata"].get("text", "") for m in results["matches"])
        query = f"Context:\n{context}\n\nQuestion:\n{query}"

    messages.append({"role": "user", "content": query})

    response = llm_client.chat.completions.create(model=MODEL, messages=messages)
    reply = response.choices[0].message.content

    messages.append({"role": "assistant", "content": reply})
    return reply


# -----------------------------
# ENDPOINTS
# -----------------------------
@app.get("/")
async def health():
    return {"status": "ok"}
@app.get("/chat")
def health():
    return {"reply": "you hit get, use post"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        chat_input = data.get("chatInput")
        session_id = data.get("sessionId")
        if not chat_input or not session_id:
            return {"reply": "Invalid input"}
        return {"reply": chat(chat_input, session_id)}
    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
