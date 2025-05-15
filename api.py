import os
from typing import AsyncGenerator, Optional
from fastapi import Body, FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from langchain_milvus import Zilliz
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="SDPBot API", description="AI Chatbot for Smart Diet Planner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Embedding
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

class UserGoals(BaseModel):
    calories: float
    carbs: float
    fat: float
    protein: float
    
class UserBody(BaseModel):
    name: str
    age: int | None
    community: list
    goal: UserGoals
    foodType: list
    conditions: list

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    knowledge_base: str
    user_input: str
    # user_data: UserBody | None
    user_data: Optional[str] = None
    microNutrientFlag: Optional[str] = None

def get_vector_store(document_id: str):
    return Zilliz(
        collection_name=document_id,
        connection_args={"uri": os.getenv("ZILLIZ_URI_ENDPOINT"), "token": os.getenv("ZILLIZ_TOKEN")},
        index_params={"index_type": "IVF_PQ", "metric_type": "COSINE"},
        embedding_function=embeddings,
    )

def promptWithUserData(query: str, context: str, data: str):
    return f"""
    You are a highly knowledgeable and empathetic nutritionist assistant.
    Based on the given user data about his diseases and conditions he is suffering from and the reference context, give me a list of Micronutrients the user must consume based on his data
    only return the list of Top 5 Micronutrients you would suggest to the user based on his condition and NOTHING ELSE.
    Query: {query}
    Context: {context}
    User Data: {data}
    """
    
def getBotResponse(prompt: str):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5,
        stream=True,
    )
    return response

@app.get("/")
async def Home():
    return {"message": "Its Live"}

@app.post("/chat")
async def chat(request: ChatRequest):
    knowledgeBase = request.knowledge_base
    userQuery = request.user_input
    flag = request.microNutrientFlag
    if flag and flag == True:
        userData = request.user_data
        vector_store = get_vector_store(knowledgeBase)
        filterPK = "micronutrients"
        retrieved_docs = vector_store.similarity_search(
            query=userQuery, k=1, filter={"chunk_category": filterPK}
        )
        context = retrieved_docs[0].page_content.strip() or ""
        prompt = promptWithUserData(query=userQuery, context=context, data=userData)
        response = getBotResponse(prompt)
        async def stream_response() -> AsyncGenerator[str, None]:
            try:
                for chunk in response:
                    if chunk.choices:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content
            except Exception as e:
                yield f"Error: {str(e)}"
        
        return StreamingResponse(stream_response(), media_type="text/plain")
    
    vector_store = get_vector_store(knowledgeBase)
    retrieved_docs = vector_store.similarity_search_with_relevance_scores(
        query=userQuery, k=3, score_threshold=0.75
    )

    context = "\n".join(doc[0].page_content.strip() for doc in retrieved_docs) if retrieved_docs else ""

    prompt = f"""
    You are a highly knowledgeable and empathetic nutritionist assistant.
        Your role is to provide clear, evidence-based answers using below retrieved context from a trusted knowledge base.
        Always keep responses concise (under 250 words), accurate, and user-friendly.
        Never disclose your data source or say "Based on the document..., In the provided context..." etc.
        Question: {userQuery}
        Context: {context}
        Do not any extra information from your own knowledge base.
        If Context lacks relevant information to answer the question than deny the user politely explain No relevant Context was found for their question. Do not guess or answer from your own knowledge.
    """

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5,
        stream=True,
    )

    async def stream_response() -> AsyncGenerator[str, None]:
        try:
            for chunk in response:
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
        finally:
            client.close()
    return StreamingResponse(stream_response(), media_type="text/plain")

@app.delete("/delete")
async def delete_chunk(data: dict = Body(...)):
    knowledgeBase = data.get("knowledgeBase")
    partitionKey = data.get("partitionKey")
    file_ids = [f"{partitionKey}_{i}" for i in range(3855)]
    vector_store = get_vector_store(knowledgeBase)
    vector_store.delete(ids= file_ids)
    return {"message": "Chunk deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="localhost", port=8000, reload=True)