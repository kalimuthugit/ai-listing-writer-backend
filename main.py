from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables
load_dotenv()

# Initialize OpenAI client (async version for streaming)
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize FastAPI
app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class ListingRequest(BaseModel):
    property_type: str
    bedrooms: int
    bathrooms: int
    features: str
    temperature: float = 0.3  # Default lower value for factual tone

# ---------------------------
# 🚀 STREAMING GENERATION ENDPOINT
# ---------------------------
@app.post("/generate-listing")
async def generate_listing(req: ListingRequest):
    # Build the factual prompt
    prompt = (
        f"Write a factual and descriptive real estate listing for a "
        f"{req.bedrooms}-bedroom, {req.bathrooms}-bathroom {req.property_type}. "
        f"Focus only on accurate details. Avoid exaggerations or assumptions. "
        f"List features factually: {req.features}."
    )

    async def stream_response():
        # Use streaming to send chunks to frontend as they are generated
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that writes real estate listings. "
                        "Keep the writing factual, neutral, and descriptive. "
                        "Avoid creative or exaggerated language, and do not invent details."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=min(max(req.temperature, 0.0), 0.5),  # Clamp between 0.0–0.5
            stream=True,
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.get("content", "")
            if content:
                yield content
            await asyncio.sleep(0)  # let the event loop breathe

    return StreamingResponse(stream_response(), media_type="text/plain")

# ---------------------------
# HEALTH CHECK ENDPOINT
# ---------------------------
@app.get("/")
async def root():
    return {"message": "AI Listing Writer backend is running!"}
