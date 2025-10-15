from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables
load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ListingRequest(BaseModel):
    property_type: str
    bedrooms: int
    bathrooms: int
    features: str
    temperature: float = 0.3

# Detect if running on Render (Render sets a specific env var)
IS_RENDER = os.getenv("RENDER") is not None

@app.post("/generate-listing")
async def generate_listing(req: ListingRequest):
    prompt = (
        f"Write a factual and descriptive real estate listing for a "
        f"{req.bedrooms}-bedroom, {req.bathrooms}-bathroom {req.property_type}. "
        f"Focus only on accurate details. Avoid exaggerations or assumptions. "
        f"List features factually: {req.features}."
    )

    # ✅ If running on Render → non-streaming mode (fixes "no content" issue)
    if IS_RENDER:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that writes real estate listings. "
                        "Keep writing factual, neutral, and descriptive. "
                        "Avoid exaggerated or invented details."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=min(max(req.temperature, 0.0), 0.5),
        )

        content = response.choices[0].message.content.strip()
        return PlainTextResponse(content)

    # ✅ Otherwise, use streaming locally
    async def stream_response():
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that writes real estate listings. "
                        "Keep writing factual, neutral, and descriptive. "
                        "Avoid exaggerated or invented details."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=min(max(req.temperature, 0.0), 0.5),
            stream=True,
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.get("content", "")
            if content:
                yield content
            await asyncio.sleep(0)

    return StreamingResponse(stream_response(), media_type="text/plain")


@app.get("/")
async def root():
    return {"message": "AI Listing Writer backend is running!"}
