from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

@app.post("/generate-listing")
async def generate_listing(req: ListingRequest):
    prompt = (
        f"Write a compelling real estate listing for a "
        f"{req.bedrooms}-bedroom, {req.bathrooms}-bathroom {req.property_type} "
        f"with features: {req.features}."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional real estate copywriter."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return {"listing": response.choices[0].message.content.strip()}
