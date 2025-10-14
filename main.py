from fastapi import FastAPI
from pydantic import BaseModel
import openai, os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Later restrict to your frontend domain
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
    prompt = f"""
    Write a compelling real estate listing for a {req.bedrooms}-bedroom, {req.bathrooms}-bathroom {req.property_type}
    with features: {req.features}.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return {"listing": response.choices[0].message.content.strip()}
