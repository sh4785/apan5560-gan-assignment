"""FastAPI server exposing the trained MNIST GAN (Assignment 3, Part 2).

Run locally:  uvicorn app.main:app --reload
Docs:         http://localhost:8000/docs
"""
import base64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from helper_lib.model import Generator
from helper_lib.utils import load_model
from helper_lib.generator import generate_samples, samples_to_png

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gan_generator.pth"

app = FastAPI(title="Generative AI API", description="MNIST GAN digit generator")

generator = None
if MODEL_PATH.exists():
    generator = load_model(Generator(), str(MODEL_PATH), device="cpu")


def _get_generator():
    if generator is None:
        raise HTTPException(status_code=503, detail="GAN weights not found. Run `python train_gan.py` first.")
    return generator


class GanRequest(BaseModel):
    num_samples: int = Field(16, ge=1, le=64, description="How many digits to generate")
    seed: Optional[int] = Field(None, description="Optional random seed for reproducible output")


@app.get("/")
def root():
    return {"message": "API is running. Visit /docs to try the endpoints.",
            "gan_model_loaded": generator is not None}


@app.post("/gan/generate")
def gan_generate(req: GanRequest):
    """Generate digits and return the grid as a base64-encoded PNG (JSON response)."""
    imgs = generate_samples(_get_generator(), num_samples=req.num_samples, seed=req.seed)
    png = samples_to_png(imgs)
    return {
        "num_samples": req.num_samples,
        "seed": req.seed,
        "image_format": "png",
        "image_base64": base64.b64encode(png).decode("utf-8"),
    }


@app.get("/gan/generate_image", responses={200: {"content": {"image/png": {}}}})
def gan_generate_image(num_samples: int = Query(16, ge=1, le=64), seed: Optional[int] = None):
    """Generate digits and return the PNG image directly (open this URL in a browser)."""
    imgs = generate_samples(_get_generator(), num_samples=num_samples, seed=seed)
    return Response(content=samples_to_png(imgs), media_type="image/png")
