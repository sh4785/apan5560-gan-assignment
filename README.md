# Assignment 3: GAN on MNIST + FastAPI + Docker

APANPS5560 Applied Generative AI, Module 6.

A GAN that generates handwritten digits, trained on MNIST and served through a FastAPI app in Docker.

## Project structure

```
gan-assignment/
├── helper_lib/
│   ├── __init__.py
│   ├── data_loader.py   # get_mnist_loader(): MNIST scaled to [-1, 1]
│   ├── model.py         # Generator, Discriminator, get_model("GAN")
│   ├── trainer.py       # train_gan(): alternating D / G updates
│   ├── generator.py     # generate_samples(), samples_to_png()
│   └── utils.py         # save_model(), load_model()
├── app/
│   └── main.py          # FastAPI endpoints
├── models/
│   └── gan_generator.pth  # trained weights (created by train_gan.py)
├── outputs/
│   └── sample_grid.png    # sample digits after training
├── train_gan.py
├── requirements.txt
└── Dockerfile
```

## Architecture

**Generator**: noise (B, 100) → Linear to 7×7×128 → reshape (B, 128, 7, 7) → ConvTranspose2d 128→64 (k4, s2, p1) → 14×14, BatchNorm2d, ReLU → ConvTranspose2d 64→1 (k4, s2, p1) → 28×28, Tanh.

**Discriminator**: image (B, 1, 28, 28) → Conv2d 1→64 (k4, s2, p1) → 14×14, LeakyReLU(0.2) → Conv2d 64→128 (k4, s2, p1) → 7×7, BatchNorm2d, LeakyReLU(0.2) → Flatten → Linear → 1 (Sigmoid probability).

Training: BCE loss, Adam (lr 2e-4, betas 0.5/0.999), batch size 128, 15 epochs.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_gan.py
uvicorn app.main:app --reload
```

## Run with Docker

```bash
docker build -t gan-api .
docker run -p 8000:8000 gan-api
```

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/` | Health check |
| POST | `/gan/generate` | Body `{"num_samples": 16, "seed": 7}` → JSON with base64 PNG grid |
| GET | `/gan/generate_image?num_samples=16&seed=7` | Returns the PNG directly (open in browser) |

Interactive docs: http://localhost:8000/docs

```bash
curl http://localhost:8000/
curl -X POST http://localhost:8000/gan/generate -H "Content-Type: application/json" -d '{"num_samples": 9, "seed": 7}'
curl "http://localhost:8000/gan/generate_image?num_samples=16&seed=7" --output digits.png
```
