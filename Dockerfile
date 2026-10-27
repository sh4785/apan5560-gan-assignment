FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch keeps the image much smaller than the default CUDA build
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Copy code + trained weights (models/gan_generator.pth must exist before building)
COPY helper_lib/ helper_lib/
COPY app/ app/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
