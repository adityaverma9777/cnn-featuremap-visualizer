# FastAPI Local AI Inference Backend

This backend serves a CIFAR-10 style PyTorch `SimpleCNN` model for image inference.

## Project Structure

- `app/main.py` - FastAPI server and model inference code
- `model/final_model.pth` - model weights file loaded at startup
- `requirements.txt` - Python dependencies

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Server will run on `localhost`.

## Predict Endpoint

`POST /predict`

- Form field: `file` (image upload)
- Preprocessing:
  - Resize to `32x32`
  - Convert to tensor
  - Normalize with mean `0.5` and std `0.5`

### Example response

```json
{
  "prediction": "dog",
  "confidence": 0.87,
  "class_probabilities": [
    { "class": "dog", "probability": 0.87 },
    { "class": "cat", "probability": 0.06 }
  ],
  "activations": {
    "conv1": [0.12, 0.04],
    "conv2": [0.21, -0.05],
    "conv3": [0.08, 0.32]
  },
  "heatmap": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

Activation arrays contain globally average-pooled values per channel from `conv1`, `conv2`, and `conv3`.
The `heatmap` field is a base64-encoded PNG Grad-CAM map generated from `conv3` for the predicted class and resized to the uploaded image dimensions.
The `class_probabilities` list contains per-class softmax scores, which can be used for top-k charts.

## React Frontend (Charts)

The React frontend is in `frontend/` and uses Recharts to display:

- Top 5 class probabilities
- Average activation values for `conv1`, `conv2`, and `conv3`

### Run frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the FastAPI server at `http://127.0.0.1:8000`.

## Deploy Both Services on Render

This repo includes `render.yaml` so Render can provision both services from GitHub:

- `ai-inference-backend` (Python web service)
- `ai-inference-frontend` (static site from `frontend/`)

### 1. Push this repo to GitHub

Render will deploy from your GitHub repository.

### 2. Create Blueprint in Render

1. In Render Dashboard, click **New +** -> **Blueprint**.
2. Connect your GitHub repo.
3. Render detects `render.yaml` and creates both services.

### 3. Configure Environment Variables

After first creation, set:

- Backend service:
  - `CORS_ALLOW_ORIGINS=https://<your-frontend>.onrender.com`
- Frontend service:
  - `VITE_API_BASE_URL=https://<your-backend>.onrender.com`

If you use custom domains, use those domain URLs instead.

### 4. Redeploy

Trigger a redeploy for both services after setting env vars.

### 5. Verify

- Open frontend URL from Render.
- Upload an image and run prediction.
- Confirm requests go to `https://<your-backend>.onrender.com/predict`.

## Config Files for Hosting

- Backend env template: `.env.example`
- Frontend env template: `frontend/.env.example`
- Render blueprint: `render.yaml`
