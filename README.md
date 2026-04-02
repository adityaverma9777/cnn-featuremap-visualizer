# CNN Feature Map Visualizer

Understand how the CNN is actually making decisions.

This is a local image-recognition + visualization project with:

- FastAPI backend for inference
- React + Vite frontend for interactive visualization
- Trained CNN model at `model/final_model.pth`

## Dataset

The model was trained on CIFAR-10 in:

- `Colab Session/imagerecognisitionmodeltraining.ipynb`

CIFAR-10 classes:

- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Run Locally (PowerShell)

### 1) Clone

```bash
git clone https://github.com/adityaverma9777/cnn-featuremap-visualizer.git
cd cnn-featuremap-visualizer
```

### 2) Start backend + frontend together

From project root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\start-backend.ps1
```

The launcher script automatically:

- checks Python and npm
- installs backend dependencies if missing
- installs frontend dependencies if missing
- finds a free backend port (starting from 8000)
- starts backend in one PowerShell window
- starts frontend in another PowerShell window
- auto-connects frontend to the selected backend port

## URLs

- Backend URL is printed by launcher (for example `http://127.0.0.1:8001` if 8000 is blocked)
- Frontend is usually `http://127.0.0.1:5173`

## Verify It Works

1. Open frontend URL shown by Vite.
2. Upload an image.
3. Run prediction and check output:
   - predicted class
   - confidence
   - class probabilities
   - activation/feature visualization
4. Open backend docs at `<backend-url>/docs`.

## Notes

- This repo is local-only by design.
- No `.env` setup is required.
- Backend CORS is open for local browser development.
