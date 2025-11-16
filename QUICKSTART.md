# Quick Start Guide - Dementia Detection Web App

Get up and running in 5 minutes! 🚀

## Prerequisites

- Python 3.11+ installed
- Node.js 18+ installed
- (Optional) Docker installed

## Option 1: Local Development (Recommended for Hackathon)

### Step 1: Backend Setup

```bash
# Navigate to project root
cd emoryhacks

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
python -m uvicorn api.main:app --reload
```

Backend will run at `http://localhost:8000`

### Step 2: Frontend Setup (New Terminal)

```bash
# Navigate to webapp directory
cd webapp

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will run at `http://localhost:3000`

### Step 3: Test It!

1. Open `http://localhost:3000` in your browser
2. Record audio or upload an audio file
3. Click "Analyze" to get predictions

## Option 2: Docker (Fastest)

```bash
# Build and start everything
docker-compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

## Option 3: Using Startup Scripts

### Windows:
```bash
# Terminal 1 - Backend
start_api.bat

# Terminal 2 - Frontend
start_frontend.bat
```

### Mac/Linux:
```bash
# Terminal 1 - Backend
chmod +x start_api.sh
./start_api.sh

# Terminal 2 - Frontend
chmod +x start_frontend.sh
./start_frontend.sh
```

## Testing the API

Test the API directly:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/your/audio.wav"
```

## Training Models (Optional)

If you want to train your own models:

```bash
cd emoryhacks
python src/run_training.py
```

Place trained models in `emoryhacks/models/` directory.

## Troubleshooting

### Backend won't start
- Check Python version: `python --version` (need 3.11+)
- Make sure virtual environment is activated
- Check if port 8000 is already in use

### Frontend won't start
- Check Node version: `node --version` (need 18+)
- Delete `node_modules` and run `npm install` again
- Check if port 3000 is already in use

### API connection errors
- Make sure backend is running on port 8000
- Check `webapp/.env` file has correct `VITE_API_URL`

### Model not found warnings
- This is normal if you haven't trained models yet
- The API will still work with dummy predictions for demo purposes
- Train models using the training scripts to get real predictions

## Next Steps

1. **Train Models**: Use your training data to create ML models
2. **Deploy to AWS**: See `DEPLOYMENT.md` for cloud deployment
3. **Customize UI**: Edit files in `webapp/src/` to customize the interface
4. **Add Features**: Extend the API in `emoryhacks/api/main.py`

## Need Help?

- Check `DEPLOYMENT.md` for detailed deployment instructions
- Check `README_DEPLOYMENT.md` for architecture overview
- Review API docs at `http://localhost:8000/docs` (when backend is running)

## Important Notes

⚠️ **Research Use Only**: This tool is for research purposes and is NOT a medical device.

🔒 **Privacy**: Audio files are processed in memory and not stored permanently.

📊 **Models**: You need trained models in `emoryhacks/models/` for real predictions.


