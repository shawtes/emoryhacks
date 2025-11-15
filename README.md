# 🧠 Dementia Detection Web Application

A scalable, production-ready web application for dementia detection through speech/audio analysis, designed for healthcare professionals, hospitals, and clinics.

> ⚠️ **Research Use Only** - This tool is for research purposes and is NOT a medical device. Not intended for clinical diagnosis.

---

## 📐 Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  React/TypeScript Frontend (Port 3000)               │   │
│  │  • Audio Recording (Microphone)                      │   │
│  │  • File Upload (Drag & Drop)                          │   │
│  │  • Results Display                                    │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP/REST API
                        │ (CORS enabled)
┌───────────────────────▼─────────────────────────────────────┐
│                      API LAYER                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  FastAPI Backend (Port 8000)                         │   │
│  │  • POST /predict - Audio analysis endpoint           │   │
│  │  • GET /health - Health check                        │   │
│  │  • GET / - API info                                  │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                   PROCESSING LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Preprocessing│→ │  Feature     │→ │   ML Model   │      │
│  │ • Denoising  │  │  Extraction  │  │  Inference   │      │
│  │ • Normalize  │  │ • MFCC       │  │ • Ensemble   │      │
│  │ • Resample   │  │ • GTCC       │  │ • RandomForest│     │
│  └──────────────┘  │ • Formants   │  └──────────────┘      │
│                    │ • F0, etc.   │                        │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Audio Input (WAV/MP3/WebM)
    ↓
[FastAPI receives file]
    ↓
[Preprocessing Pipeline]
    ├─→ Load audio (soundfile)
    ├─→ Spectral denoising (noisereduce)
    └─→ Peak normalization
    ↓
[Feature Extraction]
    ├─→ Frame-level features (MFCC, GTCC, Formants, F0)
    ├─→ High-level features (pause stats, speaking rate)
    └─→ Feature aggregation (mean, std, etc.)
    ↓
[ML Model Inference]
    ├─→ Load trained model (joblib)
    ├─→ Predict probability
    └─→ Calculate confidence
    ↓
[Response]
    └─→ JSON: {prediction, probability, confidence, message}
```

### Component Architecture

#### Frontend (React/TypeScript)
- **Entry Point**: `webapp/src/main.tsx`
- **Main App**: `webapp/src/App.tsx` - Orchestrates components
- **Components**:
  - `AudioRecorder` - Browser microphone recording
  - `FileUploader` - Drag & drop file upload
  - `ResultsDisplay` - Prediction results visualization
- **State Management**: React hooks (useState)
- **API Communication**: Fetch API

#### Backend (FastAPI/Python)
- **API Server**: `emoryhacks/api/main.py`
- **Preprocessing**: `emoryhacks/src/preprocess.py`
- **Feature Extraction**: `emoryhacks/src/features.py`
- **ML Models**: `emoryhacks/src/ml_train.py`, `ensemble_train.py`
- **Model Storage**: `emoryhacks/models/` (trained models)

---

## 📁 Project Structure

```
shawtestclone/
│
├── 📂 emoryhacks/                    # Backend (Python/FastAPI)
│   ├── 📂 api/                      # FastAPI application
│   │   ├── __init__.py
│   │   └── main.py                  # Main API server, endpoints
│   │
│   ├── 📂 src/                      # ML pipeline & processing
│   │   ├── __init__.py
│   │   ├── preprocess.py            # Audio preprocessing (denoise, normalize)
│   │   ├── features.py              # Feature extraction (MFCC, GTCC, etc.)
│   │   ├── features_agg.py          # Feature aggregation for ML
│   │   ├── ml_train.py              # RandomForest training
│   │   ├── ensemble_train.py        # Ensemble model training
│   │   ├── build_dataset.py         # Dataset building utilities
│   │   ├── data_ingest.py           # Data ingestion helpers
│   │   ├── generate_splits.py       # Cross-validation splits
│   │   └── run_training.py           # Training orchestration
│   │
│   ├── 📂 data/                     # Data directories
│   │   ├── raw/                     # Original audio files
│   │   ├── interim/                 # Preprocessed audio
│   │   └── processed/               # Extracted features
│   │
│   ├── 📂 models/                   # Trained ML models (add your models here)
│   │   └── (trained .joblib files)
│   │
│   ├── 📂 reports/                  # Training reports & metrics
│   │   └── metrics/                 # Cross-validation results
│   │
│   ├── requirements.txt             # Python dependencies
│   ├── README.md                    # Backend documentation
│   └── PLAN.md                      # Project plan & milestones
│
├── 📂 webapp/                       # Frontend (React/TypeScript)
│   ├── 📂 src/
│   │   ├── 📂 components/           # React components
│   │   │   ├── AudioRecorder.tsx    # Microphone recording component
│   │   │   ├── AudioRecorder.css
│   │   │   ├── FileUploader.tsx      # File upload component
│   │   │   ├── FileUploader.css
│   │   │   ├── ResultsDisplay.tsx    # Results visualization
│   │   │   └── ResultsDisplay.css
│   │   │
│   │   ├── App.tsx                  # Main application component
│   │   ├── App.css                  # Main app styles
│   │   ├── main.tsx                 # React entry point
│   │   ├── index.css                # Global styles
│   │   └── types.ts                 # TypeScript type definitions
│   │
│   ├── index.html                   # HTML entry point
│   ├── package.json                 # Node.js dependencies
│   ├── tsconfig.json                # TypeScript configuration
│   ├── vite.config.ts               # Vite build configuration
│   ├── Dockerfile                   # Frontend container
│   ├── nginx.conf                   # Nginx config for production
│   └── README.md                    # Frontend documentation
│
├── 📂 .ebextensions/                # AWS Elastic Beanstalk config
│   └── python.config                # EB Python configuration
│
├── 🐳 Docker Configuration
│   ├── Dockerfile                   # Backend container image
│   ├── docker-compose.yml           # Full stack orchestration
│   └── .dockerignore                # Docker ignore patterns
│
├── ☁️ AWS Deployment Files
│   ├── application.py               # EB entry point
│   ├── Procfile                     # Process file for EB/Heroku
│   └── ecs-task-definition.json     # ECS/Fargate task definition
│
├── 🚀 Startup Scripts
│   ├── start_api.sh                 # Backend startup (Linux/Mac)
│   ├── start_api.bat                # Backend startup (Windows)
│   ├── start_frontend.sh            # Frontend startup (Linux/Mac)
│   └── start_frontend.bat           # Frontend startup (Windows)
│
├── 📚 Documentation
│   ├── README.md                    # This file (main documentation)
│   ├── QUICKSTART.md                # Quick start guide
│   ├── README_DEPLOYMENT.md         # Deployment overview
│   └── DEPLOYMENT.md                # Detailed AWS deployment guide
│
└── 📝 Configuration Files
    ├── .gitignore                   # Git ignore patterns
    └── (venv/)                      # Python virtual environment (gitignored)
```

---

## 🏗️ Technology Stack

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **ML Libraries**: scikit-learn, joblib
- **Audio Processing**: librosa, soundfile, noisereduce, webrtcvad
- **Server**: Uvicorn (ASGI)

### Frontend
- **Framework**: React 18
- **Language**: TypeScript
- **Build Tool**: Vite
- **Styling**: CSS3 (no frameworks - lightweight)

### Deployment
- **Containerization**: Docker, Docker Compose
- **Cloud Platforms**: AWS (Elastic Beanstalk, ECS, Lambda)
- **Web Server**: Nginx (frontend production)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- (Optional) Docker

### Option 1: Local Development

**Backend:**
```bash
cd emoryhacks
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
python -m uvicorn api.main:app --reload
```

**Frontend (new terminal):**
```bash
cd webapp
npm install
npm run dev
```

Visit `http://localhost:3000`

### Option 2: Docker
```bash
docker-compose up --build
```

### Option 3: Startup Scripts
```bash
# Windows
start_api.bat        # Terminal 1
start_frontend.bat    # Terminal 2

# Mac/Linux
./start_api.sh        # Terminal 1
./start_frontend.sh   # Terminal 2
```

---

## 🔌 API Endpoints

### `POST /predict`
Upload audio file for analysis.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (audio file: WAV, MP3, WebM, etc.)

**Response:**
```json
{
  "prediction": "dementia" | "no_dementia",
  "probability": 0.75,
  "confidence": "high" | "medium" | "low",
  "message": "Prediction: Dementia. Probability: 75.0%..."
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### `GET /`
API information.

**Response:**
```json
{
  "status": "ok",
  "message": "Dementia Detection API - Research Use Only",
  "model_loaded": true
}
```

---

## 🧪 Testing

### Test API with cURL
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/audio.wav"
```

### Test Frontend
1. Open `http://localhost:3000`
2. Record audio or upload file
3. Click "Analyze" to see predictions

---

## 📦 Deployment

### AWS Elastic Beanstalk (Recommended for Hackathon)
```bash
pip install awsebcli
eb init -p python-3.11 dementia-detection-api
eb create dementia-detection-env
eb deploy
```

### Docker Production
```bash
# Build
docker build -t dementia-api .
docker build -t dementia-frontend ./webapp

# Run
docker run -p 8000:8000 dementia-api
docker run -p 3000:80 dementia-frontend
```

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed deployment instructions.

---

## 🔧 Configuration

### Environment Variables

**Backend:**
- `PYTHONUNBUFFERED=1` - Python logging
- `MODEL_PATH` - Optional: custom model path

**Frontend:**
- `VITE_API_URL` - Backend API URL (default: `http://localhost:8000`)

### Model Setup
1. Train models using `emoryhacks/src/run_training.py`
2. Place trained `.joblib` files in `emoryhacks/models/`
3. API auto-discovers models on startup

---

## 📊 Key Features

✅ **Audio Input**
- Browser microphone recording
- File upload (drag & drop)
- Multiple audio formats supported

✅ **ML Pipeline**
- Preprocessing (denoising, normalization)
- Feature extraction (62-dimensional feature vectors)
- Ensemble model inference

✅ **Results Display**
- Prediction (dementia/no_dementia)
- Probability score
- Confidence level
- User-friendly visualization

✅ **Scalability**
- Docker containerization
- AWS-ready deployment
- Stateless API design
- Horizontal scaling support

---

## ⚠️ Important Notes

- **Research Use Only**: Not a medical device
- **Model Required**: Train models before production use
- **Privacy**: Audio processed in memory, not stored
- **HIPAA**: Ensure compliance for production healthcare use

---

## 🐛 Troubleshooting

### Backend Issues
- **Port 8000 in use**: Change port with `--port 8001`
- **Model not found**: Place models in `emoryhacks/models/`
- **Audio errors**: Check file format (WAV/MP3 supported)

### Frontend Issues
- **API connection**: Check `VITE_API_URL` environment variable
- **CORS errors**: Verify backend CORS configuration
- **Build errors**: Delete `node_modules` and reinstall

---

## 📚 Additional Documentation

- [QUICKSTART.md](./QUICKSTART.md) - 5-minute setup guide
- [DEPLOYMENT.md](./DEPLOYMENT.md) - Detailed AWS deployment
- [README_DEPLOYMENT.md](./README_DEPLOYMENT.md) - Deployment overview
- [webapp/README.md](./webapp/README.md) - Frontend-specific docs

---

## 🤝 Contributing

This is a hackathon project. For production use:
1. Train models with your dataset
2. Add authentication/authorization
3. Implement HIPAA compliance measures
4. Add comprehensive error handling
5. Set up monitoring and logging

---

## 📝 License

Research use only - See project license file.

---

## 🔗 Useful Links

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [React Docs](https://react.dev/)
- [AWS Elastic Beanstalk](https://aws.amazon.com/elasticbeanstalk/)
- [Docker Docs](https://docs.docker.com/)


