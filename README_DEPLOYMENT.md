# Dementia Detection Web App - Quick Start

A scalable web application for dementia detection through speech/audio analysis, built for healthcare professionals.

## 🏗️ Architecture

- **Backend**: FastAPI (Python) - ML inference API
- **Frontend**: React + TypeScript - Modern web interface
- **Deployment**: AWS-ready (Elastic Beanstalk, ECS, or Lambda)

## 🚀 Quick Start (Local Development)

### Backend Setup

1. Navigate to backend directory:
```bash
cd emoryhacks
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the API server:
```bash
python -m uvicorn api.main:app --reload
```

API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd webapp
```

2. Install dependencies:
```bash
npm install
```

3. Start development server:
```bash
npm run dev
```

Frontend will be available at `http://localhost:3000`

## 🐳 Docker Deployment

### Using Docker Compose (Full Stack)

```bash
docker-compose up --build
```

This starts both backend (port 8000) and frontend (port 3000).

### Individual Services

**Backend only:**
```bash
docker build -t dementia-api .
docker run -p 8000:8000 dementia-api
```

**Frontend only:**
```bash
cd webapp
docker build -t dementia-frontend .
docker run -p 3000:80 dementia-frontend
```

## ☁️ AWS Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed AWS deployment instructions.

### Quick AWS Elastic Beanstalk Deploy

```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.11 dementia-detection-api

# Create and deploy
eb create dementia-detection-env
eb deploy
```

## 📋 Prerequisites

- Python 3.11+
- Node.js 18+
- Trained ML models (place in `emoryhacks/models/`)
- AWS account (for cloud deployment)

## 🔧 Configuration

### Backend

- Model path: Set `MODEL_PATH` environment variable or place models in `emoryhacks/models/`
- API port: Default 8000 (configurable via `--port`)

### Frontend

- API URL: Set `VITE_API_URL` environment variable (default: `http://localhost:8000`)

## 📁 Project Structure

```
.
├── emoryhacks/          # Backend (Python/FastAPI)
│   ├── api/            # FastAPI application
│   ├── src/            # ML pipeline code
│   ├── models/         # Trained models (add your models here)
│   └── requirements.txt
├── webapp/             # Frontend (React/TypeScript)
│   ├── src/
│   │   ├── components/
│   │   └── App.tsx
│   └── package.json
├── Dockerfile          # Backend container
├── docker-compose.yml  # Full stack deployment
└── DEPLOYMENT.md       # Detailed deployment guide
```

## 🧪 Testing

### Test API Endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_audio.wav"
```

### Expected Response

```json
{
  "prediction": "no_dementia",
  "probability": 0.75,
  "confidence": "high",
  "message": "Prediction: No Dementia. Probability: 75.0%. This is a research tool only..."
}
```

## ⚠️ Important Notes

- **Research Use Only**: This tool is for research purposes and is NOT a medical device
- **Model Required**: You need to train and place models in `emoryhacks/models/` for real predictions
- **Privacy**: Audio files are processed in memory and not stored
- **HIPAA Compliance**: For production use, ensure HIPAA compliance measures are in place

## 🐛 Troubleshooting

### Backend Issues

- **Model not found**: Ensure models are in `emoryhacks/models/` directory
- **Audio processing errors**: Check audio file format (WAV, MP3 supported)
- **Memory errors**: Increase Docker/container memory limits

### Frontend Issues

- **API connection errors**: Check `VITE_API_URL` environment variable
- **CORS errors**: Ensure backend CORS is configured correctly

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [AWS Elastic Beanstalk Guide](https://docs.aws.amazon.com/elasticbeanstalk/)

## 📝 License

Research use only - See project license file.


