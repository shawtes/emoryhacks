# 🚀 Quick Start Guide

## ⚠️ Important: Directory Structure

This project has **separate frontend and backend** directories:

```
shawtestclone/
├── emoryhacks/     ← Backend (Python/FastAPI)
└── webapp/         ← Frontend (React/TypeScript) ← YOU ARE HERE
```

## 🎯 Quick Commands

### Start Backend (Terminal 1)

```powershell
# Navigate to backend directory
cd emoryhacks

# Activate virtual environment
..\venv\Scripts\Activate.ps1

# Install dependencies (first time only)
pip install -r requirements.txt

# Start API server
python -m uvicorn api.main:app --reload
```

Backend runs at: `http://localhost:8000`

### Start Frontend (Terminal 2)

```powershell
# Navigate to frontend directory
cd webapp

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

Frontend runs at: `http://localhost:3000`

## 🐳 Or Use Docker (Easiest)

From the **root directory** (`shawtestclone`):

```powershell
docker-compose up --build
```

This starts both backend and frontend automatically!

## 📝 Common Mistakes

### ❌ Wrong: Running npm from root
```powershell
# This won't work - package.json is in webapp/
npm run dev
```

### ✅ Correct: Navigate to webapp first
```powershell
cd webapp
npm run dev
```

## 🛠️ Using Startup Scripts

### Windows PowerShell

**Backend:**
```powershell
.\start_api.bat
```

**Frontend:**
```powershell
.\start_frontend.bat
```

These scripts automatically navigate to the correct directories!

## 📚 More Help

- See [QUICKSTART.md](./QUICKSTART.md) for detailed setup
- See [README.md](./README.md) for full documentation
- See [DEPLOYMENT.md](./DEPLOYMENT.md) for AWS deployment


