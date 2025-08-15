# Flood Detection Web App

A simple web application for flood detection using AI/ML models.

## Features
- Upload images to detect flood conditions
- Real-time AI predictions
- Risk level assessment
- Interactive web interface

## Quick Start

### Local Development
```bash
pip install -r requirements.txt
python app.py
```
Visit `http://localhost:5000`

### Deploy to Vercel

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Deploy:
```bash
vercel --prod
```

## Files Structure
```
web app/
├── app.py              # Flask backend
├── model.py            # ML model logic
├── requirements.txt    # Dependencies
├── vercel.json        # Vercel config
└── templates/
    └── index.html     # Frontend
```

## API Endpoints
- `GET /` - Main web interface
- `POST /api/predict` - Image prediction
- `GET /api/health` - Health check
- `GET /api/info` - Model information

## Usage
1. Upload satellite/aerial imagery
2. Click "Analyze Image" 
3. View flood detection results
4. Get risk assessment and recommendations

## Model
- CNN-based flood detection
- Classes: flooded, normal
- Input: 224x224 RGB images
- Output: prediction with confidence score
