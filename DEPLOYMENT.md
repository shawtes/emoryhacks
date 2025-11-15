# Deployment Guide

This guide covers deploying the Dementia Detection web app to AWS.

## Architecture Overview

- **Backend**: FastAPI (Python) - handles audio processing and ML inference
- **Frontend**: React/TypeScript - user interface for audio upload/recording
- **Deployment Options**:
  1. **AWS Elastic Beanstalk** (Recommended for hackathon - simplest)
  2. **AWS ECS/Fargate** (More scalable, containerized)
  3. **AWS Lambda + API Gateway** (Serverless, cost-effective for low traffic)

## Prerequisites

1. AWS Account
2. AWS CLI installed and configured
3. Docker (for containerized deployment)
4. EB CLI (for Elastic Beanstalk deployment)

## Option 1: AWS Elastic Beanstalk (Simplest)

### Setup

1. Install EB CLI:
```bash
pip install awsebcli
```

2. Initialize Elastic Beanstalk:
```bash
eb init -p python-3.11 dementia-detection-api --region us-east-1
```

3. Create application version:
```bash
eb create dementia-detection-env
```

### Configuration

Create `.ebextensions/01_packages.config`:
```yaml
packages:
  yum:
    libsndfile: []
    ffmpeg: []
```

### Deploy

```bash
eb deploy
```

### Environment Variables

Set via EB console or CLI:
```bash
eb setenv PYTHONUNBUFFERED=1
```

## Option 2: Docker + AWS ECS/Fargate

### Build and Push Docker Image

1. Build Docker image:
```bash
docker build -t dementia-detection-api .
```

2. Tag for ECR:
```bash
docker tag dementia-detection-api:latest <account-id>.dkr.ecr.<region>.amazonaws.com/dementia-detection-api:latest
```

3. Push to ECR:
```bash
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/dementia-detection-api:latest
```

### Create ECS Task Definition

Use the provided `ecs-task-definition.json` or create via AWS Console:
- Task memory: 2GB (minimum for ML models)
- Task CPU: 1 vCPU
- Container port: 8000

### Deploy Frontend

1. Build frontend:
```bash
cd webapp
npm install
npm run build
```

2. Deploy to S3 + CloudFront:
```bash
aws s3 sync dist/ s3://your-bucket-name/
aws cloudfront create-invalidation --distribution-id <id> --paths "/*"
```

Or use AWS Amplify for automatic deployments.

## Option 3: AWS Lambda (Serverless)

### Package for Lambda

1. Create deployment package:
```bash
pip install -r emoryhacks/requirements.txt -t lambda-package/
cp -r emoryhacks lambda-package/
```

2. Create Lambda function with:
   - Runtime: Python 3.11
   - Handler: `api.main.handler`
   - Memory: 3008 MB (for ML models)
   - Timeout: 5 minutes

3. Use API Gateway to expose as REST API

## Environment Variables

Set these in your deployment environment:

- `PYTHONUNBUFFERED=1` - For proper logging
- `MODEL_PATH` - Optional: path to model file (defaults to auto-discovery)

## Model Deployment

1. Train your model using the training scripts
2. Upload model files to S3:
```bash
aws s3 cp emoryhacks/models/ s3://your-bucket/models/ --recursive
```

3. Download models on startup or mount as EFS volume

## Frontend Deployment

### Option A: AWS Amplify (Recommended)

1. Connect your Git repository
2. Build settings:
   - Build command: `cd webapp && npm install && npm run build`
   - Output directory: `webapp/dist`
3. Set environment variable: `VITE_API_URL=https://your-api-url.com`

### Option B: S3 + CloudFront

1. Build frontend:
```bash
cd webapp
npm install
npm run build
```

2. Upload to S3:
```bash
aws s3 sync dist/ s3://your-bucket-name/ --delete
```

3. Configure CloudFront distribution pointing to S3 bucket

## Health Checks

The API includes a health check endpoint:
- `GET /health` - Returns `{"status": "healthy"}`

Configure your load balancer/health checks to use this endpoint.

## Scaling Considerations

### Backend Scaling

- **Elastic Beanstalk**: Auto-scaling groups (min: 1, max: 10 instances)
- **ECS**: Service auto-scaling based on CPU/memory
- **Lambda**: Automatic scaling (up to 1000 concurrent executions)

### Cost Optimization

- Use reserved instances for predictable workloads
- Consider Spot instances for non-production
- Use CloudFront caching for frontend assets
- Enable S3 lifecycle policies for old model versions

## Monitoring

1. **CloudWatch Logs**: Application logs automatically collected
2. **CloudWatch Metrics**: CPU, memory, request count
3. **X-Ray**: For distributed tracing (optional)

## Security

1. **HTTPS**: Use AWS Certificate Manager (ACM) for SSL certificates
2. **CORS**: Update CORS settings in `api/main.py` for production
3. **IAM Roles**: Use IAM roles for AWS service access
4. **Secrets**: Use AWS Secrets Manager for sensitive data

## Troubleshooting

### Common Issues

1. **Model not loading**: Check model path and file permissions
2. **Memory errors**: Increase instance/task memory allocation
3. **Timeout errors**: Increase timeout settings
4. **CORS errors**: Update CORS configuration in API

### Logs

View logs:
- **EB**: `eb logs`
- **ECS**: CloudWatch Logs
- **Lambda**: CloudWatch Logs

## Quick Start (Local Testing)

1. Start backend:
```bash
cd emoryhacks
python -m uvicorn api.main:app --reload
```

2. Start frontend:
```bash
cd webapp
npm install
npm run dev
```

3. Test at http://localhost:3000

## Production Checklist

- [ ] Models trained and uploaded
- [ ] Environment variables configured
- [ ] HTTPS/SSL certificates set up
- [ ] CORS configured for production domain
- [ ] Health checks configured
- [ ] Monitoring and alerts set up
- [ ] Backup strategy for models
- [ ] Documentation updated with production URLs


