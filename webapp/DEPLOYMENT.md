# Deployment & Scalability Guide

Complete guide for deploying the frontend to production environments with Docker, AWS, and scalability considerations.

## Table of Contents

1. [Docker Deployment](#docker-deployment)
2. [AWS Deployment Options](#aws-deployment-options)
3. [Scalability Strategies](#scalability-strategies)
4. [Performance Optimization](#performance-optimization)
5. [Monitoring & Logging](#monitoring--logging)
6. [Security](#security)
7. [CI/CD](#cicd)

## Docker Deployment

### Local Docker Build

```bash
# Build Docker image
docker build -t dementia-detection-frontend .

# Run container
docker run -p 3000:80 dementia-detection-frontend
```

### Docker Compose

The project includes `docker-compose.yml` at the root for full-stack deployment:

```bash
# Start both frontend and backend
docker-compose up

# Build and start
docker-compose up --build

# Run in background
docker-compose up -d
```

### Dockerfile Structure

Multi-stage build for optimal image size:

```dockerfile
# Stage 1: Build
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: Production
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**Benefits**:
- Smaller final image (~25MB vs ~200MB)
- No build tools in production
- Optimized for production

### Nginx Configuration

The `nginx.conf` includes:

- **SPA Routing**: All routes serve `index.html`
- **API Proxy**: `/api` proxied to backend
- **Gzip Compression**: Enabled for text assets
- **Static Asset Caching**: 1 year cache for immutable assets
- **Security Headers**: Can be added

## AWS Deployment Options

### Option 1: AWS Amplify (Recommended for Frontend)

**Best for**: Automatic deployments, CI/CD integration, simple setup

#### Setup Steps

1. **Connect Repository**
   - Go to AWS Amplify Console
   - Connect GitHub/GitLab/Bitbucket
   - Select repository and branch

2. **Build Settings**
   ```yaml
   version: 1
   frontend:
     phases:
       preBuild:
         commands:
           - cd webapp
           - npm install
       build:
         commands:
           - npm run build
     artifacts:
       baseDirectory: webapp/dist
       files:
         - '**/*'
     cache:
       paths:
         - webapp/node_modules/**/*
   ```

3. **Environment Variables**
   ```
   VITE_API_URL=https://your-api-url.com
   ```

4. **Deploy**
   - Automatic on git push
   - Manual deploy available
   - Preview deployments for PRs

**Advantages**:
- ✅ Automatic HTTPS/SSL
- ✅ Global CDN (CloudFront)
- ✅ Automatic deployments
- ✅ Preview environments
- ✅ Free tier available

**Cost**: ~$0.15/GB data transfer (first 15GB free/month)

### Option 2: S3 + CloudFront

**Best for**: Full control, custom domains, existing S3 infrastructure

#### Setup Steps

1. **Build Frontend**
   ```bash
   cd webapp
   npm install
   npm run build
   ```

2. **Create S3 Bucket**
   ```bash
   aws s3 mb s3://dementia-detection-frontend
   aws s3 website s3://dementia-detection-frontend --index-document index.html --error-document index.html
   ```

3. **Upload Files**
   ```bash
   aws s3 sync dist/ s3://dementia-detection-frontend/ --delete
   ```

4. **Configure CloudFront**
   - Create distribution
   - Origin: S3 bucket
   - Default root object: `index.html`
   - Error pages: 404 → `/index.html` (for SPA routing)
   - SSL certificate: ACM certificate
   - Custom domain: Optional

5. **Set Bucket Policy**
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Sid": "PublicReadGetObject",
         "Effect": "Allow",
         "Principal": "*",
         "Action": "s3:GetObject",
         "Resource": "arn:aws:s3:::dementia-detection-frontend/*"
       }
     ]
   }
   ```

**Advantages**:
- ✅ Full control over configuration
- ✅ Custom domains
- ✅ Cost-effective for high traffic
- ✅ Integration with other AWS services

**Cost**: ~$0.023/GB storage + $0.085/GB transfer (first 1GB free)

### Option 3: ECS/Fargate (Containerized)

**Best for**: Container orchestration, microservices, existing ECS infrastructure

#### Setup Steps

1. **Build and Push to ECR**
   ```bash
   # Login to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

   # Build image
   docker build -t dementia-detection-frontend ./webapp

   # Tag image
   docker tag dementia-detection-frontend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/dementia-detection-frontend:latest

   # Push image
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/dementia-detection-frontend:latest
   ```

2. **Create ECS Task Definition**
   ```json
   {
     "family": "dementia-detection-frontend",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "256",
     "memory": "512",
     "containerDefinitions": [
       {
         "name": "frontend",
         "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/dementia-detection-frontend:latest",
         "portMappings": [
           {
             "containerPort": 80,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "VITE_API_URL",
             "value": "https://api.example.com"
           }
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/dementia-detection-frontend",
             "awslogs-region": "us-east-1",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }
   ```

3. **Create ECS Service**
   - Use Application Load Balancer
   - Configure health checks
   - Set auto-scaling policies

**Advantages**:
- ✅ Container orchestration
- ✅ Auto-scaling
- ✅ Service discovery
- ✅ Integration with ALB

**Cost**: ~$0.04/vCPU-hour + $0.004/GB-RAM-hour (Fargate)

### Option 4: Elastic Beanstalk

**Best for**: Simple container deployment, managed platform

#### Setup Steps

1. **Install EB CLI**
   ```bash
   pip install awsebcli
   ```

2. **Initialize**
   ```bash
   cd webapp
   eb init -p docker dementia-detection-frontend --region us-east-1
   ```

3. **Create Environment**
   ```bash
   eb create dementia-detection-frontend-env
   ```

4. **Deploy**
   ```bash
   eb deploy
   ```

**Advantages**:
- ✅ Simple deployment
- ✅ Managed platform
- ✅ Auto-scaling
- ✅ Health monitoring

## Scalability Strategies

### Horizontal Scaling

#### Frontend Scaling

**Static Assets** (S3 + CloudFront):
- Automatically scales to unlimited requests
- Global CDN reduces latency
- No server management needed

**Containerized** (ECS/Fargate):
- Auto-scaling based on CPU/memory
- Target: 70% CPU utilization
- Min: 1 task, Max: 10 tasks (adjust as needed)

**Load Balancing**:
- Application Load Balancer (ALB)
- Health checks: `GET /` (200 OK)
- Sticky sessions: Not needed (stateless)

### Vertical Scaling

For containerized deployments:
- Increase CPU: 256 → 512 → 1024
- Increase Memory: 512MB → 1024MB → 2048MB
- Monitor CloudWatch metrics

### Caching Strategy

#### Static Assets
- **CloudFront**: Cache static assets (JS, CSS, images)
- **Cache-Control**: `public, max-age=31536000, immutable`
- **Cache Invalidation**: On deployments

#### API Responses
- **Browser Cache**: Cache API responses where appropriate
- **CDN Cache**: Cache public API responses (if applicable)

### CDN Configuration

**CloudFront Settings**:
- **TTL**: 1 year for static assets
- **Compression**: Gzip/Brotli enabled
- **HTTP/2**: Enabled
- **Origin Shield**: Enable for cost reduction

### Database Considerations

Frontend doesn't directly connect to database, but consider:
- API response caching
- Optimistic UI updates
- Request deduplication

## Performance Optimization

### Build Optimizations

#### Code Splitting
```typescript
// Lazy load routes
const Assessment = lazy(() => import('./pages/Assessment'))
const Home = lazy(() => import('./pages/Home'))
```

#### Tree Shaking
- Vite automatically removes unused code
- Use ES modules for better tree-shaking

#### Asset Optimization
- Images: Use WebP format
- Icons: Use SVG sprites
- Fonts: Subset fonts, use `font-display: swap`

### Runtime Optimizations

#### React Optimizations
- `React.memo()` for expensive components
- `useMemo()` for expensive calculations
- `useCallback()` for event handlers

#### Bundle Size
- Monitor bundle size: `npm run build -- --analyze`
- Code splitting by route
- Dynamic imports for large libraries

### Network Optimizations

#### Compression
- Gzip/Brotli compression (Nginx/CloudFront)
- Minify CSS/JS
- Optimize images

#### HTTP/2
- Enable HTTP/2 on ALB/CloudFront
- Server push (if needed)

## Monitoring & Logging

### CloudWatch Metrics

**Key Metrics to Monitor**:
- Request count
- Error rate (4xx, 5xx)
- Response time
- Cache hit ratio (CloudFront)

### CloudWatch Logs

**Log Groups**:
- `/aws/amplify/dementia-detection` (Amplify)
- `/ecs/dementia-detection-frontend` (ECS)
- `/aws/cloudfront/distribution` (CloudFront)

### Application Monitoring

**Recommended Tools**:
- **AWS X-Ray**: Distributed tracing
- **CloudWatch Synthetics**: Uptime monitoring
- **CloudWatch Alarms**: Alert on errors

### Error Tracking

Consider integrating:
- **Sentry**: Error tracking and monitoring
- **LogRocket**: Session replay
- **New Relic**: APM (Application Performance Monitoring)

## Security

### HTTPS/SSL

- **AWS Certificate Manager (ACM)**: Free SSL certificates
- **Force HTTPS**: Redirect HTTP → HTTPS
- **HSTS**: Enable HTTP Strict Transport Security

### Security Headers

Add to Nginx/CloudFront:
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Content-Security-Policy: default-src 'self'
Strict-Transport-Security: max-age=31536000
```

### Environment Variables

**Never commit**:
- API keys
- Secrets
- Production URLs

**Use**:
- AWS Secrets Manager
- AWS Systems Manager Parameter Store
- Environment variables in deployment platform

### CORS Configuration

Configure in backend API:
```python
# Allow only production domain
origins = [
    "https://dementia-detection.example.com",
    "https://www.dementia-detection.example.com"
]
```

## CI/CD

### GitHub Actions Example

```yaml
name: Deploy Frontend

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      
      - name: Install dependencies
        run: |
          cd webapp
          npm ci
      
      - name: Build
        run: |
          cd webapp
          npm run build
        env:
          VITE_API_URL: ${{ secrets.VITE_API_URL }}
      
      - name: Deploy to S3
        run: |
          aws s3 sync webapp/dist/ s3://dementia-detection-frontend/ --delete
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      
      - name: Invalidate CloudFront
        run: |
          aws cloudfront create-invalidation --distribution-id ${{ secrets.CLOUDFRONT_DISTRIBUTION_ID }} --paths "/*"
```

### GitLab CI Example

```yaml
deploy:
  stage: deploy
  image: node:18
  script:
    - cd webapp
    - npm ci
    - npm run build
    - aws s3 sync dist/ s3://dementia-detection-frontend/ --delete
    - aws cloudfront create-invalidation --distribution-id $CLOUDFRONT_DISTRIBUTION_ID --paths "/*"
  only:
    - main
```

## Cost Optimization

### AWS Cost Breakdown

**Amplify**:
- Free tier: 15GB data transfer/month
- After: $0.15/GB

**S3 + CloudFront**:
- S3: $0.023/GB storage (first 5GB free)
- CloudFront: $0.085/GB transfer (first 1TB free)
- Requests: $0.0075/10,000 requests

**ECS Fargate**:
- vCPU: $0.04/hour
- Memory: $0.004/GB-hour
- Example: 1 task (0.25 vCPU, 512MB) = ~$8/month

### Optimization Tips

1. **Use CloudFront caching** to reduce origin requests
2. **Enable compression** to reduce bandwidth
3. **Use S3 lifecycle policies** for old assets
4. **Monitor unused resources** and delete them
5. **Use reserved capacity** for predictable workloads

## Disaster Recovery

### Backup Strategy

- **Code**: Git repository (GitHub/GitLab)
- **Build artifacts**: S3 versioning enabled
- **Configuration**: Infrastructure as Code (Terraform/CloudFormation)

### Recovery Procedures

1. **Redeploy from Git**: Latest code always available
2. **Restore from S3**: Previous build artifacts
3. **Rollback**: Deploy previous version
4. **Failover**: Multi-region deployment (if needed)

## Troubleshooting

### Common Issues

1. **404 Errors on Refresh**
   - Fix: Configure SPA routing in Nginx/CloudFront
   - Solution: All routes → `index.html`

2. **API Connection Errors**
   - Check: `VITE_API_URL` environment variable
   - Verify: CORS configuration on backend
   - Test: API endpoint accessibility

3. **Build Failures**
   - Check: Node.js version compatibility
   - Verify: All dependencies installed
   - Review: Build logs for errors

4. **Slow Load Times**
   - Enable: CloudFront CDN
   - Optimize: Bundle size
   - Enable: Compression

### Debugging

```bash
# Check CloudFront logs
aws cloudfront get-distribution --id <distribution-id>

# Check S3 bucket
aws s3 ls s3://dementia-detection-frontend/

# Check ECS logs
aws logs tail /ecs/dementia-detection-frontend --follow

# Test locally with production build
npm run build
npm run preview
```

## Production Checklist

- [ ] Environment variables configured
- [ ] HTTPS/SSL certificates set up
- [ ] CORS configured for production domain
- [ ] CDN configured (CloudFront)
- [ ] Monitoring and alerts set up
- [ ] Error tracking configured
- [ ] Security headers added
- [ ] Backup strategy in place
- [ ] CI/CD pipeline configured
- [ ] Documentation updated with production URLs
- [ ] Load testing completed
- [ ] Disaster recovery plan documented

## Resources

- [AWS Amplify Documentation](https://docs.aws.amazon.com/amplify/)
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [AWS CloudFront Documentation](https://docs.aws.amazon.com/cloudfront/)
- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [Docker Documentation](https://docs.docker.com/)
- [Nginx Documentation](https://nginx.org/en/docs/)


