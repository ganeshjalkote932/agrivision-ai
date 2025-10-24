# 🚀 AgriVision AI Deployment Guide

## Quick Deployment Options

### 1. 🟣 Deploy to Heroku (Recommended)

**One-Click Deploy:**
[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/yourusername/agrivision-ai)

**Manual Deploy:**
```bash
# Install Heroku CLI
# Create Heroku app
heroku create your-agrivision-app

# Deploy
git push heroku main

# Open app
heroku open
```

### 2. 🚂 Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template/your-template-id)

1. Click "Deploy on Railway"
2. Connect your GitHub account
3. Select the repository
4. Deploy automatically

### 3. ⚡ Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

### 4. 🐳 Docker Deployment

```bash
# Build image
docker build -t agrivision-ai .

# Run container
docker run -p 5000:5000 agrivision-ai
```

### 5. 🌐 Deploy to Netlify (Static + Functions)

1. Connect GitHub repository to Netlify
2. Set build command: `pip install -r requirements.txt`
3. Set publish directory: `dist`
4. Deploy

## Environment Variables

Set these in your deployment platform:

```bash
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
MAX_CONTENT_LENGTH=104857600  # 100MB
```

## Custom Domain Setup

### Heroku
```bash
heroku domains:add agrivision.yourdomain.com
```

### Railway
1. Go to Railway dashboard
2. Select your project
3. Go to Settings > Domains
4. Add custom domain

## SSL Certificate

Most platforms provide free SSL certificates automatically:
- ✅ Heroku: Automatic with custom domains
- ✅ Railway: Automatic
- ✅ Vercel: Automatic
- ✅ Netlify: Automatic

## Performance Optimization

### For Production:
1. **Use Gunicorn** (add to requirements.txt):
   ```
   gunicorn==21.2.0
   ```

2. **Update Procfile**:
   ```
   web: gunicorn simple_web_app:app
   ```

3. **Enable Gzip Compression**
4. **Use CDN for static assets**

## Monitoring & Analytics

### Add to your deployment:
- **Error Tracking**: Sentry
- **Analytics**: Google Analytics
- **Uptime Monitoring**: UptimeRobot
- **Performance**: New Relic

## Scaling

### Heroku Scaling:
```bash
# Scale up
heroku ps:scale web=2

# Scale down
heroku ps:scale web=1
```

### Railway Scaling:
- Automatic scaling based on traffic
- Configure in Railway dashboard

## Backup Strategy

1. **Database**: Not applicable (stateless app)
2. **Uploaded Files**: Configure cloud storage
3. **Code**: GitHub repository backup

## Security Checklist

- ✅ HTTPS enabled
- ✅ Secret key configured
- ✅ File upload limits set
- ✅ Input validation implemented
- ✅ CORS configured if needed

## Troubleshooting

### Common Issues:

1. **Build Fails**:
   - Check Python version in runtime.txt
   - Verify requirements.txt dependencies

2. **App Crashes**:
   - Check logs: `heroku logs --tail`
   - Verify environment variables

3. **File Upload Issues**:
   - Check MAX_CONTENT_LENGTH setting
   - Verify file format support

### Getting Help:

- 📧 Email: support@agrivision.ai
- 💬 GitHub Issues: [Report Issue](https://github.com/yourusername/agrivision-ai/issues)
- 📖 Documentation: [Wiki](https://github.com/yourusername/agrivision-ai/wiki)

---

**AgriVision AI - Deployed with ❤️**