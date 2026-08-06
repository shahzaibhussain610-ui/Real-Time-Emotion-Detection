# Deploy to Streamlit Cloud - Step by Step Guide

## Your App is Ready to Deploy!

Your code is already on GitHub at:
**https://github.com/shahzaibhussain610-ui/Real-Time-Emotion-Detection**

## Deployment Steps:

### 1. Visit Streamlit Cloud
Go to: **https://share.streamlit.io**

### 2. Sign In
- Click "Sign in" button
- Choose "Sign in with GitHub"
- Authorize Streamlit to access your GitHub account

### 3. Create New App
- Click "New app" button
- Fill in the details:

**Repository:**
```
shahzaibhussain610-ui/Real-Time-Emotion-Detection
```

**Branch:**
```
master
```

**Main file path:**
```
streamlit_app.py
```

### 4. Advanced Settings (Important!)
Click "Advanced settings" and add:

**Python version:**
```
3.11
```

**Requirements file:**
```
requirements.txt
```

### 5. Deploy
- Click "Deploy!"
- Wait 2-3 minutes for deployment
- Your app will be live at: `https://[your-app-name].streamlit.app`

## Important Notes:

### Model File Size Warning
Your model file (`models/emotion_dnn_model.keras`) is 68.79 MB, which exceeds GitHub's 50 MB limit. 

**Solutions:**
1. **Use Git LFS** (Recommended):
   ```bash
   git lfs install
   git lfs track "*.keras"
   git add .gitattributes
   git commit -m "Add Git LFS for model files"
   git push origin master
   ```

2. **Use External Storage**:
   - Upload model to Google Drive, Dropbox, or AWS S3
   - Modify app to download model from URL
   - This is better for Streamlit Cloud

### Alternative: Use Flask Version
If Streamlit Cloud has issues with the large model, use the Flask version:
- Deploy on: https://www.pythonanywhere.com
- Or: https://render.com
- Or: https://railway.app

## Testing Locally First:

Before deploying, test locally:
```bash
streamlit run streamlit_app.py
```

## Need Help?

If deployment fails:
1. Check Streamlit Cloud logs
2. Ensure model file is accessible
3. Verify all dependencies in requirements.txt
4. Check Python version compatibility

## Your Live App URL:
After deployment, share your app at:
```
https://[your-app-name].streamlit.app