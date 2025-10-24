# 🌱 AgriVision AI - Hyperspectral Crop Disease Detection

An advanced AI-powered web application for detecting crop diseases using hyperspectral data analysis. Upload .npy, .npz, or .tiff files and get instant disease detection results with confidence scores and visualizations.

## 🚀 Live Demo

**[Access the Application](https://your-app-url.herokuapp.com)** *(Will be updated after deployment)*

## ✨ Features

- **Multi-format Support**: Upload .npy, .npz, .tiff, .tif hyperspectral files
- **AI-Powered Detection**: Advanced algorithms analyze spectral signatures
- **Interactive Visualizations**: Real-time charts and graphs
- **Confidence Scoring**: Get reliability metrics for each prediction
- **Responsive Design**: Works on desktop, tablet, and mobile
- **API Access**: RESTful API for programmatic access

## 🛠️ Quick Start

### Option 1: Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/ganeshjalkote932/agrivision-ai.git
   cd agrivision-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python simple_web_app.py
   ```

4. **Open your browser**
   ```
   http://localhost:5000
   ```

### Option 2: Deploy to Heroku

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/ganeshjalkote932/agrivision-ai)

1. Click the "Deploy to Heroku" button above
2. Create a Heroku account if needed
3. Choose an app name
4. Click "Deploy app"

### Option 3: Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template/your-template-id)

## 📊 How It Works

1. **Upload**: Select your hyperspectral data file (.npy, .npz, .tiff)
2. **Process**: The system automatically preprocesses and validates your data
3. **Analyze**: AI algorithms analyze spectral signatures for disease indicators
4. **Results**: Get detailed predictions with confidence scores and visualizations

## 🔧 API Usage

### Upload and Analyze

```bash
curl -X POST -F "file=@your_data.npy" http://localhost:5000/api/predict
```

### Response Format

```json
{
  "success": true,
  "predictions": [
    {
      "sample_index": 0,
      "prediction": 1,
      "prediction_label": "Diseased",
      "disease_probability": 0.85,
      "confidence": 0.92
    }
  ],
  "metadata": {
    "file_type": "npy",
    "original_shape": [100, 131],
    "file_size_mb": 0.52
  }
}
```

## 📁 Supported File Formats

| Format | Description | Example Use Case |
|--------|-------------|------------------|
| `.npy` | NumPy array files | Single hyperspectral dataset |
| `.npz` | Compressed NumPy archives | Multiple datasets with metadata |
| `.tiff/.tif` | TIFF image files | Hyperspectral image cubes |

## 🎯 Data Requirements

- **Spectral Range**: 400-2500 nm (recommended)
- **Bands**: 50-300 spectral bands
- **Format**: Reflectance values (0-1 range preferred)
- **Size**: Maximum 100MB per file

## 🏗️ Project Structure

```
agrivision-ai/
├── simple_web_app.py          # Main Flask application
├── templates/                 # HTML templates
│   ├── simple_index.html     # Home page
│   ├── simple_upload.html    # Upload interface
│   └── simple_results.html   # Results display
├── uploads/                   # Uploaded files (auto-created)
├── requirements.txt           # Python dependencies
├── Procfile                  # Heroku deployment config
├── runtime.txt               # Python version specification
└── README.md                 # This file
```

## 🔬 Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: Bootstrap 5, HTML5, JavaScript
- **AI/ML**: scikit-learn, NumPy
- **Visualization**: Matplotlib
- **File Processing**: PIL, tifffile
- **Deployment**: Heroku, Railway, or local

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/ganeshjalkote932/agrivision-ai/issues)
- **Documentation**: [Wiki](https://github.com/ganeshjalkote932/agrivision-ai/wiki)
- **Email**: your.email@example.com

## 🙏 Acknowledgments

- Built for agricultural technology advancement
- Supports precision farming initiatives
- Contributes to sustainable crop management

---

**AgriVision AI - Made with ❤️ for sustainable agriculture**