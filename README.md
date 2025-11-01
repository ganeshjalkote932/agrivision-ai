---
title: AgriVision AI - Multi-User Platform
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🌱 AgriVision AI - Multi-User Hyperspectral Crop Disease Detection Platform

An advanced AI-powered web application for detecting crop diseases using hyperspectral data analysis with multi-user support and role-based access control.

## 🚀 Features

- **Multi-User Support**: Farmer, Model Trainer, and Administrator roles
- **Role-Based Access**: Different permissions for different user types
- **File Format Support**: .npy, .npz, .tiff, .tif hyperspectral files
- **Real-Time Analysis**: Instant disease detection with confidence scores
- **Interactive Visualizations**: Spectral signatures and result charts
- **Analysis History**: Track and review past analyses
- **Secure Authentication**: Password-protected accounts with access codes

## 👥 User Roles

### 👨‍🌾 Farmer
- Upload and analyze hyperspectral crop data
- View disease detection results with confidence scores
- Access personal analysis history
- Download analysis reports
- **Access**: Open registration (no code required)

### 🔬 Model Trainer
- All Farmer features
- Advanced analytics and model insights
- Enhanced data visualization options
- **Access Code**: `AGRI2024MT`

### 👑 Administrator
- All features available
- User management capabilities
- System-wide statistics and monitoring
- Platform administration tools
- **Access Code**: `AGRI2024ADMIN`

## 🔐 Demo Accounts

**Administrator Account:**
- Username: `admin`
- Password: `AgriVision2024!`

## 📊 How to Use

1. **Register/Login**: Create an account or use demo credentials
2. **Upload Data**: Select your hyperspectral file (.npy, .npz, .tiff, .tif)
3. **Analyze**: Click "Analyze" to process your data
4. **View Results**: See disease detection results with visualizations
5. **Check History**: Review your past analyses in the History tab

## 🔧 Supported File Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `.npy` | NumPy array files | Single hyperspectral dataset |
| `.npz` | Compressed NumPy archives | Multiple datasets with metadata |
| `.tiff/.tif` | TIFF image files | Hyperspectral image cubes |

## 📈 Analysis Output

- **Disease Detection**: Healthy vs Diseased classification
- **Confidence Scores**: Reliability metrics for each prediction
- **Spectral Visualization**: Interactive plots of spectral signatures
- **Statistical Summary**: Overall analysis statistics
- **Sample-Level Results**: Individual predictions for each sample

## 🛡️ Security Features

- **Password Protection**: Secure user authentication
- **Role-Based Access**: Different permissions for different roles
- **Access Codes**: Special codes required for Model Trainer and Administrator roles
- **Session Management**: Secure user sessions

## 🌐 Technology Stack

- **Frontend**: Gradio (Python-based web interface)
- **Backend**: Python with NumPy, Pandas, Matplotlib
- **Database**: SQLite for user management and history
- **AI/ML**: scikit-learn for disease detection algorithms
- **Deployment**: Hugging Face Spaces

## 📝 Access Codes

- **Farmer**: No access code required
- **Model Trainer**: `AGRI2024MT`
- **Administrator**: `AGRI2024ADMIN`

## 🤝 Contributing

This project is open for contributions! Feel free to:
- Report issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built for advancing precision agriculture
- Supports sustainable farming practices
- Contributes to global food security initiatives

---

**AgriVision AI - Empowering farmers with AI-driven crop disease detection technology** 🌾