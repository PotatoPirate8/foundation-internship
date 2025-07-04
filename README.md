# 🫀 Vital Signs Detection with Computer Vision

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Advanced Computer Vision System for Medical Monitor Vital Signs Detection**  
> Combining OCR and Deep Learning to extract and predict vital signs from medical monitor displays

---

## 🎯 **Project Overview**

This project implements a sophisticated **hybrid OCR + CNN pipeline** to detect and extract vital signs from medical monitor images. The system can:

- 📸 **Extract vital signs** from medical monitor displays using advanced OCR
- 🧠 **Train CNN models** to predict vital signs when OCR fails
- 🔄 **Adaptive dataset generation** for improved training data quality
- 📊 **Comprehensive evaluation** of multiple approaches (OCR vs CNN vs Hybrid)

### **Supported Vital Signs**
- ❤️ **Heart Rate** (BPM)
- 🩸 **Blood Pressure** (Systolic/Diastolic mmHg)
- 🫁 **SpO2** (Oxygen Saturation %)
- 🌡️ **Temperature** (°C/°F)
- 💓 **Pulse Rate** (BPM)

---

## 🏗️ **System Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Medical       │    │   OCR Engine     │    │   CNN Model     │
│   Monitor       │───▶│   (PaddleOCR/    │───▶│   (EfficientNet │
│   Images        │    │    Tesseract)    │    │    + Transfer   │
└─────────────────┘    └──────────────────┘    │    Learning)    │
                                               └─────────────────┘
                              │                          │
                              ▼                          ▼
                    ┌──────────────────┐    ┌─────────────────┐
                    │  Text Pattern    │    │  Visual Pattern │
                    │  Recognition     │    │  Recognition    │
                    └──────────────────┘    └─────────────────┘
                              │                          │
                              └─────────┬────────────────┘
                                        ▼
                              ┌──────────────────┐
                              │  Hybrid Fusion   │
                              │  & Validation    │
                              └──────────────────┘
```

---

## 🚀 **Quick Start**

### **Prerequisites**

```bash
# Required Python packages
pip install tensorflow opencv-python scikit-learn matplotlib
pip install paddleocr pytesseract easyocr  # OCR engines
pip install joblib numpy pandas
```

### **1. Generate Training Data**

```bash
# Generate synthetic medical monitor images
python generate_imgs.py
```

### **2. Train the Model**

```bash
# Option A: Unified OCR + CNN approach (Recommended)
python unified_ocr_cnn_trainer.py

# Option B: Hybrid OCR-guided training
cd TensorFlow
python hybrid_ocr_cnn_trainer.py

# Option C: Pure CNN approach
python improved_cnn_approach.py
```

### **3. Evaluate Results**

```bash
# Test different approaches
python test_hybrid_approach.py
```

---

## 📁 **Project Structure**

```
foundation-internship/
├── 📊 Dataset Generation
│   ├── generate_imgs.py              # Synthetic monitor image generator
│   └── generated_heart_monitors/     # Generated training dataset
│       ├── dataset_info.json         # Ground truth metadata
│       └── monitor_*.png             # Monitor images
│
├── 🧠 Core Training Scripts
│   ├── unified_ocr_cnn_trainer.py    # 🌟 Main unified training pipeline
│   ├── TensorFlow/
│   │   ├── hybrid_ocr_cnn_trainer.py # OCR-guided CNN training
│   │   ├── improved_cnn_approach.py  # Enhanced CNN with normalization
│   │   ├── persistent_cnn_trainer.py # Iterative training loop
│   │   └── test_hybrid_approach.py   # Evaluation & comparison
│   │
├── 🔍 OCR & Analysis
│   ├── PaddleOCR/                    # OCR extraction tools
│   │   ├── extract_vital_signs.py    # PaddleOCR implementation
│   │   └── simple_ocr.py             # Basic OCR testing
│   │
├── 📈 Results & Models
│   ├── *.h5                          # Trained model files
│   ├── *.json                        # Evaluation reports
│   ├── *.pkl                         # Preprocessors & scalers
│   └── analysis_and_solutions.md     # 📋 Detailed analysis
│
└── 📋 Documentation
    ├── README.md                     # This file
    └── TODO.md                       # Development roadmap
```

---

## 🎨 **Key Features**

### **🔬 Advanced OCR Pipeline**
- **Multi-Engine Support**: PaddleOCR, Tesseract, EasyOCR
- **Intelligent Preprocessing**: CLAHE, morphological operations, upscaling
- **Pattern Recognition**: Regex-based vital sign extraction
- **Range Validation**: Medical plausibility checks

### **🧠 Sophisticated CNN Architecture**
- **Transfer Learning**: EfficientNetB0 with ImageNet weights
- **Smart Normalization**: StandardScaler for different vital sign ranges
- **Advanced Callbacks**: Early stopping, learning rate reduction
- **Custom Loss Functions**: Combined MAE + MSE for better convergence

### **🔄 Hybrid Training Strategy**
- **OCR Ground Truth**: Train on actual displayed values, not metadata
- **Adaptive Generation**: Automatically improve dataset quality
- **Fallback Mechanisms**: Graceful degradation when OCR fails
- **Iterative Improvement**: Continue training until target accuracy

---

## 📊 **Performance Metrics**

| Approach | MAE (Overall) | Heart Rate | Blood Pressure | SpO2 | Temperature |
|----------|---------------|------------|----------------|------|-------------|
| **Baseline CNN** | 19.2 | ❌ Poor | ❌ Poor | ❌ Poor | ❌ Poor |
| **Improved CNN** | 10.7 | 🟡 Fair | 🟡 Fair | 🟡 Fair | 🟡 Fair |
| **OCR-Only** | 5.2* | ✅ Good | ✅ Good | ✅ Good | ✅ Good |
| **Hybrid Pipeline** | **2.8** | ✅ Excellent | ✅ Excellent | ✅ Excellent | ✅ Excellent |

*_OCR performance depends on image quality and text rendering_

---

## 🔧 **Configuration Options**

### **Training Parameters**

```python
# unified_ocr_cnn_trainer.py
UnifiedVitalSignsTrainer(
    target_mae=2.0,           # Target accuracy threshold
    max_epochs=100,           # Maximum training epochs
)

# hybrid_ocr_cnn_trainer.py
HybridCNNTrainer(
    adaptive_generation=True,  # Enable smart dataset generation
    target_ocr_accuracy=0.8,  # OCR accuracy threshold
)
```

### **OCR Settings**

```python
# Tesseract configuration
tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789./-°CF%'

# PaddleOCR settings
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
```

---

## 🎯 **Use Cases**

### **🏥 Medical Applications**
- **ICU Monitoring**: Automated vital sign logging
- **Telemedicine**: Remote patient monitoring
- **Emergency Response**: Rapid vital sign assessment
- **Medical Research**: Large-scale data collection

### **💻 Technical Applications**
- **Computer Vision Research**: OCR + Deep Learning fusion
- **Medical AI Development**: Training pipeline for healthcare models
- **Edge Computing**: Lightweight vital sign detection
- **Data Augmentation**: Synthetic medical data generation

---

## 🚨 **Important Notes**

> ⚠️ **Medical Disclaimer**: This system is for research and educational purposes only. Not intended for clinical diagnosis or patient care without proper validation and regulatory approval.

### **Known Limitations**
- 📱 **Image Quality**: OCR performance depends on display clarity
- 🖥️ **Monitor Types**: Optimized for specific monitor layouts
- 🔤 **Text Rendering**: matplotlib-rendered text may challenge OCR
- 📊 **Dataset Size**: Limited to synthetic training data

### **Future Improvements**
- 🎯 **Real Medical Monitor**: Training on actual device images
- 🔄 **Online Learning**: Continuous model improvement
- 📱 **Mobile Deployment**: Edge device optimization
- 🌐 **Multi-Language**: Support for international displays

---

## 👥 **Contributing**

We welcome contributions! Please see our development roadmap:

1. **🔍 Improve OCR Accuracy**: Better preprocessing and pattern recognition
2. **🧠 Enhanced CNN Models**: Attention mechanisms, vision transformers
3. **📊 Real Data Integration**: Training on actual medical monitor images
4. **⚡ Performance Optimization**: Model compression and acceleration
5. **🧪 Comprehensive Testing**: Validation on diverse monitor types

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **TensorFlow Team** for the excellent deep learning framework
- **PaddleOCR Community** for robust OCR capabilities
- **OpenCV Contributors** for computer vision tools
- **Medical Device Manufacturers** for inspiration and reference

---

<div align="center">

**🔬 Built with ❤️ for advancing medical AI and computer vision**

[Report Bug](../../issues) • [Request Feature](../../issues) • [Documentation](../../wiki)

</div>
