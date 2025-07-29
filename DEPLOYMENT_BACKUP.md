# 🚀 **Deployment Guide - Advanced ML Sentiment Analysis**

## 📋 **Quick Deployment Options**

### **Option 1: Streamlit Cloud (Recommended)**
1. **Fork/Clone**: Repository is ready at `LixPix/H3-NLP-Amazon-Products-SEntiment-Analysis`
2. **Connect**: Link your GitHub account to [Streamlit Cloud](https://streamlit.io/cloud)
3. **Deploy**: Select `streamlit_dashboard.py` as main file
4. **Auto-Deploy**: Changes automatically deploy from main branch

### **Option 2: Local Development**
```bash
# Clone repository
git clone https://github.com/LixPix/H3-NLP-Amazon-Products-SEntiment-Analysis.git
cd H3-NLP-Amazon-Products-SEntiment-Analysis

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run streamlit_dashboard.py
```

### **Option 3: Advanced ML Dashboard**
```bash
# For full ML capabilities including BERT
streamlit run advanced_streamlit_dashboard.py
```

---

## 🔧 **Technical Requirements**

### **Core Dependencies**
```
streamlit>=1.28.0
pandas>=2.1.1
matplotlib>=3.7.0
plotly>=5.17.0
wordcloud>=1.9.2
nltk>=3.8.1
cryptography>=41.0.0
```

### **Advanced ML Dependencies (Optional)**
```
scikit-learn>=1.3.0
transformers>=4.36.0
torch>=2.1.0
seaborn>=0.12.0
joblib>=1.3.0
```

---

## 📊 **Dashboard Features**

### **Main Dashboard** (`streamlit_dashboard.py`)
- ✅ **Core Analytics**: Sentiment analysis and visualization
- ✅ **Interactive Filters**: Real-time data filtering
- ✅ **Word Clouds**: Visual text mining
- ✅ **Performance Optimized**: Fast loading and responsive
- ✅ **Privacy Compliant**: Encrypted user data

### **Advanced Dashboard** (`advanced_streamlit_dashboard.py`)
- ✅ **ML Integration**: BERT, Random Forest, SVM models
- ✅ **Real-time Prediction**: Interactive model testing
- ✅ **Model Comparison**: Performance benchmarking
- ✅ **Enterprise Features**: Advanced analytics and insights

---

## 🚀 **Live Demo**

**URL**: [https://h3-nlp-amazon-products-sentiment-analysis.streamlit.app](https://h3-nlp-amazon-products-sentiment-analysis.streamlit.app)

---

## 📝 **Development Workflow**

### **1. Data Processing**
```bash
# Run Jupyter notebook to process data and train models
jupyter notebook jupyter_notebooks/NLP_Amazon_Products_Sentiment_Anlysis.ipynb
```

### **2. Dashboard Testing**
```bash
# Test main dashboard
streamlit run streamlit_dashboard.py

# Test advanced features
streamlit run advanced_streamlit_dashboard.py
```

### **3. Deploy to Cloud**
- **Push changes** to GitHub main branch
- **Streamlit Cloud** auto-deploys changes
- **Monitor** deployment logs in Streamlit dashboard

---

## 🔒 **Security & Privacy**

- **Data Encryption**: ProfileName columns encrypted with Fernet (AES-128-CBC)
- **GDPR Compliant**: No personal data exposed in visualizations
- **Secure Processing**: All analysis performed on encrypted/anonymized data
- **Cloud Security**: Streamlit Cloud provides HTTPS and secure hosting

---

## 📈 **Performance Optimization**

### **Caching Strategy**
- `@st.cache_data` for data loading (1-hour TTL)
- Optimized data types (categories for sentiment)
- Pre-calculated aggregations for faster rendering

### **Resource Management**
- **Main Dashboard**: <100MB RAM, suitable for Streamlit Cloud free tier
- **Advanced Dashboard**: <500MB RAM, requires GPU for optimal BERT performance

---

## 🛠️ **Troubleshooting**

### **Common Issues**

**Data File Not Found**
```python
# Ensure preprocessed data exists
# Run Jupyter notebook first to generate H3_reviews_preprocessed.csv
```

**BERT Model Loading Issues**
```python
# Install transformers
pip install transformers torch

# Or use main dashboard for core features without BERT
streamlit run streamlit_dashboard.py
```

**Deployment Timeout**
```
# Use main dashboard for faster deployment
# Advanced features require more resources
```

---

## 📞 **Support**

- **Repository**: [GitHub Issues](https://github.com/LixPix/H3-NLP-Amazon-Products-SEntiment-Analysis/issues)
- **Documentation**: See `README.md` for detailed project information
- **Streamlit Community**: [Streamlit Forums](https://discuss.streamlit.io/)

---

*🚀 This deployment guide covers both basic and advanced deployment scenarios for optimal user experience.*
