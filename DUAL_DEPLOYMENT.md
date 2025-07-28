# 🚀 **Dual Dashboard Deployment Guide**

## 📊 **Two Streamlit Cloud Deployments Available**

This project provides **two distinct dashboards** for different use cases:

### **🎯 Dashboard 1: Main Analytics Dashboard**
- **File**: `streamlit_dashboard.py`
- **Focus**: Streamlined sentiment analysis with traditional NLP
- **Best For**: Business stakeholders, quick insights, production use
- **Features**: Fast loading, clean UI, essential analytics

### **🤖 Dashboard 2: Advanced ML Dashboard** 
- **File**: `advanced_streamlit_dashboard.py`
- **Focus**: State-of-the-art ML with BERT, Random Forest, SVM
- **Best For**: Data scientists, ML engineers, advanced analysis
- **Features**: BERT integration, model comparison, deep insights

---

## 🌐 **Cloud Deployment Instructions**

### **Option 1: Deploy Main Analytics Dashboard**

1. **Go to**: [share.streamlit.io](https://share.streamlit.io)
2. **Click**: "New app"
3. **Configure**:
   ```
   Repository: LixPix/H3-NLP-Amazon-Products-SEntiment-Analysis
   Branch: main
   Main file path: streamlit_dashboard.py
   App URL: amazon-sentiment-analytics
   ```
4. **Deploy**: Click "Deploy!"

**Expected URL**: `https://amazon-sentiment-analytics.streamlit.app`

### **Option 2: Deploy Advanced ML Dashboard**

1. **Go to**: [share.streamlit.io](https://share.streamlit.io)
2. **Click**: "New app"
3. **Configure**:
   ```
   Repository: LixPix/H3-NLP-Amazon-Products-SEntiment-Analysis
   Branch: main
   Main file path: advanced_streamlit_dashboard.py
   App URL: amazon-sentiment-advanced-ml
   ```
4. **Deploy**: Click "Deploy!"

**Expected URL**: `https://amazon-sentiment-advanced-ml.streamlit.app`

---

## 📋 **Deployment Comparison**

| Feature | Main Dashboard | Advanced ML Dashboard |
|---------|----------------|----------------------|
| **Load Time** | ~3 seconds | ~10 seconds |
| **Memory Usage** | ~100MB | ~500MB |
| **Dependencies** | Minimal | Full ML stack |
| **ML Models** | Rule-based + Traditional | BERT + Ensemble |
| **Target Users** | Business users | Technical users |
| **Use Case** | Daily operations | Research & development |

---

## 🔧 **Technical Requirements**

### **Main Dashboard Dependencies:**
```txt
streamlit>=1.28.0
pandas>=2.1.1
matplotlib>=3.7.0
plotly>=5.17.0
wordcloud>=1.9.2
nltk>=3.8.1
cryptography>=41.0.0
```

### **Advanced ML Dashboard Dependencies:**
```txt
streamlit>=1.28.0
pandas>=2.1.1
plotly>=5.17.0
scikit-learn>=1.3.0
transformers>=4.36.0
torch>=2.1.0
seaborn>=0.12.0
joblib>=1.3.0
```

---

## 🚀 **Quick Deployment Steps**

### **Step 1: Create Both Apps**
1. Deploy **Main Dashboard** first (faster setup)
2. Deploy **Advanced ML Dashboard** second (requires more resources)

### **Step 2: Test Both Deployments**
- **Main**: Quick business analytics
- **Advanced**: ML model testing and comparison

### **Step 3: Share Appropriate Links**
- **Business Team**: Main dashboard URL
- **Technical Team**: Advanced ML dashboard URL

---

## 💡 **Recommended Deployment Strategy**

1. **Primary Deployment**: Use **Main Dashboard** for general access
2. **Secondary Deployment**: Use **Advanced ML Dashboard** for technical analysis
3. **Load Balancing**: Direct users based on their needs

### **URL Naming Convention:**
- **Main**: `amazon-sentiment-analytics.streamlit.app`
- **Advanced**: `amazon-sentiment-advanced-ml.streamlit.app`

---

## 📞 **Support & Troubleshooting**

### **If Main Dashboard Fails:**
- Check basic dependencies in `requirements.txt`
- Verify data file `H3_reviews_preprocessed.csv` exists

### **If Advanced ML Dashboard Fails:**
- May require higher memory tier on Streamlit Cloud
- BERT model might need internet access during first load
- Consider disabling BERT for cloud deployment if needed

---

**🎯 Both dashboards provide enterprise-grade sentiment analysis with different technical approaches!**
