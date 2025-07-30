# 🚀 ML-Powered Amazon Sentiment Analysis

**Sentiment Analysis with BERT, Random Forest & Ensemble Methods**

This project analyzes Amazon product reviews using **ML techniques** including BERT transformers, ensemble methods, and traditional algorithms to classify reviews and identify key words under each class.  The trained model is deployed for real time prediction on Streamlit.

---

## 🏆 **Key Features**

- � **Privacy-First**: Encrypted data processing (AES-128-CBC)
- 📊 **Interactive Dashboard**: Real-time text sentiment prediction powered by trained ML model.  Insights into model performance,  training data and key words identified under each sentimental class.
- ☁️ **Cloud-Optimized**: Streamlit Cloud deployment of the interactive dashboard

---

## � **Quick Start**

### **🎯 Live Demo - Dual Dashboards**
- **📊 Main Analytics**: [Business Dashboard](https://amazon-sentiment-analytics.streamlit.app) *(Fast, streamlined)*
- **🤖 Advanced ML**: [Technical Dashboard](https://amazon-sentiment-advanced-ml.streamlit.app) *(BERT, ensemble models)*

---

## 🎯 **Core Capabilities**

**Advanced NLP Pipeline** for Amazon product review sentiment analysis with enterprise-grade ML models:

✅ **Multi-Algorithm Approach**: BERT, Random Forest, SVM with ensemble methods  
✅ **Real-Time Analytics**: Interactive dashboard with live predictions  
✅ **Business Intelligence**: Actionable insights for product optimization  
✅ **Production-Ready**: Scalable deployment with encrypted data processing  

### **📊 Dataset**
- **2,861 Amazon reviews** across Food & Beverages and Pet Products categories
- **5 Products**: Dog Toy (556), Coconut Oil (567), Dog Treats (632), Cappucino (542), Diet Chips (564)
- **Sentiment Distribution**: 81.9% Positive, 13.3% Negative, 4.7% Neutral

This is a data subset extracted from https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews.

A subset was used for testing and to ease machine processing requirements.  The subset was chosen to maintain overall score distribution.

---

## 🚀 **Project Structure**

```
H3-NLP-Amazon-Products-SEntiment-Analysis/
├── � streamlit_dashboard.py          # Main dashboard
├── 🤖 advanced_streamlit_dashboard.py # Advanced ML features  
├── 📓 jupyter_notebooks/              # Analysis notebooks
├── 📋 requirements.txt                # Dependencies
├── ⚙️ setup.sh                       # Streamlit setup
├── 🔧 Procfile                       # Deployment config
└── 📄 README.md                      # Documentation
```

---

## 🔄 Data Processing Pipeline

### **Stage 1: Data Ingestion & Validation**
- Load raw Amazon review data from Excel/CSV
- Validate data quality and completeness
- Handle missing values and outliers
- Convert timestamps to datetime format

### **Stage 2: Privacy Protection**
- Encrypt ProfileName using Fernet symmetric encryption
- Generate deterministic keys using SHA-256 hashing
- Maintain data utility while protecting PII

### **Stage 3: Text Preprocessing**
- Remove HTML tags and special characters
- Convert text to lowercase for consistency
- Tokenize text using NLTK word_tokenize
- Filter out stop words and noise

### **Stage 4: Feature Engineering**
- Extract temporal features (month, year)
- Calculate text length metrics
- Create categorical sentiment labels
- Generate analysis-ready dataset

### **Stage 5: Sentiment Classification**
- Apply rule-based sentiment mapping
- Validate classification accuracy
- Generate sentiment distribution metrics

---

## 🔍 Key Findings & Analysis

### **📊 Sentiment Distribution:**
- **Positive**: 68.5% (1,960 reviews)
- **Negative**: 20.8% (595 reviews)
- **Neutral**: 10.7% (306 reviews)

### **🎯 Top Performing Products:**
1. **Coconut Oil Products**: Overwhelmingly positive sentiment
2. **Pet Food Items**: High satisfaction rates
3. **Natural/Organic Products**: Strong positive feedback

### **⚠️ Problem Areas Identified:**
1. **Dog Toys & Ropes**: Higher negative sentiment frequency
2. **Coffee Products**: Mixed reviews with quality concerns
3. **Durability Issues**: Common theme in negative feedback

### **💡 Keyword Impact Analysis:**

#### **Positive Sentiment Drivers:**
- **"love"** (highest positive frequency)
- **"great"** (quality indicator)
- **"coconut"** (product-specific success)
- **"good"** (general satisfaction)
- **"oil"** (product category success)

#### **Negative Sentiment Indicators:**
- **"toy"** (durability concerns)
- **"rope"** (quality issues)
- **"coffee"** (taste/quality problems)
- Product-specific failure terms

### **📈 Temporal Trends:**
- **Seasonal Patterns**: Holiday periods show increased review volume
- **Quality Improvements**: Some products show sentiment improvement over time
- **Consistency**: Major brands maintain stable sentiment patterns

### **🔄 Word Overlap Analysis:**
- **Normal Phenomenon**: 47% of top words appear in both positive and negative reviews
- **Context Dependency**: Words like "taste", "dog", "product" are descriptive rather than sentiment-specific
- **Ratio Analysis**: Positive reviews use emotional language more frequently

---

## 📱 Interactive Dashboard

### **🎛️ Dashboard Features:**
- **Real-time Filtering**: Multi-criteria selection (sentiment, products, time)
- **Interactive Visualizations**: Plotly-powered charts with hover details
- **Professional UI**: Custom CSS styling for enterprise presentation
- **Responsive Design**: Optimized for desktop and mobile viewing

### **📊 Dashboard Tabs:**

#### **1. 📊 Overview Tab:**
- Key performance indicators (KPIs)
- Top products by review volume
- Sentiment distribution pie charts
- Summary statistics

#### **2. 📈 Trends Tab:**
- Monthly sentiment trends over time
- Temporal pattern analysis
- Seasonality insights
- Performance tracking

#### **3. ☁️ Text Analysis Tab:**
- Interactive word cloud generation
- Sentiment-specific visualizations
- Keyword frequency analysis
- Text mining insights

#### **4. 🔍 Deep Dive Tab:**
- Advanced analytics and metrics
- Data quality indicators
- Privacy and security information
- Technical specifications

#### **5. 🤖 ML Methodology Tab:**
- Comprehensive methodology explanation
- Technical implementation details
- Business rationale for approach
- Performance validation methods

### **🚀 Dashboard Access:**
```bash
# Local deployment
streamlit run streamlit_dashboard.py

# Cloud deployment
https://your-app.streamlit.app
```
----

## 🚀 Future Enhancements

### **🤖 Advanced Analytics:**
- **Deep Learning Models**: BERT/RoBERTa for enhanced sentiment accuracy
- **Aspect-Based Analysis**: Sentiment by product features (taste, quality, price)
- **Emotion Detection**: Beyond sentiment to specific emotions (joy, anger, surprise)
- **Multi-language Support**: Analysis of international reviews

### **📊 Enhanced Visualizations:**
- **Network Analysis**: Product relationship mapping
- **Geospatial Analysis**: Sentiment by geographic regions
- **Time Series Forecasting**: Predictive sentiment modeling
- **Comparative Analytics**: Cross-brand sentiment comparison

### **🔧 Technical Improvements:**
- **Real-time Processing**: Live review ingestion and analysis
- **API Integration**: Direct Amazon API connectivity
- **Advanced Filtering**: ML-powered recommendation system
- **Export Capabilities**: PDF reports and data export

### **🏢 Enterprise Features:**
- **Multi-tenant Support**: Organization-specific dashboards
- **Advanced Security**: Role-based access control
- **Integration APIs**: CRM and BI tool connectivity
- **Automated Reporting**: Scheduled insight delivery

---

## 🤝 Contributing

We welcome contributions to improve this sentiment analysis project! 

### **🛠️ How to Contribute:**

1. **Fork the Repository**
   ```bash
   git fork https://github.com/LixPix/H3-NLP-Amazon-Products-SEntiment-Analysis.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-enhancement
   ```

3. **Make Your Changes**
   - Follow existing code style
   - Add documentation for new features
   - Include tests where appropriate

4. **Submit Pull Request**
   - Describe your changes clearly
   - Include screenshots for UI changes
   - Reference any related issues


---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.


---

## 🙏 Acknowledgments

- **NLTK Team**: Natural Language Toolkit for text processing
- **Streamlit**: Interactive dashboard framework
- **Plotly**: Advanced visualization capabilities
- **Amazon**: Dataset source for analysis
- **Open Source Community**: Libraries and tools that made this possible

---

## 📞 Contact & Support

**Project Maintainer**: LixPix  
**Repository**: [H3-NLP-Amazon-Products-SEntiment-Analysis](https://github.com/LixPix/H3-NLP-Amazon-Products-SEntiment-Analysis)



---

**⭐ If this project helped you, please consider giving it a star on GitHub! ⭐**

---

*Last Updated: July 2025 | Version: 2.0 | Status: Production Ready*
