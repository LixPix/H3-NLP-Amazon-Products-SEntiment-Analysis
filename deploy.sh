#!/bin/bash

# Cloud Deployment Script for Amazon Reviews Analytics Dashboard
# This script helps prepare and test your dashboard for cloud deployment

echo "🚀 Preparing Amazon Reviews Analytics Dashboard for Cloud Deployment"
echo "================================================================="

# Check if required files exist
echo "📁 Checking required files..."

if [ ! -f "H3_reviews_preprocessed.csv" ]; then
    echo "❌ Error: H3_reviews_preprocessed.csv not found"
    echo "💡 Please ensure your data file is in the project root directory"
    exit 1
fi

if [ ! -f "streamlit_dashboard.py" ]; then
    echo "❌ Error: streamlit_dashboard.py not found"
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found"
    exit 1
fi

echo "✅ All required files found"

# Check file sizes for cloud deployment optimization
echo "📊 Checking file sizes for cloud optimization..."

CSV_SIZE=$(du -h "H3_reviews_preprocessed.csv" | cut -f1)
echo "📄 Data file size: $CSV_SIZE"

if [ $(stat -c%s "H3_reviews_preprocessed.csv") -gt 25000000 ]; then
    echo "⚠️  Warning: Data file is larger than 25MB"
    echo "💡 Consider using file upload functionality for cloud deployment"
fi

# Test local installation
echo "🔧 Testing local installation..."

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip not found. Please install pip"
    exit 1
fi

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Test dashboard startup
echo "🎯 Testing dashboard startup..."
echo "Starting Streamlit dashboard for testing..."
echo "Press Ctrl+C to stop the test and continue with deployment"

# Start Streamlit with cloud-optimized settings
streamlit run streamlit_dashboard.py --server.headless true --server.port 8503 &
STREAMLIT_PID=$!

sleep 5

# Check if Streamlit is running
if ps -p $STREAMLIT_PID > /dev/null; then
    echo "✅ Dashboard started successfully on port 8503"
    echo "🌐 Visit: http://localhost:8503"
    echo ""
    echo "🚀 Ready for cloud deployment!"
    echo ""
    echo "Next steps for Streamlit Cloud deployment:"
    echo "1. Push your code to GitHub"
    echo "2. Go to https://share.streamlit.io"
    echo "3. Connect your GitHub repository"
    echo "4. Deploy with main file: streamlit_dashboard.py"
    echo ""
    echo "Press Ctrl+C to stop the test server"
    
    # Wait for user to stop
    wait $STREAMLIT_PID
else
    echo "❌ Dashboard failed to start"
    echo "Please check the error messages above"
    exit 1
fi

echo "🎉 Deployment preparation complete!"
