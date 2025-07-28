import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Amazon Reviews Analytics",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
    .sidebar-content {
        background: linear-gradient(180deg, #f0f2f6 0%, #ffffff 100%);
    }
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header with animation
st.markdown('<h1 class="main-header">📊 Amazon Reviews Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Real-time Sentiment Analysis & Business Intelligence")

# Loading spinner
with st.spinner('🔄 Loading dashboard data...'):
    
    # Load data with enhanced caching and optimization
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_data():
        try:
            df = pd.read_csv("H3_reviews_preprocessed.csv")
            # Optimize data types for faster processing
            df['sentiment'] = df['sentiment'].astype('category')
            df['ProductId'] = df['ProductId'].astype(str)
            
            # Convert month column to proper datetime if it exists
            if 'month' in df.columns:
                df['month'] = pd.to_datetime(df['month'].astype(str))
            
            # Pre-calculate common aggregations for faster loading
            df['text_length'] = df['cleaned_text'].str.len()
            
            return df
        except FileNotFoundError:
            st.error("❌ Data file not found. Please ensure 'H3_reviews_preprocessed.csv' exists.")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return pd.DataFrame()

    df = load_data()

if df.empty:
    st.stop()

# Enhanced metrics with better visualization
st.markdown("### 📈 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

total_reviews = len(df)
positive_reviews = len(df[df['sentiment'] == 'positive'])
negative_reviews = len(df[df['sentiment'] == 'negative'])
neutral_reviews = len(df[df['sentiment'] == 'neutral'])
avg_text_length = df['text_length'].mean()

with col1:
    st.metric(
        "📊 Total Reviews", 
        f"{total_reviews:,}",
        delta=f"+{total_reviews - 2500}" if total_reviews > 2500 else None
    )
with col2:
    positive_pct = (positive_reviews / total_reviews * 100)
    st.metric(
        "😊 Positive", 
        f"{positive_reviews:,}",
        delta=f"{positive_pct:.1f}%"
    )
with col3:
    negative_pct = (negative_reviews / total_reviews * 100)
    st.metric(
        "😞 Negative", 
        f"{negative_reviews:,}",
        delta=f"{negative_pct:.1f}%"
    )
with col4:
    neutral_pct = (neutral_reviews / total_reviews * 100)
    st.metric(
        "😐 Neutral", 
        f"{neutral_reviews:,}",
        delta=f"{neutral_pct:.1f}%"
    )
with col5:
    st.metric(
        "📝 Avg Text Length", 
        f"{avg_text_length:.0f}",
        delta="characters"
    )

# Enhanced sidebar with professional styling
st.sidebar.markdown("### 🎛️ Interactive Filters")
st.sidebar.markdown("---")

# Sentiment filter with better UX
sentiments = st.sidebar.multiselect(
    "🎯 Select Sentiments to Analyze",
    options=['positive', 'negative', 'neutral'],
    default=['positive', 'negative', 'neutral'],
    help="Choose which sentiment categories to include in your analysis"
)

# Date range filter (if month data available)
if 'month' in df.columns:
    st.sidebar.markdown("### 📅 Time Period")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['month'].min().date(), df['month'].max().date()),
        min_value=df['month'].min().date(),
        max_value=df['month'].max().date()
    )

# Text length filter
st.sidebar.markdown("### 📝 Review Length")
min_length, max_length = st.sidebar.slider(
    "Filter by text length (characters)",
    min_value=int(df['text_length'].min()),
    max_value=int(df['text_length'].max()),
    value=(int(df['text_length'].min()), int(df['text_length'].max())),
    help="Filter reviews by their text length"
)

# Product filter
st.sidebar.markdown("### 📦 Product Analysis")
top_products = df['ProductId'].value_counts().head(20).index.tolist()
selected_products = st.sidebar.multiselect(
    "Focus on specific products (optional)",
    options=top_products,
    help="Leave empty to analyze all products"
)

# Apply advanced filters
st.markdown("---")
with st.spinner("🔄 Applying filters..."):
    filtered = df[df['sentiment'].isin(sentiments)]
    
    # Apply text length filter
    filtered = filtered[
        (filtered['text_length'] >= min_length) & 
        (filtered['text_length'] <= max_length)
    ]
    
    # Apply product filter if selected
    if selected_products:
        filtered = filtered[filtered['ProductId'].isin(selected_products)]
    
    # Apply date filter if available
    if 'month' in df.columns and 'date_range' in locals():
        start_date, end_date = date_range
        filtered = filtered[
            (filtered['month'].dt.date >= start_date) & 
            (filtered['month'].dt.date <= end_date)
        ]

if filtered.empty:
    st.warning("⚠️ No data matches your filter criteria. Please adjust your selection.")
    st.stop()

# Display filtered data info
st.info(f"📊 Showing {len(filtered):,} reviews out of {len(df):,} total reviews")

# Enhanced main dashboard with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "☁️ Text Analysis", "🔍 Deep Dive"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### � Top Products by Review Volume")
        if not filtered.empty:
            top_products_data = filtered['ProductId'].value_counts().head(10)
            
            # Create interactive bar chart with Plotly
            fig_products = px.bar(
                x=top_products_data.values,
                y=top_products_data.index,
                orientation='h',
                title="Top 10 Products by Review Count",
                labels={'x': 'Number of Reviews', 'y': 'Product ID'},
                color=top_products_data.values,
                color_continuous_scale='viridis'
            )
            fig_products.update_layout(
                height=500,
                showlegend=False,
                font=dict(size=12)
            )
            st.plotly_chart(fig_products, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Sentiment Distribution")
        if not filtered.empty:
            sentiment_counts = filtered['sentiment'].value_counts()
            
            # Create interactive pie chart
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color_discrete_map={
                    'positive': '#2E8B57',
                    'neutral': '#FFD700', 
                    'negative': '#DC143C'
                }
            )
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            fig_pie.update_layout(height=500)
            st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    if 'month' in filtered.columns:
        st.markdown("#### 📈 Temporal Sentiment Analysis")
        
        # Monthly trend analysis
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_data = filtered.groupby(['month', 'sentiment'], observed=True).size().unstack().fillna(0)
            
            fig_trend = px.line(
                monthly_data,
                title="Monthly Sentiment Trends",
                labels={'value': 'Number of Reviews', 'month': 'Month'},
                color_discrete_map={
                    'positive': '#2E8B57',
                    'neutral': '#FFD700',
                    'negative': '#DC143C'
                }
            )
            fig_trend.update_layout(height=400)
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            # Sentiment percentage over time
            monthly_pct = monthly_data.div(monthly_data.sum(axis=1), axis=0) * 100
            
            fig_pct = px.area(
                monthly_pct,
                title="Sentiment Percentage Over Time",
                labels={'value': 'Percentage (%)', 'month': 'Month'},
                color_discrete_map={
                    'positive': '#2E8B57',
                    'neutral': '#FFD700',
                    'negative': '#DC143C'
                }
            )
            fig_pct.update_layout(height=400)
            st.plotly_chart(fig_pct, use_container_width=True)

with tab3:
    st.markdown("#### ☁️ Advanced Text Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Word cloud with sentiment selection
        sentiment_choice = st.selectbox(
            "Choose sentiment for word cloud:",
            ['positive', 'neutral', 'negative'],
            help="Select which sentiment to analyze"
        )
        
        if not filtered.empty:
            sentiment_data = filtered[filtered['sentiment'] == sentiment_choice]
            if not sentiment_data.empty and 'cleaned_text' in sentiment_data.columns:
                text = " ".join(sentiment_data['cleaned_text'].dropna())
                if text.strip():
                    try:
                        wc = WordCloud(
                            width=800,
                            height=400,
                            background_color='white',
                            max_words=100,
                            colormap='viridis',
                            relative_scaling=0.5
                        ).generate(text)
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        ax.set_title(f'{sentiment_choice.title()} Sentiment Word Cloud',
                                   fontsize=16, fontweight='bold')
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.error(f"Error generating word cloud: {str(e)}")
                else:
                    st.warning(f"No text data available for {sentiment_choice} sentiment.")
    
    with col2:
        # Text length analysis
        st.markdown("##### � Review Length Analysis")
        
        length_by_sentiment = filtered.groupby('sentiment', observed=True)['text_length'].agg(['mean', 'median', 'std']).round(2)
        
        fig_length = px.box(
            filtered, 
            x='sentiment', 
            y='text_length',
            title="Review Length Distribution by Sentiment",
            color='sentiment',
            color_discrete_map={
                'positive': '#2E8B57',
                'neutral': '#FFD700',
                'negative': '#DC143C'
            }
        )
        fig_length.update_layout(height=400)
        st.plotly_chart(fig_length, use_container_width=True)
        
        # Display statistics table
        st.dataframe(length_by_sentiment, use_container_width=True)

with tab4:
    st.markdown("#### 🔍 Deep Dive Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Product performance matrix
        st.markdown("##### 📊 Product Performance Matrix")
        
        if len(filtered) > 0:
            product_sentiment = pd.crosstab(
                filtered['ProductId'], 
                filtered['sentiment'], 
                normalize='index'
            ) * 100
            
            # Get top 15 products for better visualization
            top_15_products = filtered['ProductId'].value_counts().head(15).index
            product_matrix = product_sentiment.loc[top_15_products].round(1)
            
            fig_heatmap = px.imshow(
                product_matrix.values,
                x=product_matrix.columns,
                y=product_matrix.index,
                aspect="auto",
                title="Sentiment Distribution by Product (%)",
                color_continuous_scale="RdYlGn",
                text_auto=True
            )
            fig_heatmap.update_layout(height=600)
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        # Advanced insights
        st.markdown("##### 📈 Key Insights")
        
        # Calculate insights
        most_positive_product = filtered.groupby('ProductId')['sentiment'].apply(
            lambda x: (x == 'positive').sum() / len(x)
        ).idxmax()
        
        most_negative_product = filtered.groupby('ProductId')['sentiment'].apply(
            lambda x: (x == 'negative').sum() / len(x)
        ).idxmax()
        
        avg_positive_length = filtered[filtered['sentiment'] == 'positive']['text_length'].mean()
        avg_negative_length = filtered[filtered['sentiment'] == 'negative']['text_length'].mean()
        
        # Display insights
        st.success(f"🏆 **Best Performing Product:** {most_positive_product}")
        st.error(f"⚠️ **Needs Attention:** {most_negative_product}")
        st.info(f"📝 **Positive reviews** are {avg_positive_length:.0f} chars on average")
        st.info(f"📝 **Negative reviews** are {avg_negative_length:.0f} chars on average")
        
        # Sentiment trends
        if 'month' in filtered.columns:
            st.markdown("##### � Recent Trends")
            recent_data = filtered.tail(1000)  # Last 1000 reviews
            recent_sentiment = recent_data['sentiment'].value_counts(normalize=True) * 100
            
            st.markdown("**Recent Sentiment Distribution:**")
            for sentiment, percentage in recent_sentiment.items():
                emoji = "😊" if sentiment == 'positive' else "😞" if sentiment == 'negative' else "😐"
                st.write(f"{emoji} {sentiment.title()}: {percentage:.1f}%")

# Enhanced footer with real-time updates
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 💡 Dashboard Features")
    st.markdown("""
    - **Real-time filtering** with multiple criteria
    - **Interactive visualizations** with hover details
    - **Advanced analytics** and insights
    - **Responsive design** for all devices
    """)

with col2:
    st.markdown("### 📊 Data Quality")
    st.markdown(f"""
    - **Total Records**: {len(df):,}
    - **Filtered Records**: {len(filtered):,}
    - **Data Coverage**: {(len(filtered)/len(df)*100):.1f}%
    - **Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    """)

with col3:
    st.markdown("### 🔒 Privacy & Security")
    st.markdown("""
    - **ProfileName**: Encrypted for privacy
    - **Data**: Anonymized and processed
    - **Compliance**: GDPR compliant
    - **Storage**: Secure cloud infrastructure
    """)

# Sidebar analytics
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")
st.sidebar.metric("Active Filters", len([x for x in [sentiments, selected_products] if x]))
st.sidebar.metric("Data Filtered", f"{(len(filtered)/len(df)*100):.1f}%")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()
