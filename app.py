import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Page configuration
st.set_page_config(page_title="Data Analysis", layout="wide")
st.title("Escort Services Data Analysis")

st.markdown("""
This dashboard provides comprehensive analysis of escort services data. The analysis is divided into 4 main sections:

1. **Distributions** - Shows the distribution of key features like age, height and clothing sizes
2. **Correlations** - Analyzes relationships between different attributes
3. **Hypothesis** - Tests hypothesis about correlation between breast size and price
4. **Price Prediction** - ML model to predict prices based on features
""")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("data.csv")
    data = data.replace(u'\xa0', u'', regex=True)
    return data

data = load_data()

# Show raw data
if st.checkbox("Show raw data"):
    st.subheader("Below is the raw data from our dataset. You can see the first few rows and basic information about the columns.")
    st.write(data.head())
    # st.write(data.info())
    
    # Add new columns
    data['Salary'] = data['Price_USD'] * 22
    data['Before Retirement'] = 58 - data['Age']
    
    # Display data
    st.subheader("Modified data with 2 new columns")
    st.dataframe(data.head())
    
    # Add download button for updated dataset
    st.download_button(
        label="Download updated data",
        data=data.to_csv(index=False).encode('utf-8'),
        file_name='new_data.csv',
        mime='text/csv'
    )

# Data processing
numeric_columns = ['Age', 'Height', 'Weight', 'Price_USD']
categorical_columns = ['Size', 'Metro']

# Data cleaning
cleaned_data = data.copy()
cleaned_data[numeric_columns] = cleaned_data[numeric_columns].fillna(cleaned_data[numeric_columns].median())
cleaned_data[categorical_columns] = cleaned_data[categorical_columns].fillna(cleaned_data[categorical_columns].mode().iloc[0])

# Tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Correlations", "Hypothesis", "Price Prediction"])

with tab1:
    st.markdown("""
    # Distribution Analysis
    ## Here we analyze the distribution of key features in our dataset:
    
    ### 1. *Age Distribution* - Bar chart showing frequency of different age groups (using Seaborn)
    ### 2. *Height Distribution* - Stem plot showing height frequencies (using Matplotlib)
    ### 3. *Size Distribution* - Pie chart showing proportion of different clothing sizes (using Matplotlib)
    
    **Libraries used:**
    - **Matplotlib - for basic plotting functionality**
    - **Seaborn - for statistical data visualization**
    """)
    
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    fig1.set_facecolor('#f5f5f5')

    sns.countplot(x='Age', 
                data=cleaned_data,
                color='#FF69B4',
                alpha=0.7,
                ax=ax1)

    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.set_title('Age Distribution', 
                fontsize=20, 
                pad=20,
                fontweight='bold',
                color='#2f4f4f')
    ax1.set_xlabel('Age', fontsize=14, labelpad=10, color='#2f4f4f')
    ax1.set_ylabel('Frequency', fontsize=14, labelpad=10, color='#2f4f4f')

    # Chart settings
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#2f4f4f')
    ax1.spines['bottom'].set_color('#2f4f4f')

    for container in ax1.containers:
        ax1.bar_label(container, padding=3, color='#2f4f4f')

    ax1.set_facecolor('white')
    plt.tick_params(colors='#2f4f4f')
    st.pyplot(fig1)

    # Height distribution plot
    height_bins = pd.cut(cleaned_data['Height'], bins=range(153, 185, 2), right=False)
    height_counts = height_bins.value_counts().sort_index()

    fig2, ax2 = plt.subplots(figsize=(15, 6))
    fig2.set_facecolor('#f0f0f0')

    markerline, stemlines, baseline = ax2.stem(
        height_counts.index.astype(str),
        height_counts,
        basefmt=" ",
        linefmt='darkblue',
        markerfmt='o'
    )

    plt.setp(markerline, color='darkred', markersize=8)
    plt.setp(stemlines, linewidth=2, alpha=0.6)

    ax2.set_title('Height Distribution',
                fontsize=18,
                pad=20,
                fontweight='bold',
                color='#2f4f4f')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

    # Size pie chart
    size_counts = cleaned_data['Size'].value_counts().to_dict()
    sizes = list(size_counts.keys())
    counts = list(size_counts.values())

    fig3, ax3 = plt.subplots(figsize=(12, 8))
    ax3.pie(counts, startangle=140)
    ax3.set_title('Size Distribution', fontsize=16)
    plt.legend(title="Sizes",
            labels=[f'Size {size}: {count}' for size, count in zip(sizes, counts)],
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1))
    plt.axis('equal')
    st.pyplot(fig3)

with tab2:
    st.markdown("""
    # Correlation Analysis
    ### This section explores relationships between different features:
    
    1. ### **Breast Size vs Age** - Box plot showing breast size distribution across age groups
    2. ### **Height vs Clothing Size** - Scatter plot with regression line showing relationship between height and clothing size
    """)
    
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Convert Age to integers
    cleaned_data['Age'] = cleaned_data['Age'].astype(int)
    
    sns.boxplot(x="Age", y="Boobs", 
                data=cleaned_data,
                palette=sns.color_palette("rainbow", n_colors=len(cleaned_data['Age'].unique())),
                width=0.7,
                ax=ax1)
    
    ax1.set_title("Breast Size Distribution by Age Groups", 
                    fontsize=14, 
                    pad=15,
                    color='#2f4f4f')
    
    ax1.set_xlabel("Age", fontsize=12, color='#2f4f4f')
    ax1.set_ylabel("Breast Size", fontsize=12, color='#2f4f4f')
    
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_facecolor('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    st.pyplot(fig1)
    st.markdown("""### How to Read This Box Plot:
    - Each box represents a different age group
    - The middle line in each box is the median breast size for that age
    - The box itself shows where 50% of the data falls (25th to 75th percentile)
    - The points extend to show the full range of sizes
    - Any points beyond the boxes are potential outliers""")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.regplot(x="Height", y="Size", 
                data=cleaned_data,
                scatter_kws={
                    "s": 80,
                    "color": "#FF69B4",
                    "alpha": 0.6,
                    "edgecolor": "white"
                },
                line_kws={
                    "color": "#4B0082",
                    "linewidth": 2,
                    "linestyle": "--"
                },
                ax=ax2)

    ax2.set_title("Clothing Size vs Height Relationship", 
                    fontsize=14, 
                    pad=15, 
                    color='#2f4f4f')

    ax2.set_xlabel("Height (cm)", fontsize=12, color='#2f4f4f')
    ax2.set_ylabel("Clothing Size", fontsize=12, color='#2f4f4f')

    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_facecolor('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#2f4f4f')
    ax2.spines['bottom'].set_color('#2f4f4f')

    st.pyplot(fig2)
    st.markdown("""### How to Read This Scatter Plot:
    - The pink dots show actual data points, with some transparency to show overlapping
    - The purple dashed line is the regression line that shows the general trend
    - An upward slope indicates a positive correlation: as height increases, clothing size tends to increase
    - Points far from the line represent cases that deviate from the general trend""")

with tab3:
    st.markdown("""
    ### Hypothesis Testing
    This section tests our main hypothesis about pricing and breast size:
    
    Initial Hypothesis: Escorts with larger breast sizes command higher prices for their services.
    
    We'll examine this through:
    1. **Price Distribution by Breast Size** - Interactive histogram showing price ranges across breast sizes
    2. **Average Service Cost Analysis** - Bar chart and statistical breakdown of average prices by breast size
    
    Let's analyze the data to see if there's a significant correlation between breast size and pricing.
    """)
    
    # Create price bins
    price_bins = pd.cut(cleaned_data['Price_USD'], 
                    bins=range(0, 1001, 100), 
                    right=False,
                    include_lowest=True)

    cleaned_data['Price_Bins'] = price_bins.apply(
        lambda x: f'{int(x.left)}-{int(x.right)} USD' if pd.notna(x) else 'Other'
    )

    # Plotly histogram
    fig = px.histogram(cleaned_data, 
                    x='Price_Bins', 
                    color='Boobs',
                    title='Price Distribution by Breast Size',
                    labels={'Price_Bins': 'Price (USD)', 
                            'Boobs': 'Breast Size',
                            'count': 'Count'})

    fig.update_layout(
        xaxis_title='Price Range (USD)',
        yaxis_title='Count',
        xaxis={'categoryorder': 'array',
            'categoryarray': [f'{i}-{i+100} USD' for i in range(0, 1000, 100)]}
    )

    st.plotly_chart(fig, use_container_width=True)

    # Average prices
    st.subheader("Average Service Cost by Breast Size")
    average_prices = cleaned_data.groupby('Boobs')['Price_USD'].mean().round(2)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.write("Detailed Information:")
        for size, price in average_prices.items():
            st.write(f"Breast Size {size}: ${price}")

    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        average_prices.plot(kind='bar', ax=ax)
        plt.title('Average Service Cost by Breast Size')
        plt.xlabel('Breast Size')
        plt.ylabel('Cost ($)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=0)
        
        for i, v in enumerate(average_prices):
            plt.text(i, v + 5, f'${v:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
    st.markdown("""
    ### Model Analysis Results
    
    Our initial hypothesis about significant price variations based on physical parameters has been disproven:
    - The data shows that earnings are relatively uniform across different physical characteristics
    - This suggests that other factors (such as location, services, or market conditions) may be more influential
        """)
    

with tab4:

    st.subheader("Price Prediction")
    st.markdown("""
    ### Model Details
    
    For price prediction, we utilized:
    - **Random Forest Regressor** as our main model
    - **StandardScaler** for feature normalization
    - **Cross-validation** to ensure model robustness
    - **GridSearchCV** for hyperparameter tuning
    
    Key model characteristics:
    - Features used: Age, Height, Weight, Breast Size, Clothing Size
    - Train/Test split: 80/20
    - Model performance metrics were optimized using RMSE
    - Hyperparameters were tuned for optimal performance
    
    """)
    
    # Load saved model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        loaded_scaler = pickle.load(file)
    
    st.subheader("Enter parameters for price prediction:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=58, value=25)
        height = st.number_input("Height (cm)", min_value=140, max_value=200, value=165)
        weight = st.number_input("Weight (kg)", min_value=40, max_value=120, value=55)
        
    with col2:
        breast_size = st.number_input("Breast Size", min_value=1, max_value=12, value=2)
        size = st.number_input("Size", min_value=30, max_value=85, value=42)

    if st.button("Calculate Price"):
        # Prepare data for prediction
        input_data = pd.DataFrame({
            'Age': [age],
            'Boobs': [breast_size],
            'Height': [height], 
            'Size': [size],
            'Weight': [weight]
        })
        
        # Scale the input data
        input_scaled = loaded_scaler.transform(input_data)
        
        # Get prediction
        prediction = model.predict(input_scaled)[0]
        
        # Format prediction as float with 2 decimal places
        predicted_price = float(prediction)
        
        col1, col2, col3 = st.columns(3)
                
        with col1:
            st.metric("Predicted Price", f"${predicted_price:.2f}")
        
        with col2:
            st.metric("Minimum Price", f"${predicted_price * 0.8:.2f}")
        
        with col3:
            st.metric("Maximum Price", f"${predicted_price * 1.2:.2f}")
            
        st.write("""
        The prediction shows:
        - The most likely price based on your inputs
        - A minimum recommended price (20% below prediction)
        - A maximum recommended price (20% above prediction)
        """)
