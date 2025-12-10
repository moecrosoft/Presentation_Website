import streamlit as st
from streamlit_lottie import st_lottie
import json
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.figure_factory as ff
import pydeck as pdk
from PIL import Image

df = pd.read_csv("HDB_Resale_Prices_Data_Engineered.csv")

df_imp = pd.read_csv("HDB_Resale_Prices_Features_Importances.csv")

st.set_page_config(page_title='Group 2',page_icon='💻',layout='wide')

col1,col2,col3 = st.columns([1,2,1])

with col2:
    st.markdown('# Group 2')
    st.subheader('Meet Our Team! :wave:')
    
    members = [
        {'name': 'Darryl', 'role': 'Feature Engineer'},
        {'name': 'Mabel', 'role': 'EDA Specialist'},
        {'name': 'Isabel', 'role': 'Research Analyst'},
        {'name': 'Chew', 'role': 'Data Engineer'},
        {'name': 'Moe', 'role': 'Machine Learning Engineer'},
    ]

    for member in members:
        st.write(f'- **{member['name']}**   ({member['role']})')
        
    st.markdown("<hr style='border: 2px solid #bbb;'>", unsafe_allow_html=True)

    st.subheader('Workflow Overview')
    st.write(
        '''
        - Data Cleaning on the original HDB Resale Prices dataset
        - EDA to analyse the original dataset
        - Feature Engineering to create new features
        - Build ML models to predict HDB Resale Prices
        - EDA on model feature importances
        - Present our findings and insights
        '''
    )

    st.markdown("<hr style='border: 2px solid #bbb;'>", unsafe_allow_html=True)

    st.subheader('Exploratory Data Analysis (EDA)')
    st.write(
        '''
        - For the EDA, we explored the original HDB resale dataset to understand key patterns, trends, and relationships in the data
        - This included analysing price distributions, identifying influential variables such as floor area, lease years, and accessibility, and examining temporal trends in resale prices
        - These insights helped guide our feature engineering and informed which factors were most relevant for building accurate prediction models
        '''
    )

    st.markdown("<hr style='border: 2px solid #bbb;'>", unsafe_allow_html=True)

    st.subheader('Feature Engineering')
    st.write(
        '''
        - Feature Engineering is Where the Model Learns “Intelligence”
        - Raw data alone cannot explain resale prices well
        - Feature engineering transforms simple data into meaningful signals the model can use
        - Turned Raw HDB Data Into Structured Insights
        - Added Real-World Singapore Context
        - Integrated external datasets to reflect actual factors buyers care about
        - Engineered 80+ New Features Across Key Domains
        - Improved Model Accuracy Significantly
        - Created Features That Are Both Predictive and Interpretable
        - Ensures Our Final ML Model Reflects Real Singapore Housing Dynamics
        '''
    )

    st.markdown("<hr style='border: 2px solid #bbb;'>", unsafe_allow_html=True)

    st.subheader('Machine Learning')
    st.write(
        '''
        - Load the Feature Engineered Dataset into a dataframe
        - Split the dataset into features and target
        - Create training and testing sets
        - Define evaluation functions
        - Build a preprocessor pipeline for the data
        - Create a Linear Regression model pipeline and evaluate the model
        - Create a Random Forest model pipeline and evaluate the model
        - Compare the performances of the two models
        - Create a feature importances dataframe from the better performing model
        - Export the dataframe as csv to facilitate further analysis
        '''
    )

    st.markdown("<hr style='border: 2px solid #bbb;'>", unsafe_allow_html=True)

    st.subheader('Machine Learning Model Performance Comparison')
    st.write('')
    st.write('')

    img = Image.open('ML.jpg')
    st.image(img,use_column_width=True)

    st.markdown("<hr style='border: 2px solid #bbb;'>", unsafe_allow_html=True)

    st.subheader('The HDB Resale Market ')
    st.write('The hdb resale market is dominated by a few well defined structural factors')
    st.write('')
    st.write('')
    
    img3 = Image.open('Photo_3.jpg')
    st.image(img3)

st.markdown("<hr style='border: 2px solid #bbb;'>", unsafe_allow_html=True)

with st.container():
    left_col,right_col = st.columns(2)
    with left_col:
        important_cols = ['resale_price','floor_area_sqm','remaining_lease_years','lease_age','storey_avg']

        numeric_df = df[important_cols]
        
        # Create Label Mapping for Prettier Heatmap Labels 
        label_mapping = {
            'resale_price': 'Resale Price',
            'floor_area_sqm': 'Floor Area (sqm)',
            'remaining_lease_years': 'Remaining Lease (Years)',
            'lease_age': 'Lease Age',
            'storey_avg': 'Storey Average'
        }
        
        # Compute Correlation Matrix
        corr = numeric_df.corr()
        
        # Rename Labels for Display Only 
        corr = corr.rename(index=label_mapping, columns=label_mapping)

        fig, ax = plt.subplots(figsize=(10,5))
        sns.heatmap(corr, annot=True, cmap='Blues', fmt=".2f", square=True)
        ax.set_title("Heatmap of Most Influential Variables\n", fontsize=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45,ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center')
        st.pyplot(fig, use_container_width=True)

    with right_col:
        st.subheader('What does this tell us?')
        st.write('''
        - Strong correlation between floor area and resale price
        - Top 10 features that affect resale price include lease attributes, accessibility, and flat characterisitics
        ''')

with st.container():
    left_col,right_col = st.columns(2)
    with left_col:
        fig, ax = plt.subplots(figsize=(18,9))
        sns.histplot(
        df['floor_area_sqm'],
        bins='fd',       
        stat='density',
        color='skyblue',
        alpha=0.6,
        label='Histogram'
        )
    
        sns.kdeplot(
            df['floor_area_sqm'],
            color='red',
            linewidth=2,
            label='KDE Curve'
        )

        ax.set_title('Distribution of Floor Area (sqm)', fontsize=20)
        ax.set_xlabel('Floor Area (sqm)', fontsize=16)
        ax.set_ylabel('Density', fontsize=16)
        st.pyplot(fig, use_container_width=True)
    with right_col:
        st.subheader('Why is this important?')
        st.write('''
        - Policies and prediction of HDB resale prices should focus only on the top features
        - Not all features contribute to resale price equally
        ''')

st.markdown("<hr style='border: 2px solid #bbb;'>", unsafe_allow_html=True)

with st.container():
    left_col,right_col = st.columns(2)
    with left_col:
        top10 = df_imp.head(10)
        fig1, ax1 = plt.subplots(figsize=(18,9))
        sns.barplot(data=top10, x='importance', y='feature', hue='feature', palette='viridis', dodge=False, ax=ax1)
        ax1.set_title('Top 10 Most Important Features Influencing Resale Prices', fontsize=20)
        ax1.set_xlabel('Importance Score', fontsize=16)
        ax1.set_ylabel('Feature', fontsize=16)
        ax1.tick_params(axis='both', labelsize=14)
        ax1.legend().remove()  # optional, removes redundant legend for features
        st.pyplot(fig1, use_container_width=True)
    with right_col:
        st.subheader('Accessibility Impacts on Resale Value')
        st.write('- The resale market rewards accessibility disproportionately.')
        st.write('###### **Time to CBD is ranked 2nd for features that influence resale prices**')
        st.write('''
        - Accessibility is a primary factor in resale value
        - A 5-10 minute reduction in travel time is worth much more than lease attibutes or ammenities in the surrounding area
        ''')
        
st.markdown("<hr style='border: 2px solid #bbb;'>", unsafe_allow_html=True) 

with st.container():

    df['year_quarter'] = df['year'].astype(str) + " Q" + df['quarter'].astype(str)

    top10 = df_imp.head(10)

    quarterly = df.groupby(['year', 'quarter'])['resale_price'].mean()
    
    left_col,right_col = st.columns(2)
    with left_col:
        fig, ax = plt.subplots(figsize=(18,9))
        quarterly.plot(marker='o', ax=ax)
        ax.set_title('Average Resale Price by Quarter', fontsize=20)
        ax.set_xlabel('Year - Quarter', fontsize=16)
        ax.set_ylabel('Average Resale Price ($)', fontsize=16)
        st.pyplot(fig, use_container_width=True)
    
        # Top 10 features barplot
        fig1, ax1 = plt.subplots(figsize=(18,9))
        sns.barplot(data=top10, x='importance', y='feature', hue='feature', palette='viridis', dodge=False, ax=ax1)
        ax1.set_title('Top 10 Most Important Features Influencing Resale Prices', fontsize=20)
        ax1.set_xlabel('Importance Score', fontsize=16)
        ax1.set_ylabel('Feature', fontsize=16)
        ax1.tick_params(axis='both', labelsize=14)
        ax1.legend().remove()  # optional, removes redundant legend for features
        st.pyplot(fig1, use_container_width=True)

    with right_col:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.subheader('Year of HDB sale and Resale Price')
        st.write('''
        - There is a sharp jump in the resale prices of HDBs from 2019-2020 which coincides with the COVID-19 pandemic
        - Construction delays led to a surge in demand of resale flats: BTOs got delayed (92 projects) which pushed people towards resale flats
        
        ''')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('- Year is also a top feature in our ML model which shows that market inflation (macroeconomic factors) contributes greatly to resale prices and not just flat attributes.')

st.markdown("<hr style='border: 2px solid #bbb;'>", unsafe_allow_html=True)


with st.container():
    left_col,right_col = st.columns(2)
    with left_col:
        fig, ax = plt.subplots(figsize=(18,9))
        sns.histplot(
        df['floor_area_sqm'],
        bins='fd',       
        stat='density',
        color='skyblue',
        alpha=0.6,
        label='Histogram'
        )
    
        sns.kdeplot(
            df['floor_area_sqm'],
            color='red',
            linewidth=2,
            label='KDE Curve'
        )

        ax.set_title('Distribution of Floor Area (sqm)', fontsize=20)
        ax.set_xlabel('Floor Area (sqm)', fontsize=16)
        ax.set_ylabel('Density', fontsize=16)
        st.pyplot(fig, use_container_width=True)

    with right_col:
        st.subheader('Floor Area Distribution')
        st.write('''
        - The floor area distribution shows clear peaks around 70–120 sqm, reflecting the standard sizes of common HDB flat types
        - The right-skew indicates that larger flats are less common, and the KDE curve highlights distinct clusters rather than a smooth spread
        - This supports the insight that floor area is the strongest predictor of resale price, as buyers consistently pay more for additional space
        ''')
        
st.markdown("<hr style='border: 2px solid #bbb;'>", unsafe_allow_html=True)

with st.container():
    left_col,right_col = st.columns(2)
    with left_col:
        img1 = Image.open('Photo_1.jpg')
        st.image(img1,caption='lease_commence_year and remaining_lease_years are in our top features',use_column_width=True)
    with right_col:
        st.subheader('Lease age and Depreciation')
        st.write('''
        ###### **Lease age matters more for mid-aged flats than very old ones due to diminishing marginal impact**
        - Remaining_lease_years has less importance than lease_commence_year
        - For flats above 30 years of age, depreciation due to lease loss slows
        - Could be due to Government Schemes, Locational advantages and Home Improvement Programmes
        ''')
st.write('')
st.write('')
        
with st.container():
    left_col,right_col = st.columns(2)
    with left_col:
        df_imp = df_imp[df_imp['feature'] != 'resale_price']
        
        # Top 10 most important features from feature importances CSV
        top_features = df_imp.head(10)['feature'].tolist()
        
        # Check which of these exist in the engineered dataset
        available_features = [f for f in top_features if f in df.columns]
        missing_features   = [f for f in top_features if f not in df.columns]
        
        # Build a subset using only features that exist + resale_price
        corr_subset = df[available_features + ['resale_price']]
        
        # Correlation matrix
        corr_matrix = corr_subset.corr()
        
        # Correlation values with resale_price (sorted)
        corr_with_price = corr_matrix['resale_price'].sort_values(ascending=False)
        
        # Prettify labels
        label_map = {col: col.replace('_', ' ').title() for col in corr_matrix.columns}
        label_map['resale_price'] = 'Resale Price'   # custom rename
        
        corr_matrix_pretty = corr_matrix.rename(index=label_map, columns=label_map)
        fig, ax = plt.subplots(figsize=(10,5))
        sns.heatmap(
        corr_matrix_pretty, 
        annot=True, 
        cmap='Blues', 
        fmt=".2f", 
        square=True
        )
        ax.set_title("Correlation Between Important Features and Resale Price\n", fontsize=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45,ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center')
        st.pyplot(fig, use_container_width=True)
        
    with right_col:
        st.subheader('However, their correlation to resale value is weaker than floor area and year of sale')
        st.write('''
        - We can deduce that as the flat ages, loss of lease years depreciates lease value less sharply
        - For flats above 30 years of age, depreciation due to lease loss slows
        - Could be due to Government Schemes, Locational advantages and Home Improvement Programmes
        ''')
        
st.markdown("<hr style='border: 2px solid #bbb;'>", unsafe_allow_html=True)

col1,col2,col3 = st.columns([1,2,1])

with col2:
    st.markdown('# Conclusion')
    st.subheader('Although we have reached a high prediction accuracy, HDB resale price is more nuanced than what has been discussed in this model')
    st.subheader('There are many other factors such as Government Regulations, age distribution in certain towns as well as the priorities of the buyers, that can greatly impact the resale value as well as the percieved value of a HDB')
