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
# with col3:
#     with open('team.json','r') as f:
#         team_gif = json.load(f)
#     st_lottie(team_gif,height=350,key='team')

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

# with col3:
#     with open('charts.json','r') as f:
#         chart_gif = json.load(f)
#     st_lottie(chart_gif,height=350,key='chart')

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
    # st.subheader('Conclusion')
    # st.write('''
    # - Although we have reached a high prediction accuracy, HDB resale price is more nuanced than what has been discussed in this model
    # - There are many other factors such as Government Regulations, age distribution in certain towns as well as the priorities of the buyers, that can greatly impact the resale value as well as the percieved value of a HDB
    # ''')
    st.markdown('# Conclusion')
    st.subheader('Although we have reached a high prediction accuracy, HDB resale price is more nuanced than what has been discussed in this model')
    st.subheader('There are many other factors such as Government Regulations, age distribution in certain towns as well as the priorities of the buyers, that can greatly impact the resale value as well as the percieved value of a HDB')

# with st.container():
#     left_col,right_col = st.columns(2)
#     with left_col:
#         max_price = int(df['resale_price'].max())
#         price_limit = st.slider("Max resale price", 0, max_price, max_price)

#         filtered_df = df[df['resale_price'] <= price_limit]

#         fig, ax = plt.subplots(figsize=(12,6))

#         sns.histplot(
#             filtered_df['resale_price'],
#             bins='fd',
#             stat='density',
#             color='skyblue',
#             alpha=0.6,
#             label='Histogram',
#             ax=ax
#         )

#         sns.kdeplot(
#             filtered_df['resale_price'],
#             color='red',
#             linewidth=2,
#             label='KDE Curve',
#             ax=ax
#         )

#         ax.set_title('Total Feature Importance by Category')
#         ax.set_xlabel('Total Importance Score')
#         ax.set_ylabel('Category')
#         ax.legend()

#         st.pyplot(fig,use_container_width=False)
        
#     with right_col:
#         st.subheader('Estate Age vs Depreciation')
#         st.write('''
#         The model shows that lease related features do influence resale prices, but less strongly than flat attributes such as floor area and accessibility. According to our research, HDB flats suffer a sharp depreciation in resale value around the 35 year mark of their lease, however this depreciation is not uniform across all towns. 
#         Depreciation in resale value can be mediated by Location / Accessibility
#         Amenities / Accessibility has greater importance resale price of HDB flat
#         Less mature towns rely on locational attributes to maintain value
#         ''')

# st.write('---')

# with st.container():
#     left_col,right_col = st.columns(2)
#     with left_col:
#         max_sqm = int(df['floor_area_sqm'].max())
#         sqm_limit = st.slider('Max Floor Area (sqm)',0,max_sqm,max_sqm)

#         filtered_df_sqm = df[df['floor_area_sqm'] <= sqm_limit]

#         fig2,ax2 = plt.subplots(figsize=(12,6))

#         sns.histplot(
#             filtered_df_sqm['floor_area_sqm'],
#             bins='fd',       
#             stat='density',
#             color='skyblue',
#             alpha=0.6,
#             label='Histogram'
#         )

#         sns.kdeplot(
#             filtered_df_sqm['floor_area_sqm'],
#             color='red',
#             linewidth=2,
#             label='KDE Curve'
#         )

#         ax2.set_title('Distribution of Floor Area (sqm)')
#         ax2.set_xlabel('Floor Area (sqm)')
#         ax2.set_ylabel('Density')
#         ax2.legend()

#         st.pyplot(fig2,use_container_width=False)
#     with right_col:
#         st.subheader('Distribution of Floor Area (sqm)')
#         st.write('''
#         The histogram and KDE curve on the left illustrate the distribution of floor area (in square meters) for HDB flats in the dataset. 
#         From the plot, we can observe that the majority of flats have a floor area ranging from approximately 20 to 100 square meters, with a peak around 40-60 square meters. 
#         This indicates that smaller flats are more common in the dataset, which is typical for urban housing markets where space is limited. 
#         The distribution also shows a right skew, suggesting that there are fewer larger flats with greater floor areas.
#         ''')

# st.write('---')

# with st.container():
#     left_col,right_col = st.columns(2)
#     with left_col:
#         max_sample = 10000
#         sample_limit = st.slider('Number of Samples',0,max_sample,2000)

#         filtered_sample_df = df.sample(n=sample_limit,random_state=42)

#         fig3,ax3 = plt.subplots(figsize=(12,6))
#         sns.scatterplot(
#             data=filtered_sample_df,
#             x='floor_area_sqm',
#             y='resale_price',
#             color='blue',
#             alpha=0.7
#         )

#         ax3.set_xlabel('Floor Area (sqm)')
#         ax3.set_ylabel('Resale Price ($)')
#         ax3.set_title("Relationship Between Floor Area and Resale Price (Sample of 2000 Flats)")

#         st.pyplot(fig3,use_container_width=False)
#     with right_col:
#         st.subheader('Relationship Between Floor Area and Resale Price')
#         st.write('''
#         The scatter plot on the left illustrates the relationship between the floor area (in square meters) of HDB flats and their resale prices. 
#         From the plot, we can observe a positive correlation between floor area and resale price, indicating that larger flats tend to have higher resale values. 
#         This trend is expected as larger flats generally offer more living space and amenities, making them more desirable in the housing market.
#         ''')

# st.write('---')

# with st.container():
#     left_col,right_col = st.columns(2)
#     with left_col:
#         max_sample_2 = 10000
#         sample_limit_2 = st.slider('Number of Samples',0,max_sample_2,2000,key='sample2')

#         filtered_sample_df_2 = df.sample(n=sample_limit_2,random_state=42)

#         fig4,ax4 = plt.subplots(figsize=(12,6))

#         sns.scatterplot(
#             data=filtered_sample_df_2,
#             x='remaining_lease_years',
#             y='resale_price',
#             color='green',
#             alpha=0.7
#         )

#         ax4.set_xlabel('Remaining Lease Years')
#         ax4.set_ylabel('Resale Price ($)')
#         ax4.set_title('Relationship Between Remaining Lease Years and Resale Price')

#         st.pyplot(fig4,use_container_width=False)
#     with right_col:
#         st.subheader('Relationship Between Remaining Lease Years and Resale Price')
#         st.write('''
#         The scatter plot on the left illustrates the relationship between the remaining lease years of HDB flats and their resale prices. 
#         From the plot, we can observe a general trend where flats with more remaining lease years tend to have higher resale prices. 
#         This is likely due to the fact that flats with longer leases are more attractive to buyers, as they offer greater security of tenure and potential for future value appreciation.
#         ''')

# st.write('---')

# with st.container():
#     left_col,right_col = st.columns(2)
#     with left_col:
#         max_sample_3 = 10000
#         sample_limit_3 = st.slider('Number of Samples',0,max_sample_3,2000,key='sample3')

#         filtered_sample_df_3 = df.sample(n=sample_limit_3,random_state=42)

#         fig5,ax5 = plt.subplots(figsize=(12,6))

#         sns.boxplot(data=filtered_sample_df_3,x='flat_type',y='resale_price')

#         ax5.set_xlabel('Flat Type')
#         ax5.set_ylabel('Resale Price ($)')
#         ax5.set_title('Relationship Between Flat Type and Resale Price')

#         st.pyplot(fig5,use_container_width=False)
#     with right_col:
#         st.subheader('Relationship Between Flat Type and Resale Price')
#         st.write('''
#         The box plot on the left illustrates the relationship between different flat types and their resale prices. 
#         From the plot, we can observe that larger flat types, such as Executive flats, tend to have higher resale prices compared to smaller flat types like 1-Room or 2-Room flats. 
#         This trend is expected as larger flats generally offer more space and amenities, making them more desirable in the housing market.
#         ''')

# st.write('---')

# with st.container():
#     left_col,right_col = st.columns(2)
#     with left_col:
#         max_sample_4 = 10000
#         sample_limit_4 = st.slider('Number of Samples',0,max_sample_4,2000,key='sample4')

#         filtered_sample_df_4 = df.sample(n=sample_limit_4,random_state=42)

#         fig6,ax6 = plt.subplots(figsize=(12,6))

#         sns.boxplot(data=filtered_sample_df_4,x='storey_avg',y='resale_price')

#         ax6.set_xlabel('Storey Avg')
#         ax6.set_ylabel('Resale Price ($)')
#         ax6.set_title('Relationship Between Storey Avg and Resale Price')

#         st.pyplot(fig6,use_container_width=False)
#     with right_col:
#         st.subheader('Relationship Between Storey Avg and Resale Price')
#         st.write('''
#         The box plot on the left illustrates the relationship between the average storey level of HDB flats and their resale prices. 
#         From the plot, we can observe that flats located on higher storeys tend to have higher resale prices compared to those on lower storeys. 
#         This trend may be attributed to factors such as better views, increased privacy, and reduced noise levels associated with higher floors, which can enhance the desirability and value of the flats.
#         ''')

# st.write('---')

# with st.container():
#     left_col,right_col = st.columns(2)
#     with left_col:
#         max_sample_5 = 10000
#         sample_limit_5 = st.slider('Number of Samples',0,max_sample_5,2000,key='sample5')

#         df['year_quarter'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)

#         quarterly = df.groupby(['year','quarter'])['resale_price'].mean()

#         filtered_sample_df_5 = df.sample(n=sample_limit_5,random_state=42)

#         fig7,ax7 = plt.subplots(figsize=(12,6))
#         quarterly.plot(marker='o',ax=ax7)

#         ax7.set_xlabel('Year - Quarter')
#         ax7.set_ylabel('Average Resale Price ($)')
#         ax7.set_title('Average Resale Price by Quarter')

#         st.pyplot(fig7,use_container_width=False)
#     with right_col:
#         st.subheader('Average Resale Price by Quarter')
#         st.write('''
#         The line chart on the left illustrates the trend of average resale prices of HDB flats over different quarters from 2017 to 2023. 
#         We can observe fluctuations in prices, with a general upward trend over the years. 
#         This indicates that despite short-term variations, the overall market for HDB flats has been appreciating in value during this period.
#         ''')

# st.write('---')

# with st.container():
#     left_col,right_col = st.columns(2)
#     with left_col:
#         fig8,ax8 = plt.subplots(figsize=(12,6))

#         numeric_df = df[['resale_price','floor_area_sqm','remaining_lease_years']]
#         sns.heatmap(numeric_df.corr(),annot=True,cmap='Blues',ax=ax8)

#         st.pyplot(fig8,use_container_width=False)
#     with right_col:
#         st.subheader('Investigate temporal factors (month)')
#         st.write('''
#         The heatmap on the left displays the correlation matrix for key numerical features in the dataset: resale price, floor area (sqm), and remaining lease years. 
#         From the heatmap, we can see that there is a strong positive correlation between floor area and resale price, indicating that larger flats tend to have higher resale values. 
#         Conversely, there is a weak negative correlation between remaining lease years and resale price, suggesting that flats with fewer remaining lease years may have slightly lower resale values.
#         ''')

# st.write('---')

# mrt = pd.DataFrame([
#     # North South Line (NSL)
#     ("Jurong East", 1.3331, 103.7420),
#     ("Bukit Batok", 1.3490, 103.7496),
#     ("Bukit Gombak", 1.3580, 103.7517),
#     ("Choa Chu Kang", 1.3840, 103.7448),
#     ("Yew Tee", 1.3975, 103.7477),
#     ("Kranji", 1.4250, 103.7619),
#     ("Marsiling", 1.4324, 103.7741),
#     ("Woodlands", 1.4370, 103.7864),
#     ("Admiralty", 1.4407, 103.8001),
#     ("Sembawang", 1.4491, 103.8200),
#     ("Canberra", 1.4437, 103.8334),
#     ("Yishun", 1.4292, 103.8355),
#     ("Khatib", 1.4174, 103.8339),
#     ("Yio Chu Kang", 1.3818, 103.8441),
#     ("Ang Mo Kio", 1.3691, 103.8498),
#     ("Bishan", 1.3505, 103.8480),
#     ("Braddell", 1.3402, 103.8467),
#     ("Toa Payoh", 1.3320, 103.8471),
#     ("Novena", 1.3201, 103.8439),
#     ("Newton", 1.3135, 103.8399),
#     ("Orchard", 1.3040, 103.8318),
#     ("Somerset", 1.3002, 103.8384),
#     ("Dhoby Ghaut", 1.2988, 103.8463),
#     ("City Hall", 1.2931, 103.8520),
#     ("Raffles Place", 1.2830, 103.8513),
#     ("Marina Bay", 1.2763, 103.8547),
#     ("Marina South Pier", 1.2708, 103.8630),

#     # East West Line (EWL)
#     ("Tuas Link", 1.3409, 103.6368),
#     ("Tuas West Road", 1.3360, 103.6467),
#     ("Tuas Crescent", 1.3307, 103.6543),
#     ("Gul Circle", 1.3194, 103.6601),
#     ("Joo Koon", 1.3274, 103.6785),
#     ("Pioneer", 1.3375, 103.6973),
#     ("Boon Lay", 1.3386, 103.7064),
#     ("Lakeside", 1.3443, 103.7209),
#     ("Chinese Garden", 1.3423, 103.7324),
#     ("Clementi", 1.3145, 103.7654),
#     ("Dover", 1.3112, 103.7786),
#     ("Buona Vista", 1.3070, 103.7905),
#     ("Commonwealth", 1.3023, 103.7983),
#     ("Queenstown", 1.2940, 103.8061),
#     ("Redhill", 1.2897, 103.8165),
#     ("Tiong Bahru", 1.2851, 103.8301),
#     ("Outram Park", 1.2802, 103.8391),
#     ("Tanjong Pagar", 1.2760, 103.8459),
#     ("Raffles Place", 1.2830, 103.8513),
#     ("City Hall", 1.2931, 103.8520),
#     ("Bugis", 1.3006, 103.8560),
#     ("Lavender", 1.3077, 103.8632),
#     ("Kallang", 1.3114, 103.8714),
#     ("Aljunied", 1.3168, 103.8823),
#     ("Paya Lebar", 1.3180, 103.8925),
#     ("Eunos", 1.3197, 103.9031),
#     ("Kembangan", 1.3219, 103.9120),
#     ("Bedok", 1.3245, 103.9291),
#     ("Tanah Merah", 1.3273, 103.9461),
#     ("Simei", 1.3430, 103.9530),
#     ("Tampines", 1.3534, 103.9457),
#     ("Pasir Ris", 1.3730, 103.9492),
#     ("Expo", 1.3345, 103.9617),
#     ("Changi Airport", 1.3570, 103.9884),

#     # North East Line (NEL)
#     ("HarbourFront", 1.2653, 103.8222),
#     ("Outram Park (NEL)", 1.2801, 103.8391),
#     ("Chinatown", 1.2841, 103.8442),
#     ("Clarke Quay", 1.2888, 103.8463),
#     ("Dhoby Ghaut (NEL)", 1.2988, 103.8463),
#     ("Little India", 1.3066, 103.8495),
#     ("Farrer Park", 1.3126, 103.8532),
#     ("Boon Keng", 1.3194, 103.8611),
#     ("Potong Pasir", 1.3343, 103.8690),
#     ("Woodleigh", 1.3402, 103.8705),
#     ("Serangoon", 1.3491, 103.8730),
#     ("Kovan", 1.3603, 103.8840),
#     ("Hougang", 1.3705, 103.8923),
#     ("Buangkok", 1.3821, 103.8927),
#     ("Sengkang", 1.3903, 103.8950),
#     ("Punggol", 1.4051, 103.9028),

#     # Circle Line (CCL)
#     ("HarbourFront (CCL)", 1.2653, 103.8222),
#     ("Telok Blangah", 1.2708, 103.8093),
#     ("Labrador Park", 1.2725, 103.8020),
#     ("Pasir Panjang", 1.2761, 103.7910),
#     ("Haw Par Villa", 1.2827, 103.7813),
#     ("Kent Ridge", 1.2933, 103.7847),
#     ("One-North", 1.2995, 103.7876),
#     ("Buona Vista (CCL)", 1.3070, 103.7905),
#     ("Holland Village", 1.3123, 103.7961),
#     ("Farrer Road", 1.3170, 103.8071),
#     ("Botanic Gardens", 1.3225, 103.8150),
#     ("Caldecott", 1.3376, 103.8394),
#     ("Marymount", 1.3497, 103.8451),
#     ("Bishan (CCL)", 1.3505, 103.8480),
#     ("Lorong Chuan", 1.3526, 103.8645),
#     ("Serangoon (CCL)", 1.3491, 103.8730),
#     ("Bartley", 1.3425, 103.8797),
#     ("Tai Seng", 1.3359, 103.8873),
#     ("MacPherson", 1.3267, 103.8896),
#     ("Paya Lebar (CCL)", 1.3180, 103.8925),
#     ("Dakota", 1.3025, 103.8881),
#     ("Mountbatten", 1.3012, 103.8822),
#     ("Stadium", 1.3026, 103.8753),
#     ("Nicoll Highway", 1.2992, 103.8632),
#     ("Promenade", 1.2936, 103.8591),
#     ("Esplanade", 1.2922, 103.8573),
#     ("Bras Basah", 1.2967, 103.8501),
#     ("Dhoby Ghaut (CCL)", 1.2988, 103.8463),

#     # Downtown Line (DTL)
#     ("Bukit Panjang", 1.3784, 103.7638),
#     ("Cashew", 1.3789, 103.7647),
#     ("Hillview", 1.3712, 103.7683),
#     ("Beauty World", 1.3416, 103.7757),
#     ("King Albert Park", 1.3350, 103.7833),
#     ("Sixth Avenue", 1.3242, 103.7961),
#     ("Tan Kah Kee", 1.3253, 103.8075),
#     ("Botanic Gardens (DTL)", 1.3225, 103.8150),
#     ("Stevens", 1.3172, 103.8251),
#     ("Newton (DTL)", 1.3135, 103.8399),
#     ("Little India (DTL)", 1.3066, 103.8495),
#     ("Rochor", 1.3063, 103.8520),
#     ("Bugis (DTL)", 1.3006, 103.8560),
#     ("Promenade (DTL)", 1.2936, 103.8591),
#     ("Bayfront", 1.2827, 103.8591),
#     ("Downtown", 1.2794, 103.8525),
#     ("Telok Ayer", 1.2823, 103.8491),
#     ("Chinatown (DTL)", 1.2841, 103.8442),
#     ("Fort Canning", 1.2893, 103.8467),
#     ("Bencoolen", 1.2992, 103.8502),
#     ("Jalan Besar", 1.3078, 103.8568),
#     ("Bendemeer", 1.3129, 103.8640),
#     ("Geylang Bahru", 1.3216, 103.8715),
#     ("Mattar", 1.3262, 103.8831),
#     ("MacPherson (DTL)", 1.3267, 103.8896),
#     ("Ubi", 1.3313, 103.8982),
#     ("Kaki Bukit", 1.3365, 103.9095),
#     ("Bedok North", 1.3379, 103.9251),
#     ("Bedok Reservoir", 1.3453, 103.9334),
#     ("Tampines West", 1.3454, 103.9393),
#     ("Tampines East", 1.3568, 103.9538),
#     ("Upper Changi", 1.3415, 103.9618),
#     ("Expo (DTL)", 1.3345, 103.9617),

#     # Thomson-East Coast Line (TEL)
#     ("Woodlands North", 1.4489, 103.7855),
#     ("Woodlands", 1.4370, 103.7864),
#     ("Woodlands South", 1.4322, 103.7929),
#     ("Springleaf", 1.4031, 103.8254),
#     ("Lentor", 1.3901, 103.8358),
#     ("Mayflower", 1.3719, 103.8403),
#     ("Bright Hill", 1.3567, 103.8317),
#     ("Upper Thomson", 1.3521, 103.8261),
#     ("Caldecott (TEL)", 1.3376, 103.8394),
#     ("Mount Pleasant", 1.3262, 103.8334),
#     ("Stevens (TEL)", 1.3172, 103.8251),
#     ("Napier", 1.3069, 103.8234),
#     ("Orchard Boulevard", 1.3050, 103.8281),
#     ("Orchard (TEL)", 1.3040, 103.8318),
#     ("Great World", 1.2927, 103.8321),
#     ("Havelock", 1.2889, 103.8316),
#     ("Outram Park (TEL)", 1.2802, 103.8391),
#     ("Maxwell", 1.2794, 103.8444),
#     ("Shenton Way", 1.2765, 103.8479),
#     ("Marina Bay (TEL)", 1.2763, 103.8547),
#     ("Marina South", 1.2700, 103.8670),

# ], columns=["station_name", "lat", "lon"])

# with st.container():
#     left_column,right_column = st.columns(2)
#     with left_column:
#         st.map(mrt)
#     with right_column:
#         st.subheader('MRT Stations in Singapore')
#         st.write('These are all the MRT stations in Singapore plotted on the map below.')

# st.write('---')

# primary_top30 = pd.DataFrame([
#     ("Holy Innocents' Primary School", 1.36906, 103.89020),
#     ("Ai Tong School", 1.35816, 103.83541),
#     ("Nan Chiau Primary School", 1.38620, 103.89300),
#     ("Chongfu School", 1.43181, 103.83510),
#     ("Nanyang Primary School", 1.32030, 103.80740),
#     ("CHIJ Primary (Toa Payoh)", 1.33841, 103.84878),
#     ("Methodist Girls' School (Primary)", 1.34117, 103.77560),
#     ("Tao Nan School", 1.30501, 103.91141),
#     ("Kong Hwa School", 1.31538, 103.88961),
#     ("Nan Hua Primary School", 1.31416, 103.76614),
#     ("Catholic High School (Primary)", 1.35482, 103.84500),
#     ("Rulang Primary School", 1.34979, 103.71700),
#     ("CHIJ St. Nicholas Girls' School (Primary)", 1.3718, 103.8439),
#     ("Singapore Chinese Girls' Primary School", 1.31920, 103.82600),
#     ("St. Hilda’s Primary School", 1.35538, 103.94032),
#     ("Anglo-Chinese School (Junior)", 1.30946, 103.84141),
#     ("St. Joseph’s Institution Junior", 1.31630, 103.83550),
#     ("Fairfield Methodist School (Primary)", 1.31180, 103.78570),
#     ("Rosyth School", 1.37287, 103.87451),
#     ("Kuo Chuan Presbyterian Primary School", 1.34988, 103.70912),
#     ("Maris Stella High (Primary Section)", 1.33985, 103.87568),
#     ("Henry Park Primary School", 1.31834, 103.78573),
#     ("Anglo-Chinese School (Primary)", 1.31881, 103.83582),  
#     ("Red Swastika School", 1.32492, 103.94948),
#     ("Pei Chun Public School", 1.33800, 103.86760),
#     ("Princess Elizabeth Primary School", 1.34968, 103.74103),
#     ("Pei Hwa Presbyterian Primary School", 1.34100, 103.77490),
#     ("St. Anthony’s Primary School", 1.34670, 103.70700),
#     ("Pasir Ris Primary School", 1.37770, 103.93880),
#     ("South View Primary School", 1.38190, 103.74010),
# ], columns=["school_name", "lat", "lon"])

# with st.container():
#     left_col,right_col = st.columns(2)
#     with left_col:
#         st.map(primary_top30)
#     with right_col:
#         st.subheader('Top 30 Primary Schools in Singapore')
#         st.write('These are the locations of the top 30 primary schools in Singapore plotted on the map below.')

# st.write('---')

# secondary_top30 = pd.DataFrame([
#     ("Raffles Girls' School (Secondary)", 1.31660, 103.81790),
#     ("Nanyang Girls' High School", 1.31960, 103.80760),
#     ("Hwa Chong Institution (High School)", 1.32632, 103.80358),
#     ("Raffles Institution (Secondary)", 1.34240, 103.84720),
#     ("Methodist Girls' School (Secondary)", 1.34130, 103.77620),
#     ("Anglo-Chinese School (Independent)", 1.30270, 103.78070),  # placeholder
#     ("CHIJ St. Nicholas Girls' School", 1.37180, 103.84390),
#     ("Dunman High School", 1.30690, 103.88250),
#     ("Catholic High School (Secondary)", 1.35482, 103.84500),
#     ("River Valley High School", 1.33740, 103.69770),
#     ("Cedar Girls' Secondary School", 1.33700, 103.87130),
#     ("Singapore Chinese Girls' School (Secondary)", 1.31920, 103.82600),
#     ("Victoria School", 1.30820, 103.93290),
#     ("St Joseph's Institution (Secondary)", 1.32400, 103.84050),
#     ("Maris Stella High School (Secondary)", 1.33990, 103.87560),
#     ("Anglican High School", 1.32690, 103.94650),
#     ("Paya Lebar Methodist Girls' School (Secondary)", 1.34140, 103.88370),
#     ("Crescent Girls' School", 1.29570, 103.81990),
#     ("Temasek Secondary School", 1.32390, 103.93830),
#     ("Tanjong Katong Secondary School", 1.30590, 103.89780),
#     ("St Andrew’s Secondary School", 1.33370, 103.86770),
#     ("Hai Sing Catholic School", 1.37170, 103.94790),
#     ("St Anthony’s Canossian Secondary School", 1.32350, 103.92400),
#     ("CHIJ Secondary (Toa Payoh)", 1.33850, 103.84860),
#     ("CHIJ Katong Convent", 1.30930, 103.91240),
#     ("CHIJ St Joseph’s Convent", 1.37910, 103.89040),
#     ("Holy Innocents' High School", 1.37230, 103.89080),
#     ("Kuo Chuan Presbyterian Secondary School", 1.33790, 103.84980),
#     ("Chung Cheng High School (Main)", 1.30531, 103.89152),
#     ("Nan Hua High School", 1.31470, 103.76620),
# ], columns=["school_name", "lat", "lon"])

# with st.container():
#     left_col,right_col = st.columns(2)
#     with left_col:
#         st.map(secondary_top30)
#     with right_col:
#         st.subheader('Top 30 Secondary Schools in Singapore')
#         st.write('These are the locations of the top 30 secondary schools in Singapore plotted on the map below.')

# st.write('---')

# jc_schools = pd.DataFrame([
#     ("Anderson Serangoon Junior College", 1.3615, 103.8928),
#     ("Anglo-Chinese Junior College", 1.3055, 103.8280),
#     ("Catholic Junior College", 1.3447, 103.8392),
#     ("Dunman High School (JC Section)", 1.3069, 103.8825),
#     ("Eunoia Junior College", 1.3396, 103.8873),
#     ("Hwa Chong Institution (JC)", 1.3520, 103.7710),
#     ("Jurong Pioneer Junior College", 1.3399, 103.7139),
#     ("Nanyang Junior College", 1.3812, 103.7708),
#     ("National Junior College", 1.3248, 103.8444),
#     ("Raffles Institution (JC)", 1.3424, 103.8472),
#     ("River Valley High School (JC Section)", 1.3374, 103.6977),
#     ("St Andrew’s Junior College", 1.3337, 103.8677),
#     ("St Joseph’s Institution (JC)", 1.3240, 103.8405),
#     ("Tampines Meridian Junior College", 1.3531, 103.9414),
#     ("Temasek Junior College", 1.3239, 103.9383),
#     ("Victoria Junior College", 1.3082, 103.9329),
#     ("Yishun Innova Junior College", 1.4306, 103.8338),
# ], columns=["school_name", "lat", "lon"])

# with st.container():
#     left_col,right_col = st.columns(2)
#     with left_col:
#         st.map(jc_schools)
#     with right_col:
#         st.subheader('Junior Colleges in Singapore')
#         st.write('These are the locations of the junior colleges in Singapore plotted on the map below.')

# st.write('---')

# poly_schools = pd.DataFrame([
#     ("Singapore Polytechnic", 1.3030, 103.7714),
#     ("Ngee Ann Polytechnic", 1.3378, 103.7734),
#     ("Temasek Polytechnic", 1.3446, 103.9570),
#     ("Nanyang Polytechnic", 1.3716, 103.7728),
#     ("Republic Polytechnic", 1.4335, 103.7863),
# ], columns=["school_name", "lat", "lon"])

# with st.container():
#     left_col,right_col = st.columns(2)
#     with left_col:
#         st.map(poly_schools)
#     with right_col:
#         st.subheader('Polytechnics in Singapore')
#         st.write('These are the locations of the polytechnics in Singapore plotted on the map below.')

# st.write('---')

# malls = pd.DataFrame([
#     ("Marina Bay Sands (The Shoppes)", 1.2834, 103.8607),
#     ("Marina Square", 1.2920, 103.8570),
#     ("Suntec City", 1.2930, 103.8575),
#     ("Millenia Walk", 1.2927, 103.8605),
#     ("Raffles City Shopping Centre", 1.2937, 103.8530),
#     ("Capitol Piazza", 1.2947, 103.8515),
#     ("Funan", 1.2936, 103.8497),
#     ("Bugis Junction", 1.2997, 103.8553),
#     ("Bugis+", 1.2990, 103.8543),
#     ("Duo Galleria", 1.3008, 103.8573),
#     ("Chinatown Point", 1.2851, 103.8436),
#     ("People’s Park Complex", 1.2834, 103.8430),
#     ("People’s Park Centre", 1.2841, 103.8435),
#     ("Central @ Clarke Quay", 1.2895, 103.8461),
#     ("Velocity @ Novena Square", 1.3203, 103.8431),
#     ("Square 2", 1.3208, 103.8436),
#     ("United Square", 1.3187, 103.8433),
#     ("Goldhill Plaza", 1.3185, 103.8439),

#     # North
#     ("Northpoint City (Yishun)", 1.4295, 103.8353),
#     ("Sembawang Shopping Centre", 1.4472, 103.8204),
#     ("Sun Plaza (Sembawang)", 1.4491, 103.8200),
#     ("Causeway Point (Woodlands)", 1.4353, 103.7856),
#     ("Woods Square Mall", 1.4381, 103.7871),
#     ("Vista Point", 1.4346, 103.7919),
#     ("Canberra Plaza", 1.4443, 103.8299),
#     ("Junction 10", 1.3806, 103.7613),  # Bukit Panjang border

#     # North-East
#     ("NEX (Serangoon)", 1.3508, 103.8725),
#     ("Heartland Mall (Kovan)", 1.3597, 103.8866),
#     ("Hougang Mall", 1.3714, 103.8922),
#     ("Hougang 1", 1.3726, 103.8829),
#     ("Compass One (Sengkang)", 1.3925, 103.8950),
#     ("Rivervale Mall", 1.3898, 103.9023),
#     ("Rivervale Plaza", 1.3872, 103.9025),
#     ("Waterway Point (Punggol)", 1.4060, 103.9024),
#     ("Punggol Plaza", 1.3986, 103.9079),
#     ("Oasis Terraces", 1.4057, 103.9094),
#     ("The Seletar Mall (Fernvale)", 1.3920, 103.8795),

#     # East
#     ("Tampines Mall", 1.3532, 103.9454),
#     ("Tampines 1", 1.3540, 103.9450),
#     ("Century Square", 1.3533, 103.9458),
#     ("Our Tampines Hub", 1.3562, 103.9407),
#     ("Bedok Mall", 1.3245, 103.9291),
#     ("White Sands", 1.3722, 103.9495),
#     ("Downtown East (E!Hub / Market Square)", 1.3770, 103.9545),
#     ("Parkway Parade", 1.3023, 103.9058),
#     ("i12 Katong", 1.3058, 103.9076),
#     ("Katong Shopping Centre", 1.3044, 103.9053),
#     ("Katong Square", 1.3046, 103.9059),
#     ("KINEX (OneKM)", 1.3142, 103.8971),
#     ("PLQ Mall (Paya Lebar Quarter)", 1.3170, 103.8924),
#     ("Paya Lebar Square", 1.3182, 103.8920),
#     ("SingPost Centre", 1.3187, 103.8928),
#     ("Jewel Changi Airport", 1.3592, 103.9894),
#     ("Changi City Point", 1.3341, 103.9623),

#     # West
#     ("Westgate", 1.3341, 103.7423),
#     ("Jem", 1.3331, 103.7430),
#     ("IMM Mall", 1.3330, 103.7460),
#     ("Jurong Point", 1.3396, 103.7063),
#     ("Pioneer Mall", 1.3436, 103.6978),
#     ("West Mall", 1.3506, 103.7494),
#     ("Lot One Shoppers’ Mall", 1.3852, 103.7444),
#     ("Hillion Mall", 1.3788, 103.7629),
#     ("Bukit Panjang Plaza", 1.3788, 103.7640),
#     ("The Clementi Mall", 1.3147, 103.7654),
#     ("Grantral Mall Clementi", 1.3150, 103.7641),
#     ("West Coast Plaza", 1.3044, 103.7641),
#     ("The Star Vista", 1.3065, 103.7884),
#     ("The Rail Mall", 1.3535, 103.7764),
#     ("Gek Poh Shopping Centre", 1.3464, 103.7128),
#     ("Le Quest Mall", 1.3552, 103.7487),

#     # Central / Others
#     ("Thomson Plaza", 1.3546, 103.8333),
#     ("Tiong Bahru Plaza", 1.2863, 103.8273),
#     ("Bukit Timah Plaza", 1.3293, 103.7915),
#     ("Beauty World Centre", 1.3412, 103.7757),
#     ("Beauty World Plaza", 1.3416, 103.7755),
#     ("Coronation Shopping Plaza", 1.3238, 103.8118),
#     ("Marine Parade Centre", 1.3013, 103.9042),
#     ("Anchorpoint (Queenstown)", 1.2887, 103.8066),
#     ("Queensway Shopping Centre", 1.2884, 103.8047),
#     ("Chinatown Plaza", 1.2823, 103.8419),
#     ("Peninsula Plaza", 1.2931, 103.8523),
#     ("Mustafa Centre", 1.3106, 103.8560),
#     ("Sim Lim Square", 1.3039, 103.8549),
# ], columns=["mall_name", "lat", "lon"])

# with st.container():
#     left_col,right_col = st.columns(2)
#     with left_col:
#         st.map(malls)
#     with right_col:
#         st.subheader('Shopping Malls in Singapore')
#         st.write('These are the locations of various shopping malls in Singapore plotted on the map below.')

# st.write('---')

# bus_interchanges = pd.DataFrame([
#     ("Ang Mo Kio Interchange", 1.3694, 103.8493),
#     ("Bedok Interchange", 1.3245, 103.9291),
#     ("Bishan Interchange", 1.3503, 103.8490),
#     ("Boons Lay Interchange", 1.3386, 103.7064),
#     ("Bukit Batok Interchange", 1.3501, 103.7495),
#     ("Bukit Merah Interchange", 1.2779, 103.8274),
#     ("Choa Chu Kang Interchange", 1.3851, 103.7445),
#     ("Clementi Interchange", 1.3150, 103.7654),
#     ("Compassvale Interchange", 1.3936, 103.8975),
#     ("Hougang Central Interchange", 1.3713, 103.8928),
#     ("Jurong East Interchange", 1.3333, 103.7430),
#     ("Kovan Hub", 1.3603, 103.8840),
#     ("Pasir Ris Interchange", 1.3723, 103.9490),
#     ("Punggol Interchange", 1.4050, 103.9020),
#     ("Sembawang Interchange", 1.4492, 103.8200),
#     ("Sengkang Interchange", 1.3924, 103.8950),
#     ("Serangoon Interchange", 1.3508, 103.8725),
#     ("Tampines Interchange", 1.3533, 103.9453),
#     ("Toa Payoh Interchange", 1.3330, 103.8471),
#     ("Woodlands Interchange", 1.4371, 103.7864),
#     ("Yishun Interchange", 1.4295, 103.8353),
# ], columns=["interchange_name", "lat", "lon"])

# with st.container():
#     left_col,right_col = st.columns(2)
#     with left_col:
#         st.map(bus_interchanges)
#     with right_col:
#         st.subheader('Bus Interchanges in Singapore')
#         st.write('These are the locations of bus interchanges in Singapore plotted on the map below.')
        
# st.write('---')

# hawkers = pd.DataFrame([
#     ("Maxwell Food Centre", 1.2805, 103.8442),
#     ("Chinatown Complex Food Centre", 1.2837, 103.8431),
#     ("Amoy Street Food Centre", 1.2798, 103.8457),
#     ("Old Airport Road Food Centre", 1.3067, 103.8884),
#     ("Tiong Bahru Market", 1.2848, 103.8265),
#     ("Lau Pa Sat", 1.2805, 103.8504),
#     ("Newton Food Centre", 1.3121, 103.8398),
#     ("Tekka Centre", 1.3051, 103.8520),
#     ("Golden Mile Food Centre", 1.3023, 103.8630),
#     ("Hong Lim Food Centre", 1.2837, 103.8459),
#     ("Zion Riverside Food Centre", 1.2929, 103.8338),
#     ("Berseh Food Centre", 1.3064, 103.8570),
#     ("Geylang Serai Market", 1.3183, 103.8943),
#     ("Bedok 85 Fengshan", 1.3248, 103.9338),
#     ("Chomp Chomp Food Centre", 1.3639, 103.8666),
#     ("Serangoon Garden Market", 1.3642, 103.8658),
#     ("ABC Brickworks Food Centre", 1.2864, 103.8054),
#     ("Bukit Timah Market", 1.3412, 103.7758),
#     ("Jurong West St 52 HC", 1.3475, 103.7141),
#     ("Yishun Park Hawker Centre", 1.4310, 103.8371),
# ], columns=["hawker_name", "lat", "lon"])

# with st.container():
#     left_col,right_col = st.columns(2)
#     with left_col:
#         st.map(hawkers)
#     with right_col:
#         st.subheader('Famous Hawker Centres in Singapore')
#         st.write('These are the locations of famous hawker centres in Singapore plotted on the map below.')

# st.write('---')

# hospitals = pd.DataFrame([
#     # --- ACUTE GENERAL HOSPITALS ---
#     ("Changi General Hospital", 1.341600, 103.949300),
#     ("KK Women's and Children's Hospital", 1.310600, 103.844100),
#     ("Sengkang General Hospital", 1.395300, 103.886400),
#     ("Singapore General Hospital", 1.279900, 103.834300),
#     ("Khoo Teck Puat Hospital", 1.424600, 103.839400),
#     ("Tan Tock Seng Hospital", 1.320300, 103.844700),
#     ("Woodlands Health Campus", 1.443700, 103.785900),
#     ("Alexandra Hospital", 1.288800, 103.802800),
#     ("National University Hospital", 1.293500, 103.783900),
#     ("Ng Teng Fong General Hospital", 1.334300, 103.744500),
#     ("Mount Alvernia Hospital", 1.339300, 103.840600),
#     ("Crawfurd Hospital", 1.313200, 103.776000),
#     ("Farrer Park Hospital", 1.312300, 103.854300),
#     ("Gleneagles Hospital", 1.305500, 103.822400),
#     ("Mount Elizabeth Hospital", 1.304100, 103.835200),
#     ("Mount Elizabeth Novena Hospital", 1.321300, 103.843900),
#     ("Parkway East Hospital", 1.320900, 103.914400),
#     ("Raffles Hospital", 1.300200, 103.856200),
#     ("Thomson Medical Centre", 1.320900, 103.843300),

#     # --- COMMUNITY HOSPITALS ---
#     ("Outram Community Hospital", 1.279500, 103.835900),
#     ("Sengkang Community Hospital", 1.395500, 103.887900),
#     ("Yishun Community Hospital", 1.425200, 103.839300),
#     ("Jurong Community Hospital", 1.334800, 103.744200),
#     ("Ang Mo Kio Thye Hua Kwan Hospital", 1.372100, 103.852800),
#     ("Ren Ci Community Hospital", 1.324600, 103.844300),
#     ("St. Andrew's Community Hospital", 1.341900, 103.953700),
#     ("St. Luke's Hospital", 1.338600, 103.758200),

#     # --- SPECIALISED ---
#     ("Institute of Mental Health", 1.377300, 103.878000),

# ], columns=["hospital_name", "lat", "lon"])

# with st.container():
#     left_col,right_col = st.columns(2)
#     with left_col:
#         st.map(hospitals)
#     with right_col:
#         st.subheader('Hospitals in Singapore')
#         st.write('These are the locations of hospitals in Singapore plotted on the map below.')


