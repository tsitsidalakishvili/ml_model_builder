import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from collections import Counter
import io
from sklearn.impute import SimpleImputer

# Set page config
st.set_page_config(page_title='Oil spill analysis', page_icon='🛢️', layout="wide")
st.title('Oil spill analysis')

# Create tabs
tabs = ["Oil spill analysis", "ML Model Building"]
selected_tab = st.sidebar.radio("Select a tab:", tabs)



def analyze_data(df):
    # Analysis and visualization code goes here
    
    with st.expander("Initial Records in Dataset"):
        st.dataframe(df.head())

    with st.expander("Exploring the Dataset"):
        # Shape of the dataframe
        st.write("Shape of the dataframe:", df.shape)

        # Getting more info on the dataset
        st.text("Info on the dataset:")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        # Describe dataset
        st.write("Statistical summary of the dataset:")
        st.dataframe(df.describe())

        # Display missing values count
        st.write("Missing values in each column:")
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        st.table(missing_values)

    with st.expander("Dealing with Missing Values"):
        option = st.selectbox("Choose a method:",
                            ['Select an option', 'Drop rows with missing values', 'Impute with Mean', 'Impute with Median'])

        if option == 'Drop rows with missing values':
            df.dropna(inplace=True)
            st.success("Rows with missing values dropped successfully.")
            st.dataframe(df)

        elif option == 'Impute with Mean':
            imputer = SimpleImputer(strategy='mean')
            df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=['float64', 'int64'])), columns=df.select_dtypes(include=['float64', 'int64']).columns)
            df[df.select_dtypes(include=['float64', 'int64']).columns] = df_imputed
            st.success("Missing values imputed with column mean successfully.")
            st.dataframe(df)

        elif option == 'Impute with Median':
            imputer = SimpleImputer(strategy='median')
            df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=['float64', 'int64'])), columns=df.select_dtypes(include=['float64', 'int64']).columns)
            df[df.select_dtypes(include=['float64', 'int64']).columns] = df_imputed
            st.success("Missing values imputed with column median successfully.")
            st.dataframe(df)

    # Visualization of the dataset
    # Column 1
    col1, col2 = st.columns(2)
    
    # Row 1
    with col1:
        # Chart 1: Cause of Oil Spillage
        main_causes = df['Cause Category'].value_counts()
        fig1 = go.Figure(go.Bar(x=main_causes.index, y=main_causes.values))
        fig1.update_layout(title='Cause of Oil Spillage')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Chart 2: Cause Sub Category
        main_causes_sub = df['Cause Subcategory'].value_counts()
        fig2 = go.Figure(go.Bar(x=main_causes_sub.index, y=main_causes_sub.values))
        fig2.update_layout(title='Cause Sub Category')
        st.plotly_chart(fig2, use_container_width=True)

    
    # Chart 6: Most Frequent Oil Spillers 
    most_frequent_spillers = df['Operator ID'].value_counts().nlargest(20).index
    df_top_spillers = df[df['Operator ID'].isin(most_frequent_spillers)]

    # Layout for the Most Frequent Oil Spillers chart
    st.write("Top 20 Most Frequent Spillers")
    top_spiller_counts = df_top_spillers['Operator ID'].value_counts()
    fig6 = px.bar(top_spiller_counts, x=top_spiller_counts.values, y=top_spiller_counts.index, orientation='h', labels={'y': 'Operator ID'})
    fig6.update_layout(title='Most Frequent Oil Spillers (Bar Chart)')
    st.plotly_chart(fig6, use_container_width=True)



    # Map of accidents by the most frequent spillers under the bar chart
    st.write("Locations of Accidents by Most Frequent Spillers (Map)")
    fig_map = px.scatter_mapbox(
        df_top_spillers, 
        lat='Accident Latitude', 
        lon='Accident Longitude', 
        color='Operator ID',
        hover_name='Accident City', 
        hover_data=['Accident County', 'Accident State', 'Operator ID'], 
        zoom=3, 
        height=600
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    # Add this code here to create a download button for the JSON file
    json_str = df.to_json(orient='records')
    st.download_button(
        label="Download Data as JSON",
        data=json_str,
        file_name='oil_spill_data.json',
        mime='application/json'
    )






# Oil spill analysis tab
if selected_tab == "Oil spill analysis":
    # Provide information about the app
    with st.expander('About this app'):
        st.write("""
            **What is this app about?**
            An oil spill is the release of a liquid petroleum hydrocarbon into the environment, 
            especially into the marine ecosystem, due to human activity, and is a form of pollution. 
            The specific location of the spill, the types of flora and fauna in the area, 
            the volume and nature of the oil, among other factors, can determine the extent of the damage caused by an oil spill. 
            Currently, no solution exists that can preemptively identify the causes of these oil spills and attempt to mitigate them.
            
            **What can this app do?**
            - With data analysis, this app aims to enhance our understanding of the reasons behind oil spills 
            and to contribute to their reduction by suggesting appropriate measures.
            - Users can upload a CSV file containing oil spill data for analysis and generate a comprehensive profile report.
            - The app provides visualizations that reveal critical insights, such as equipment failure being a primary cause of oil spillage.
            - It identifies major contributors to oil spills, like "Teppco Crude Pipeline", and assesses the financial, 
            environmental, and operational impacts.
            - By analyzing loss in comparison with the overall costs, we can quantify our losses and understand the gravity of each spill event.
            - Such analysis not only helps us comprehend our current situation but also assists in preparing 
            appropriate precautions for the future.
            
            **Use Case Example**
            - Predict future 'Oil volume (m3/day)' to plan production using data.
            
            Libraries used:
            - Pandas, NumPy for data handling
            - Scikit-learn for machine learning
            - Streamlit for the web app
            - Plotly for data visualization
        """)



    # Sidebar for file upload
    st.sidebar.title("Upload or Load Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    
    # Markdown section to display information about the dataset
    st.sidebar.markdown("""


    **About this Dataset**
    This database includes a record for each oil pipeline leak or spill reported to the Pipeline and Hazardous Materials Safety Administration since 2010. These records include the incident date and time, operator and pipeline, cause of incident, type of hazardous liquid and quantity lost, injuries and fatalities, and associated costs.
    """)


    


    if st.sidebar.button('Load Sample Data'):
        # Load sample data
        df = pd.read_csv('sample_data.csv')  # Adjust the path if your sample data is in a subdirectory
        st.write("Sample Data Loaded:")
        st.dataframe(df)
        analyze_data(df)  # Call to function that analyzes the data and displays charts
        

    
    elif uploaded_file is not None:
        # User uploads a file, read the file
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV file:")
        st.dataframe(df)
        analyze_data(df)  # Call to function that analyzes the data and displays charts
    else:
        # Placeholder in case no data is loaded yet
        st.info('Please upload a CSV file or load sample data.')
