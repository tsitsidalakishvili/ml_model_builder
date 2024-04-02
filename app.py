import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import json
import time
import zipfile

# Set page config
st.set_page_config(page_title='ML Model Building', page_icon='🤖')
st.title('🤖 ML Model Building with Live Data')

# Provide information about the app
with st.expander('About this app'):
    st.write("""
    **What can this app do?**
    - Build a machine learning model to predict various outcomes based on oil well operation parameters.
    - The user can either upload a CSV file or simulate live data.
    
    **Use Case Example**
    - Predict future 'Oil volume (m3/day)' to plan production using data.
    
    Libraries used:
    - Pandas, NumPy for data handling
    - Scikit-learn for machine learning
    - Streamlit for the web app
    """)

# Sidebar for model parameters
with st.sidebar:
    st.header("Model Parameters")
    # Model parameters
    parameter_n_estimators = st.number_input('Number of estimators (n_estimators)', min_value=10, max_value=1000, value=100, step=10)
    parameter_max_features = st.selectbox('Max features (max_features)', ['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2, 1)
    parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)
    parameter_bootstrap = st.selectbox('Bootstrap samples when building trees (bootstrap)', [True, False])
    parameter_random_state = 42  # Fixed to ensure reproducibility

# Function to simulate live data fetching
def fetch_live_data():
    np.random.seed(0)  # Set the random seed for reproducibility
    
    # Simulate loading data
    data = {
        'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'Oil volume (m3/day)': np.random.uniform(low=45, high=55, size=100),
        'Water volume (m3/day)': np.random.uniform(low=15, high=25, size=100),
        # Additional simulated columns
        'Volume of liquid (m3/day)': np.random.uniform(low=65, high=75, size=100),
        'Gas volume (m3/day)': np.random.uniform(low=12800, high=13300, size=100),
        'Water cut (%)': np.random.uniform(low=25, high=35, size=100),
        'Working hours': np.random.uniform(low=22, high=26, size=100),
        'Dynamic level (m)': np.random.uniform(low=1750, high=1850, size=100),
        'Reservoir pressure (atm)': np.random.uniform(low=210, high=220, size=100),
    }
    return pd.DataFrame(data)

df = pd.DataFrame()  # Initialize an empty DataFrame for later checks

# Tabs for CSV Upload and Live Data
tab1, tab2 = st.tabs(["CSV Upload", "Live Data"])

# Tab1: CSV Upload
with tab1:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)

# Tab2: Live Data
with tab2:
    df = fetch_live_data() # Always fetch new live data when tab is active
    st.write(df)





# ... your existing imports and setup ...
# ... your existing code ...
# ..# ... your existing imports and setup ...
# ... your existing code ...
# ... your existing imports and setup ...
# ... your existing code ...

# Check if data is loaded
if not df.empty:
    # Convert 'Date' to datetime and extract features
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    # Drop the 'Date' column after extracting features
    df.drop(columns=['Date'], inplace=True)

    # Allow the user to select the target variable from the remaining columns
    target_variable = st.selectbox("Select the target variable to predict:", options=df.columns)

    # Train model button
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            # Prepare the data for training
            X = df.drop(columns=[target_variable])
            y = df[target_variable]

            # Convert all data to numeric type, handling non-numeric data and missing values
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
            y = pd.to_numeric(y, errors='coerce').fillna(0)

            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize and fit the RandomForestRegressor model with the provided parameters
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Prediction results section
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            df_train = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred, 'Set': 'Train'})
            df_test = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred, 'Set': 'Test'})

            # Combine train and test results
            df_combined = pd.concat([df_train, df_test])

            # Display additional training results
            st.header('Dataset Information', divider='rainbow')

            # Display dataset information
            if 'X_train' in locals() or 'X_train' in globals():  # Check if data variables are defined
                cols = st.columns(4)
                cols[0].metric(label="No. of samples", value=X.shape[0])
                cols[1].metric(label="No. of X variables", value=X.shape[1])
                cols[2].metric(label="No. of Training samples", value=X_train.shape[0])
                cols[3].metric(label="No. of Test samples", value=X_test.shape[0])
            else:
                st.warning('👈 Upload a CSV file or click "Load example data" to get started!')

            # Display initial dataset
            with st.expander('Initial Dataset', expanded=True):
                st.dataframe(df, height=210, use_container_width=True)

            # Display train-test splits
            with st.expander('Train-Test Splits', expanded=False):
                train_col = st.columns((3, 1))
                with train_col[0]:
                    st.markdown('**X_train**')
                    st.dataframe(X_train, height=210, hide_index=True, use_container_width=True)
                with train_col[1]:
                    st.markdown('**y_train**')
                    st.dataframe(y_train, height=210, hide_index=True, use_container_width=True)

                test_col = st.columns((3, 1))
                with test_col[0]:
                    st.markdown('**X_test**')
                    st.dataframe(X_test, height=210, hide_index=True, use_container_width=True)
                with test_col[1]:
                    st.markdown('**y_test**')
                    st.dataframe(y_test, height=210, hide_index=True, use_container_width=True)

            # Zip dataset files and provide download button
            df.to_csv('dataset.csv', index=False)
            X_train.to_csv('X_train.csv', index=False)
            y_train.to_csv('y_train.csv', index=False)
            X_test.to_csv('X_test.csv', index=False)
            y_test.to_csv('y_test.csv', index=False)

            list_files = ['dataset.csv', 'X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']
            with zipfile.ZipFile('dataset.zip', 'w') as zipF:
                for file in list_files:
                    zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

            with open('dataset.zip', 'rb') as datazip:
                btn = st.download_button(
                        label='Download ZIP',
                        data=datazip,
                        file_name="dataset.zip",
                        mime="application/octet-stream"
                        )

            # Visualizing the prediction results
            st.header("Prediction Results and Feature Importance",  divider='rainbow')

            # Use st.columns to place charts side by side
            col1, col2 = st.columns(2)

            # Prediction Results chart
            with col1:
                scatter = alt.Chart(df_combined).mark_circle().encode(
                                x='Actual',
                                y='Predicted',
                                color='Set',
                                tooltip=['Actual', 'Predicted']
                            ).properties(
                                width=300,
                                height=400
                            )
                st.altair_chart(scatter, use_container_width=True)

            # Calculate feature importance
            importances = model.feature_importances_
            feature_names = X.columns
            df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

            # Feature Importance chart
            with col2:
                bars = alt.Chart(df_importance).mark_bar(size=40).encode(
                            x='Importance',
                            y=alt.Y('Feature', sort='-x')
                        ).properties(height=400)
                st.altair_chart(bars, use_container_width=True)
