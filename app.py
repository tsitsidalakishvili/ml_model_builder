
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile
import datetime
import json
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import json

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

# Initialize an empty DataFrame
df = pd.DataFrame()

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
    if st.button('Generate Live Data'):
        df = fetch_live_data()
        st.write(df)

# Model training button
if not df.empty:
    if st.button('Train Model'):
        # Sidebar - Set Parameters
        with st.sidebar:
            st.header('Model Parameters')
            parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
            parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 100, 1000, 100, 100)
            parameter_max_features = st.select_slider('Max features (max_features)', options=['sqrt', 'log2'], value='sqrt')
            parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2, 1)
            parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)
            parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)
            parameter_criterion = st.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error', 'poisson'])
            parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
            parameter_oob_score = st.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])

        # Preprocessing and model training
        with st.spinner("Preparing data ..."):
            time.sleep(1)  # Simulate data processing time
            
            # Assume first column is the target variable for simplicity
            target_variable = df.columns[0]
            X = df.loc[:, df.columns != target_variable]
            y = df[target_variable]
            
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
            y = pd.to_numeric(y, errors='coerce').fillna(0)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - parameter_split_size) / 100, random_state=parameter_random_state)
            
            rf = RandomForestRegressor(
                n_estimators=parameter_n_estimators,
                max_features=parameter_max_features,
                min_samples_split=parameter_min_samples_split,
                min_samples_leaf=parameter_min_samples_leaf,
                random_state=parameter_random_state,
                criterion=parameter_criterion,
                bootstrap=parameter_bootstrap,
                oob_score=parameter_oob_score
            )
            rf.fit(X_train, y_train)
            
            # Performance metrics
            train_mse = mean_squared_error(y_train, rf.predict(X_train))
            test_mse = mean_squared_error(y_test, rf.predict(X_test))
            train_r2 = r2_score(y_train, rf.predict(X_train))
            test_r2 = r2_score(y_test, rf.predict(X_test))
            
            # Display model performance
            st.success("Model training and evaluation complete.")
            st.write(f"Training MSE: {train_mse}, Test MSE: {test_mse}")
            st.write(f"Training R^2: {train_r2}, Test R^2: {test_r2}")


