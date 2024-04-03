import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
from st_aggrid import AgGrid, GridOptionsBuilder

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
        - Altair for visualization
        - Streamlit for the web app
    """)

# Function to simulate live data fetching
def fetch_live_data():
    np.random.seed(0)
    data = {
        'Oil volume (m3/day)': np.random.uniform(low=45, high=55, size=100),
        'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'Water volume (m3/day)': np.random.uniform(low=15, high=25, size=100),
        'Gas volume (m3/day)': np.random.uniform(low=12800, high=13300, size=100),
        'Water cut (%)': np.random.uniform(low=25, high=35, size=100),
        'Working hours': np.random.uniform(low=22, high=26, size=100),
        'Dynamic level (m)': np.random.uniform(low=1750, high=1850, size=100),
        'Reservoir pressure (atm)': np.random.uniform(low=210, high=220, size=100),
    }
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Sidebar - Data source selection
data_source = st.sidebar.radio("Select the data source:", ("Upload CSV", "Simulate Live Data"))
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.DataFrame()
else:
    df = fetch_live_data()

# Display AgGrid for interactive data selection immediately after the "About this app" section
if not df.empty:
    gob = GridOptionsBuilder.from_dataframe(df)
    gob.configure_pagination()
    gob.configure_side_bar()
    gob.configure_selection('multiple', use_checkbox=True, rowMultiSelectWithClick=True, suppressRowDeselection=False)
    grid_options = gob.build()
    grid_response = AgGrid(df, gridOptions=grid_options, height=300, width='100%', update_mode='MODEL_CHANGED', fit_columns_on_grid_load=True)
    selected_rows = grid_response['selected_rows']
    df = pd.DataFrame(selected_rows) if selected_rows else df
else:
    st.warning("No data to display. Please select a data source.")

# Continuation only if df is not empty to avoid errors
if not df.empty:
    # Exclude 'Date' from target variable options
    target_variable_options = df.columns[df.columns != 'Date'].tolist()
    target_variable = st.sidebar.selectbox('Select the target variable to predict:', target_variable_options, index=0) if target_variable_options else None

    if target_variable:
        # Set model parameters
        with st.sidebar:
            st.header('2. Set Model Hyperparameters')
            parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
            parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 100, 1000, 100, 100)
            parameter_max_features = st.selectbox('Max features (max_features)', ['auto', 'sqrt', 'log2'])
            parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2)
            parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2)
            parameter_random_state = st.slider('Seed number (random_state)', 0, 100, 42)
            parameter_criterion = st.selectbox('Performance measure (criterion)', ['squared_error', 'absolute_error', 'poisson'])
            parameter_bootstrap = st.selectbox('Bootstrap samples when building trees (bootstrap)', [True, False])
            parameter_oob_score = st.selectbox('Use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', [False, True])
            sleep_time = st.slider('Sleep time', 0, 3, 1)

        # Preprocessing and Model Training
        st.spinner("Preparing data...")
        time.sleep(sleep_time)

        X = df.drop(columns=[target_variable, 'Date'], errors='ignore')
        y = df[target_variable].astype(float)

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

        # Model Evaluation
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        rf_results = pd.DataFrame({
            'Metric': ['Training MSE', 'Test MSE', 'Training R^2', 'Test R^2'],
            'Value': [train_mse, test_mse, train_r2, test_r2]
        })

        st.success("Model training and evaluation complete.")

        # Display Model Parameters and Results
        st.header('Model Parameters and Performance')
        st.write(rf_results)

        # Feature Importance Plot
        feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.bar_chart(feature_importance)

        # Prediction Results Visualization
        st.header('Prediction Results')
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
        st.write(results_df)

else:
    st.error("Error: The data is not available or not properly loaded.")
