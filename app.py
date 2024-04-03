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
        # Add your additional simulated data columns here
    }
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Data source selection
data_source = st.sidebar.radio("Select the data source:", ("Upload CSV", "Simulate Live Data"))
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    df = fetch_live_data()

# Display AgGrid for interactive data selection
if not df.empty:
    gob = GridOptionsBuilder.from_dataframe(df)
    gob.configure_pagination()
    gob.configure_side_bar()
    gob.configure_selection('multiple', use_checkbox=True, rowMultiSelectWithClick=True, suppressRowDeselection=False)
    grid_options = gob.build()
    grid_response = AgGrid(df, gridOptions=grid_options, height=300, width='100%', update_mode='MODEL_CHANGED', fit_columns_on_grid_load=True)
    if grid_response['selected_rows']:
        selected_rows = grid_response['selected_rows']
        df = pd.DataFrame(selected_rows)
    # If no rows are selected, df remains unchanged
else:
    st.warning("No data to display. Please select a data source.")

# Continuing with model building and analysis...


    
# If df is not None, meaning data is available for analysis
if df is not None:
    # Exclude the 'Date' column from the dropdown options
    target_variable_options = df.columns[df.columns != 'Date']
    # User selects target variable from the sidebar
    target_variable = st.sidebar.selectbox('Select the target variable to predict:', target_variable_options)


# Sidebar - Set Parameters
with st.sidebar:
    st.header('2. Set Model Hyperparameters')
    parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2, 1)
    parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)
    parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error', 'poisson'])
    parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    sleep_time = st.slider('Sleep time', 0, 3, 0)





# Preprocessing and model training
with st.spinner("Preparing data ..."):
    time.sleep(sleep_time)
    
    # Define features and target
    X = df.drop(target_variable, axis=1)
    y = df[target_variable]
    
    # Ensure all data is numeric for model compatibility
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = pd.to_numeric(y, errors='coerce').fillna(0)
    
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - parameter_split_size) / 100, random_state=parameter_random_state)

    # Display initial dataset
    with st.expander('Initial dataset', expanded=True):
        st.dataframe(df, height=210, use_container_width=True)
        
    # Display train split
    with st.expander('Train split', expanded=False):
        train_col = st.columns((3,1))
        with train_col[0]:
            st.markdown('**X**')
            st.dataframe(X_train, height=210, hide_index=True, use_container_width=True)
        with train_col[1]:
            st.markdown('**y**')
            st.dataframe(y_train, height=210, hide_index=True, use_container_width=True)
            
    # Display test split
    with st.expander('Test split', expanded=False):
        test_col = st.columns((3,1))
        with test_col[0]:
            st.markdown('**X**')
            st.dataframe(X_test, height=210, hide_index=True, use_container_width=True)
        with test_col[1]:
            st.markdown('**y**')
            st.dataframe(y_test, height=210, hide_index=True, use_container_width=True)

    # Training model
    rf = RandomForestRegressor(
        n_estimators=parameter_n_estimators,
        max_features='sqrt',  # Changed from 'auto' to 'sqrt'
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        random_state=parameter_random_state,
        criterion=parameter_criterion,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score
    )
    rf.fit(X_train, y_train)
    
    # Performance metrics
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    rf_results = pd.DataFrame({
        'Metric': ['Training MSE', 'Test MSE', 'Training R2', 'Test R2'],
        'Value': [train_mse, test_mse, train_r2, test_r2]
    })
    
    st.success("Model training and evaluation complete.")


# Define parameter_max_features_metric based on parameter_max_features selection
if parameter_max_features == 'auto':
    parameter_max_features_metric = 'Auto (All features)'
elif parameter_max_features == 'sqrt':
    parameter_max_features_metric = 'Sqrt (Square root of total features)'
elif parameter_max_features == 'log2':
    parameter_max_features_metric = 'Log2 (Log base 2 of total features)'
else:
    # Default or fallback case, you can adjust this as needed
    parameter_max_features_metric = 'Custom setting not directly mapped'



# Now, when displaying the model parameters:
st.header('Model parameters', divider='rainbow')
parameters_col = st.columns(3)
parameters_col[0].metric(label="Data split ratio (% for Training Set)", value=parameter_split_size, delta="")
parameters_col[1].metric(label="Number of estimators (n_estimators)", value=parameter_n_estimators, delta="")
# Corrected to use the defined metric variable
parameters_col[2].metric(label="Max features (max_features)", value=parameter_max_features_metric, delta="")

# Display feature importance plot
importances = rf.feature_importances_
feature_names = list(X.columns)
forest_importances = pd.Series(importances, index=feature_names)
df_importance = forest_importances.reset_index().rename(columns={'index': 'feature', 0: 'value'})

bars = alt.Chart(df_importance).mark_bar(size=40).encode(
            x='value:Q',
            y=alt.Y('feature:N', sort='-x')
        ).properties(height=250)

performance_col = st.columns((2, 0.2, 3))
with performance_col[0]:
    st.header('Model performance', divider='rainbow')
    st.dataframe(rf_results.T.reset_index().rename(columns={'index': 'Parameter', 0: 'Value'}))
with performance_col[2]:
    st.header('Feature importance', divider='rainbow')
    st.altair_chart(bars, theme='streamlit', use_container_width=True)

# Prediction results
st.header('Prediction results', divider='rainbow')
s_y_train = pd.Series(y_train, name='actual').reset_index(drop=True)
s_y_train_pred = pd.Series(y_train_pred, name='predicted').reset_index(drop=True)
df_train = pd.DataFrame(data=[s_y_train, s_y_train_pred], index=None).T
df_train['class'] = 'train'
    
s_y_test = pd.Series(y_test, name='actual').reset_index(drop=True)
s_y_test_pred = pd.Series(y_test_pred, name='predicted').reset_index(drop=True)
df_test = pd.DataFrame(data=[s_y_test, s_y_test_pred], index=None).T
df_test['class'] = 'test'

df_prediction = pd.concat([df_train, df_test], axis=0)

prediction_col = st.columns((2, 0.2, 3))

# Display dataframe
with prediction_col[0]:
    st.dataframe(df_prediction, height=320, use_container_width=True)

# Display scatter plot of actual vs predicted values
with prediction_col[2]:
    scatter = alt.Chart(df_prediction).mark_circle(size=60).encode(
                    x='actual',
                    y='predicted',
                    color='class'
                )
    st.altair_chart(scatter, theme='streamlit', use_container_width=True)

# Actual vs Predicted line trend chart
if 'df' in locals() or 'df' in globals():
    # It's safe to use df here
    st.header('Actual vs Predicted Line Trend', divider='rainbow')

    # Assuming 'Date' is the meaningful variable you want on the x-axis
    # and it is present in your dataframe
    if 'Date' in df_prediction:
        x_axis = 'Date'
    else:
        x_axis = df_prediction.columns[0]  # default to the first column

    # Create line chart for actual values
    actual_line = alt.Chart(df_prediction).mark_line(color='orange').encode(
        x=alt.X(f'{x_axis}:N', axis=alt.Axis(labelAngle=-45, title='Date')),  # Set the axis title to 'Date'
        y=alt.Y('actual:Q', title='Actual'),
        tooltip=[x_axis, 'actual']
    ).interactive()

    # Create line chart for predicted values
    predicted_line = alt.Chart(df_prediction).mark_line(color='blue').encode(
        x=alt.X(f'{x_axis}:N', axis=alt.Axis(labelAngle=-45, title='Date')),  # Set the axis title to 'Date'
        y=alt.Y('predicted:Q', title='Predicted'),
        tooltip=[x_axis, 'predicted']
    ).interactive()

    # Combine the actual and predicted lines
    combined_chart = alt.layer(actual_line, predicted_line).resolve_scale(
        y='shared'
    ).properties(
        width=700,  # Set width
        height=400  # Set height
    )

    st.altair_chart(combined_chart, use_container_width=True)
