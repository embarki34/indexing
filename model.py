import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sqlparse

st.set_page_config(
    page_title="DB Index Optimizer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to parse SQL query and extract features
def extract_features_from_query(query_text):
    # Parse the SQL query
    parsed = sqlparse.parse(query_text)[0]
    
    # Initialize features
    features = {
        'select': 0, 'insert': 0, 'update': 0, 'delete': 0, 'create': 0, 
        'drop': 0, 'alter': 0, 'join': 0, 'where': 0, 'group by': 0, 
        'order by': 0, 'having': 0, 'limit': 0, 'between': 0, 'like': 0
    }
    
    # Extract query type (SELECT, INSERT, etc.)
    query_type = parsed.get_type().lower()
    if query_type in features:
        features[query_type] = 1
    
    # Convert to string for regex patterns
    query_str = str(parsed).lower()
    
    # Check for various SQL clauses
    if ' join ' in query_str:
        features['join'] = 1
    if ' where ' in query_str:
        features['where'] = 1
    if ' group by ' in query_str:
        features['group by'] = 1
    if ' order by ' in query_str:
        features['order by'] = 1
    if ' having ' in query_str:
        features['having'] = 1
    if ' limit ' in query_str:
        features['limit'] = 1
    if ' between ' in query_str:
        features['between'] = 1
    if ' like ' in query_str:
        features['like'] = 1
    
    # Count the number of ORDER BY clauses
    if features['order by'] == 1:
        features['order by'] = len(re.findall(r'order by', query_str))
    
    return features

# Function to preprocess data and train the model
def train_model(df):
    # Clean data: Remove empty rows
    df = df.dropna(how='all')
    
    # Get only the best performing index for each query
    best_indices = df.loc[df.groupby('query_name')['execution_time'].idxmin()]
    
    # Features: SQL clauses (binary flags)
    features = best_indices[['select', 'insert', 'update', 'delete', 'create', 'drop', 'alter', 
                           'join', 'where', 'group by', 'order by', 'having', 'limit', 'between', 'like']]
    
    # Target: index_type
    target = best_indices['index_type']
    
    # Encode target variable
    le = LabelEncoder()
    target_encoded = le.fit_transform(target)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate model performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Perform cross validation
    cv_scores = cross_val_score(model, features, target_encoded, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Get index descriptions
    index_descriptions = best_indices[['index_type', 'index_description']].drop_duplicates().set_index('index_type')
    
    return model, le, X_test, y_test, accuracy, cv_mean, cv_std, feature_importance, index_descriptions

# Function to display index performance metrics
def display_index_performance(df, index_type):
    # Get only best performing indices
    best_indices = df.loc[df.groupby('query_name')['execution_time'].idxmin()]
    
    # Filter data for the selected index type
    index_data = best_indices[best_indices['index_type'] == index_type]
    
    # Calculate average performance metrics
    avg_exec_time = index_data['execution_time'].mean()
    avg_memory_change = index_data['memory_change'].mean()
    
    # Create columns for metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Avg. Execution Time (s)", f"{avg_exec_time:.4f}")
    with col2:
        st.metric("Avg. Memory Change (MB)", f"{avg_memory_change:.2f}")
    
    # Create a plot comparing this index with others
    st.subheader("Performance Comparison")
    
    # Calculate average performance by index type
    index_performance = best_indices.groupby('index_type')['execution_time'].mean().reset_index()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar colors (highlight the selected index)
    colors = ['#1f77b4' if x != index_type else '#ff7f0e' for x in index_performance['index_type']]
    
    # Create bar chart
    ax.bar(index_performance['index_type'], index_performance['execution_time'], color=colors)
    ax.set_ylabel('Avg. Execution Time (s)')
    ax.set_title('Performance by Index Type')
    
    # Add labels with values
    for i, v in enumerate(index_performance['execution_time']):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    st.pyplot(fig)

def display_feature_importance(feature_importance):
    # Create a horizontal bar chart of feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance')
    
    # Create bar chart
    ax.barh(feature_importance['Feature'], feature_importance['Importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance in Index Selection')
    
    st.pyplot(fig)

# Main Streamlit app
def main():
    st.title("🔍 Database Index Optimization Advisor")
    st.write("Upload your database performance data and get recommendations for the best index type based on query characteristics.")
    
    # Initialize session state for model and data
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.label_encoder = None
        st.session_state.df = None
        st.session_state.index_descriptions = None
        st.session_state.feature_importance = None
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Upload & Train", "Query Analyzer", "Model Insights"])
    
    # Tab 1: Upload and train model
    with tab1:
        st.header("Upload Performance Data")
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        
        if uploaded_file is not None:
            # Read the uploaded CSV
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            # Display a preview of the data
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Train button
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    model, le, X_test, y_test, accuracy, cv_mean, cv_std, feature_importance, index_descriptions = train_model(df)
                    
                    # Store in session state
                    st.session_state.model = model
                    st.session_state.label_encoder = le
                    st.session_state.index_descriptions = index_descriptions
                    st.session_state.feature_importance = feature_importance
                    
                    # Display success message with model accuracy and cross validation results
                    st.success(f"Model trained successfully!")
                    st.write(f"Test Accuracy: {accuracy:.2%}")
                    st.write(f"Cross Validation Score: {cv_mean:.2%} (+/- {cv_std*2:.2%})")
    
    # Tab 2: Query Analyzer
    with tab2:
        st.header("Query Analyzer")
        
        # Check if model is trained
        if st.session_state.model is None:
            st.warning("Please upload data and train the model in the 'Upload & Train' tab first.")
        else:
            # Create two columns for input methods
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Option 1: Enter SQL Query")
                query_text = st.text_area("Enter your SQL query:", height=150, 
                                          placeholder="SELECT * FROM users WHERE age BETWEEN 18 AND 30 ORDER BY last_login DESC")
                
                if st.button("Analyze Query"):
                    if query_text:
                        # Extract features from the query
                        features = extract_features_from_query(query_text)
                        
                        # Convert to DataFrame
                        input_df = pd.DataFrame([features])
                        
                        # Make prediction
                        prediction = st.session_state.model.predict(input_df)
                        index_type = st.session_state.label_encoder.inverse_transform(prediction)[0]
                        
                        # Get confidence scores
                        probabilities = st.session_state.model.predict_proba(input_df)[0]
                        confidence = probabilities.max() * 100
                        
                        # Display results
                        st.success(f"Recommended index type: **{index_type}** (Confidence: {confidence:.1f}%)")
                        
                        # Get description
                        if index_type in st.session_state.index_descriptions.index:
                            description = st.session_state.index_descriptions.loc[index_type, 'index_description']
                            st.info(f"**Description:** {description}")
                        
                        # Display extracted features
                        st.subheader("Query Features")
                        st.write("These are the features extracted from your query:")
                        
                        # Convert dictionary to DataFrame for display
                        features_df = pd.DataFrame([features])
                        
                        # Only show non-zero features for cleaner display
                        non_zero_features = features_df.loc[:, (features_df != 0).any()]
                        st.dataframe(non_zero_features)
                        
                        # If we have the original dataset, show performance metrics
                        if st.session_state.df is not None:
                            st.subheader("Performance Metrics")
                            display_index_performance(st.session_state.df, index_type)
                    else:
                        st.error("Please enter a SQL query.")
            
            with col2:
                st.subheader("Option 2: Select Query Characteristics")
                
                # Create checkboxes for each feature
                st.write("Select SQL clauses used in your query:")
                
                # Two columns for better space usage
                c1, c2 = st.columns(2)
                
                # First column of checkboxes
                with c1:
                    select = st.checkbox("SELECT", value=True)
                    insert = st.checkbox("INSERT")
                    update = st.checkbox("UPDATE")
                    delete = st.checkbox("DELETE")
                    create = st.checkbox("CREATE")
                    drop = st.checkbox("DROP")
                    alter = st.checkbox("ALTER")
                    join = st.checkbox("JOIN")
                
                # Second column of checkboxes
                with c2:
                    where = st.checkbox("WHERE")
                    group_by = st.checkbox("GROUP BY")
                    order_by = st.checkbox("ORDER BY")
                    having = st.checkbox("HAVING")
                    limit = st.checkbox("LIMIT")
                    between = st.checkbox("BETWEEN")
                    like = st.checkbox("LIKE")
                
                # If ORDER BY is selected, ask how many
                order_by_count = 1
                if order_by:
                    order_by_count = st.number_input("Number of ORDER BY clauses:", min_value=1, max_value=5, value=1)
                
                if st.button("Get Recommendation"):
                    # Create features dictionary
                    features = {
                        'select': 1 if select else 0,
                        'insert': 1 if insert else 0,
                        'update': 1 if update else 0,
                        'delete': 1 if delete else 0,
                        'create': 1 if create else 0,
                        'drop': 1 if drop else 0,
                        'alter': 1 if alter else 0,
                        'join': 1 if join else 0,
                        'where': 1 if where else 0,
                        'group by': 1 if group_by else 0,
                        'order by': order_by_count if order_by else 0,
                        'having': 1 if having else 0,
                        'limit': 1 if limit else 0,
                        'between': 1 if between else 0,
                        'like': 1 if like else 0
                    }
                    
                    # Convert to DataFrame
                    input_df = pd.DataFrame([features])
                    
                    # Make prediction
                    prediction = st.session_state.model.predict(input_df)
                    index_type = st.session_state.label_encoder.inverse_transform(prediction)[0]
                    
                    # Get confidence scores
                    probabilities = st.session_state.model.predict_proba(input_df)[0]
                    confidence = probabilities.max() * 100
                    
                    # Display results
                    st.success(f"Recommended index type: **{index_type}** (Confidence: {confidence:.1f}%)")
                    
                    # Get description
                    if index_type in st.session_state.index_descriptions.index:
                        description = st.session_state.index_descriptions.loc[index_type, 'index_description']
                        st.info(f"**Description:** {description}")
                    
                    # If we have the original dataset, show performance metrics
                    if st.session_state.df is not None:
                        st.subheader("Performance Metrics")
                        display_index_performance(st.session_state.df, index_type)
    
    # Tab 3: Model Insights
    with tab3:
        st.header("Model Insights")
        
        if st.session_state.model is None:
            st.warning("Please upload data and train the model in the 'Upload & Train' tab first.")
        else:
            # Show feature importance
            st.subheader("Feature Importance")
            st.write("This shows which query characteristics most influence the index selection:")
            display_feature_importance(st.session_state.feature_importance)
            
            # Index type descriptions
            st.subheader("Index Types")
            
            # Display each index type with its description
            for index_type, row in st.session_state.index_descriptions.iterrows():
                with st.expander(f"{index_type.upper()} Index"):
                    st.write(row['index_description'])
                    
                    # Show average performance if we have the data
                    if st.session_state.df is not None:
                        # Get only best performing indices
                        best_indices = st.session_state.df.loc[st.session_state.df.groupby('query_name')['execution_time'].idxmin()]
                        index_data = best_indices[best_indices['index_type'] == index_type]
                        if not index_data.empty:
                            st.write(f"Average execution time: {index_data['execution_time'].mean():.4f}s")
                            
                            # Get a sample index statement
                            if 'index_statement' in index_data.columns:
                                sample_statement = index_data['index_statement'].iloc[0]
                                st.code(sample_statement, language='sql')

if __name__ == "__main__":
    main()