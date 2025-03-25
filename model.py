import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import re
import sqlparse

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="DB Index Optimizer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to parse SQL query and extract features
def extract_features_from_query(query_text):
    try:
        parsed = sqlparse.parse(query_text)[0]
    except Exception as e:
        st.error(f"Error parsing query: {e}")
        return None

    # Initialize features dictionary with binary flags for each SQL clause
    features = {
        'select': 0, 'insert': 0, 'update': 0, 'delete': 0, 'create': 0, 
        'drop': 0, 'alter': 0, 'join': 0, 'where': 0, 'group by': 0, 
        'order by': 0, 'having': 0, 'limit': 0, 'between': 0, 'like': 0
    }
    
    # Extract query type (e.g., SELECT, INSERT)
    query_type = parsed.get_type().lower()
    if query_type in features:
        features[query_type] = 1
    
    # Convert query to lowercase string for regex matching
    query_str = str(parsed).lower()
    
    # Check for specific SQL clauses
    if ' join ' in query_str:
        features['join'] = 1
    if ' where ' in query_str:
        features['where'] = 1
    if ' group by ' in query_str:
        features['group by'] = 1
    if ' order by ' in query_str:
        # Count occurrences of 'order by'
        features['order by'] = len(re.findall(r'order by', query_str))
    if ' having ' in query_str:
        features['having'] = 1
    if ' limit ' in query_str:
        features['limit'] = 1
    if ' between ' in query_str:
        features['between'] = 1
    if ' like ' in query_str:
        features['like'] = 1
    
    return features

# Function to preprocess data and train the model
def train_model(df):
    # Remove empty rows
    df = df.dropna(how='all')
    
    # Get best performing index per query (lowest execution time)
    best_indices = df.loc[df.groupby('query_name')['execution_time'].idxmin()]
    
    # Features: SQL clause flags
    feature_cols = ['select', 'insert', 'update', 'delete', 'create', 'drop', 'alter', 
                    'join', 'where', 'group by', 'order by', 'having', 'limit', 'between', 'like']
    features = best_indices[feature_cols]
    
    # Target: index_type
    target = best_indices['index_type']
    
    # Encode target variable
    le = LabelEncoder()
    target_encoded = le.fit_transform(target)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)
    train_data_size = X_train.shape  # Capture training data size
    
    # Train a RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate performance on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross validation scores
    cv_scores = cross_val_score(model, features, target_encoded, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Get unique index descriptions
    index_descriptions = best_indices[['index_type', 'index_description']].drop_duplicates().set_index('index_type')
    
    # Also return confusion matrix on test data
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return (model, le, X_test, y_test, accuracy, cv_mean, cv_std, 
            feature_importance, index_descriptions, train_data_size, conf_matrix)

# Function to display index performance metrics and plot comparisons
def display_index_performance(df, index_type):
    best_indices = df.loc[df.groupby('query_name')['execution_time'].idxmin()]
    index_data = best_indices[best_indices['index_type'] == index_type]
    
    avg_exec_time = index_data['execution_time'].mean()
    avg_memory_change = index_data['memory_change'].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Avg. Execution Time (s)", f"{avg_exec_time:.4f}")
    with col2:
        st.metric("Avg. Memory Change (MB)", f"{avg_memory_change:.2f}")
    
    st.subheader("Performance Comparison")
    index_performance = best_indices.groupby('index_type')['execution_time'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4' if x != index_type else '#ff7f0e' for x in index_performance['index_type']]
    ax.bar(index_performance['index_type'], index_performance['execution_time'], color=colors)
    ax.set_ylabel('Avg. Execution Time (s)')
    ax.set_title('Performance by Index Type')
    for i, v in enumerate(index_performance['execution_time']):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
    st.pyplot(fig)

# Function to display feature importance chart
def display_feature_importance(feature_importance):
    fig, ax = plt.subplots(figsize=(10, 8))
    feature_importance = feature_importance.sort_values('Importance')
    ax.barh(feature_importance['Feature'], feature_importance['Importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance in Index Selection')
    st.pyplot(fig)

# Function to display confusion matrix
def display_confusion_matrix(conf_matrix, le):
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = le.classes_
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# Main Streamlit app
def main():
    st.title("🔍 Database Index Optimization Advisor")
    st.write("Upload your performance data and get index recommendations based on SQL query characteristics.")
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.label_encoder = None
        st.session_state.df = None
        st.session_state.index_descriptions = None
        st.session_state.feature_importance = None
        st.session_state.train_data_size = None
        st.session_state.conf_matrix = None
    
    tab1, tab2, tab3 = st.tabs(["Upload & Train", "Query Analyzer", "Model Insights"])
    
    # Tab 1: Upload and Train
    with tab1:
        st.header("Upload Performance Data")
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    (model, le, X_test, y_test, accuracy, cv_mean, cv_std, 
                     feature_importance, index_descriptions, train_data_size, conf_matrix) = train_model(df)
                    
                    st.session_state.model = model
                    st.session_state.label_encoder = le
                    st.session_state.index_descriptions = index_descriptions
                    st.session_state.feature_importance = feature_importance
                    st.session_state.train_data_size = train_data_size
                    st.session_state.conf_matrix = conf_matrix
                    
                    st.success("Model trained successfully!")
                    st.write(f"Test Accuracy: {accuracy:.2%}")
                    st.write(f"Cross Validation Score: {cv_mean:.2%} (+/- {cv_std*2:.2%})")
                    st.write(f"Training Data Size: {train_data_size[0]} rows, {train_data_size[1]} features")
    
    # Tab 2: Query Analyzer
    with tab2:
        st.header("Query Analyzer")
        if st.session_state.model is None:
            st.warning("Please upload data and train the model in the 'Upload & Train' tab first.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Option 1: Enter SQL Query")
                query_text = st.text_area("Enter your SQL query:", height=150, 
                                          placeholder="SELECT * FROM users WHERE age BETWEEN 18 AND 30 ORDER BY last_login DESC")
                if st.button("Analyze Query", key="analyze_query"):
                    if query_text:
                        features = extract_features_from_query(query_text)
                        if features is None:
                            st.error("Failed to extract features.")
                        else:
                            input_df = pd.DataFrame([features])
                            prediction = st.session_state.model.predict(input_df)
                            index_type = st.session_state.label_encoder.inverse_transform(prediction)[0]
                            probabilities = st.session_state.model.predict_proba(input_df)[0]
                            confidence = probabilities.max() * 100
                            
                            st.success(f"Recommended index type: **{index_type}** (Confidence: {confidence:.1f}%)")
                            if index_type in st.session_state.index_descriptions.index:
                                description = st.session_state.index_descriptions.loc[index_type, 'index_description']
                                st.info(f"**Description:** {description}")
                            
                            st.subheader("Extracted Query Features")
                            features_df = pd.DataFrame([features])
                            non_zero_features = features_df.loc[:, (features_df != 0).any()]
                            st.dataframe(non_zero_features)
                            
                            if st.session_state.df is not None:
                                st.subheader("Performance Metrics")
                                display_index_performance(st.session_state.df, index_type)
                    else:
                        st.error("Please enter a SQL query.")
            
            with col2:
                st.subheader("Option 2: Select Query Characteristics")
                st.write("Manually select the SQL clauses used in your query:")
                c1, c2 = st.columns(2)
                with c1:
                    select = st.checkbox("SELECT", value=True)
                    insert = st.checkbox("INSERT")
                    update = st.checkbox("UPDATE")
                    delete = st.checkbox("DELETE")
                    create = st.checkbox("CREATE")
                    drop = st.checkbox("DROP")
                    alter = st.checkbox("ALTER")
                    join = st.checkbox("JOIN")
                with c2:
                    where = st.checkbox("WHERE")
                    group_by = st.checkbox("GROUP BY")
                    order_by = st.checkbox("ORDER BY")
                    having = st.checkbox("HAVING")
                    limit = st.checkbox("LIMIT")
                    between = st.checkbox("BETWEEN")
                    like = st.checkbox("LIKE")
                
                order_by_count = st.number_input("Number of ORDER BY clauses:", min_value=1, max_value=5, value=1) if order_by else 0
                
                if st.button("Get Recommendation", key="manual_recommend"):
                    features = {
                        'select': int(select),
                        'insert': int(insert),
                        'update': int(update),
                        'delete': int(delete),
                        'create': int(create),
                        'drop': int(drop),
                        'alter': int(alter),
                        'join': int(join),
                        'where': int(where),
                        'group by': int(group_by),
                        'order by': order_by_count,
                        'having': int(having),
                        'limit': int(limit),
                        'between': int(between),
                        'like': int(like)
                    }
                    input_df = pd.DataFrame([features])
                    prediction = st.session_state.model.predict(input_df)
                    index_type = st.session_state.label_encoder.inverse_transform(prediction)[0]
                    probabilities = st.session_state.model.predict_proba(input_df)[0]
                    confidence = probabilities.max() * 100
                    
                    st.success(f"Recommended index type: **{index_type}** (Confidence: {confidence:.1f}%)")
                    
                    if index_type in st.session_state.index_descriptions.index:
                        description = st.session_state.index_descriptions.loc[index_type, 'index_description']
                        st.info(f"**Description:** {description}")
                    
                    if st.session_state.df is not None:
                        st.subheader("Performance Metrics")
                        display_index_performance(st.session_state.df, index_type)
    
    # Tab 3: Model Insights
    with tab3:
        st.header("Model Insights")
        if st.session_state.model is None:
            st.warning("Please upload data and train the model in the 'Upload & Train' tab first.")
        else:
            st.subheader("Feature Importance")
            st.write("This shows which query features most influence index recommendations:")
            display_feature_importance(st.session_state.feature_importance)
            
            st.subheader("Training Performance")
            st.write(f"Training Data Size: {st.session_state.train_data_size[0]} rows, {st.session_state.train_data_size[1]} features")
            
            st.write("Confusion Matrix on Test Data:")
            display_confusion_matrix(st.session_state.conf_matrix, st.session_state.label_encoder)
            
            st.subheader("Index Types & Descriptions")
            for idx_type, row in st.session_state.index_descriptions.iterrows():
                with st.expander(f"{idx_type.upper()} Index"):
                    st.write(row['index_description'])
                    best_indices = st.session_state.df.loc[st.session_state.df.groupby('query_name')['execution_time'].idxmin()]
                    index_data = best_indices[best_indices['index_type'] == idx_type]
                    if not index_data.empty:
                        st.write(f"Average execution time: {index_data['execution_time'].mean():.4f}s")
                        if 'index_statement' in index_data.columns:
                            st.code(index_data['index_statement'].iloc[0], language='sql')

if __name__ == "__main__":
    main()
