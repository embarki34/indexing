import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import re
# Removed sqlparse as it's not needed for training anymore
# import sqlparse
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

# Define the features based on the CSV header (excluding 'quiri', 'target')
# Make sure this list exactly matches the keywords corresponding to your _count columns
sql_features_list = [
    'select', 'insert', 'update', 'delete', 'create', 'drop', 'alter',
    'join', 'where', 'group by', 'order by', 'having', 'limit',
    'between', 'in', 'like', 'from', 'on', 'as', 'desc', 'count', 'sum', 'avg', 'distinct'
]

# Function to parse NEW SQL query and extract features matching the training data format
def extract_features_from_query(query_text, keywords):
    """Counts occurrences of SQL keywords/clauses in a query string."""
    features = {}
    if not query_text:
        # Return dictionary with all features set to 0 if query is empty
        for keyword in keywords:
             features[f"{keyword.replace(' ', '_')}_count"] = 0
        return features

    # Convert query to lowercase string for regex matching
    query_lower = query_text.lower()

    for keyword in keywords:
        # Use word boundaries (\b) to match whole words/phrases
        # Escape special regex characters in the keyword (like *)
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        try:
            count = len(re.findall(pattern, query_lower))
        except re.error as e:
            st.error(f"Regex error for keyword '{keyword}': {e}")
            count = 0 # Default to 0 on error
        features[f"{keyword.replace(' ', '_')}_count"] = count # Use _ for spaces in feature name

    return features


# Function to preprocess data and train the model using PRE-CALCULATED features
def train_model(df, feature_keyword_list):
    # Remove empty rows if any
    df = df.dropna(subset=['quiri', 'target']) # Check essential columns
    df = df.fillna(0) # Fill any remaining NaN feature counts with 0

    # --- Feature Selection ---
    # Select the pre-calculated feature columns directly
    feature_cols = [f"{keyword.replace(' ', '_')}_count" for keyword in feature_keyword_list]

    # Verify all expected feature columns exist in the DataFrame
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing feature columns in uploaded CSV: {', '.join(missing_cols)}")
        st.stop()

    features = df[feature_cols]

    # --- Target ---
    target_col_name = 'target'
    if target_col_name not in df.columns:
        st.error(f"Missing target column '{target_col_name}' in uploaded CSV.")
        st.stop()
    target = df[target_col_name]

    # Encode target variable
    le = LabelEncoder()
    target_encoded = le.fit_transform(target)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)
    train_data_size = X_train.shape

    # Train RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # Added class_weight
    model.fit(X_train, y_train)

    # Store feature names in the model - CRITICAL for prediction consistency
    model.feature_names_in_ = features.columns.tolist() # Use list explicitly

    # Evaluate performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)


    # Cross validation
    try:
        cv_scores = cross_val_score(model, features, target_encoded, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    except Exception as e:
        st.warning(f"Could not perform cross-validation: {e}")
        cv_mean = np.nan
        cv_std = np.nan


    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=le.transform(le.classes_)) # Ensure labels match classes

    # Note: index_descriptions are no longer directly available unless added to the source CSV
    # We'll primarily rely on the predicted class name.

    return (model, le, X_test, y_test, accuracy, report, cv_mean, cv_std,
            feature_importance, train_data_size, conf_matrix, features.columns.tolist()) # Return actual feature names used


# Function to display index performance metrics and plot comparisons (REMOVED as data format changed)
# def display_index_performance(df, index_type):
#     # This function assumed 'execution_time', 'memory_change', etc. which are not in the new CSV
#     st.info("Performance metrics display is disabled as the required columns (e.g., 'execution_time') are not present in the feature dataset.")
#     pass

# Function to display feature importance chart
def display_feature_importance(feature_importance):
    if feature_importance is None or feature_importance.empty:
        st.warning("Feature importance data is not available.")
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    # Ensure sorting is correct
    feature_importance_sorted = feature_importance.sort_values('Importance', ascending=True) # Ascending for horizontal bar
    ax.barh(feature_importance_sorted['Feature'], feature_importance_sorted['Importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance in Index Selection')
    st.pyplot(fig)

# Function to display confusion matrix
def display_confusion_matrix(conf_matrix, le):
    if conf_matrix is None or le is None:
        st.warning("Confusion matrix data is not available.")
        return
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
    st.write("Upload your pre-processed SQL feature data and get index recommendations.")

    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.label_encoder = None
        # st.session_state.vectorizer = None # Removed - No longer using TF-IDF for training
        st.session_state.df_uploaded = None # Store the uploaded raw df
        # st.session_state.index_descriptions = None # Removed - Not directly available
        st.session_state.feature_importance = None
        st.session_state.train_data_size = None
        st.session_state.conf_matrix = None
        st.session_state.training_feature_names = None # Store the exact feature names model was trained on
        st.session_state.accuracy = None
        st.session_state.classification_report = None
        st.session_state.cv_mean = None
        st.session_state.cv_std = None


    tab1, tab2, tab3 = st.tabs(["Upload & Train", "Query Analyzer", "Model Insights"])

    # Tab 1: Upload and Train
    with tab1:
        st.header("Upload Pre-Processed Feature Data")
        st.markdown("Upload your CSV file with pre-calculated SQL feature counts (e.g., `select_count`, `join_count`, etc.), the raw query (`quiri`), and the target index type (`target`).")
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df_uploaded = df # Store the raw uploaded data if needed later
                st.subheader("Data Preview (First 5 Rows)")
                st.dataframe(df.head())

                # Display basic info
                st.write(f"Shape: {df.shape}")
                st.write("Columns:", df.columns.tolist())
                st.write("Target Distribution:")
                st.dataframe(df['target'].value_counts())


            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                st.session_state.df_uploaded = None # Reset on error

            if st.session_state.df_uploaded is not None and st.button("Train Model"):
                with st.spinner("Training model..."):
                    try:
                        (model, le, X_test, y_test, accuracy, report, cv_mean, cv_std,
                        feature_importance, train_data_size, conf_matrix, training_feature_names) = train_model(st.session_state.df_uploaded, sql_features_list)

                        # Store results in session state
                        st.session_state.model = model
                        st.session_state.label_encoder = le
                        st.session_state.feature_importance = feature_importance
                        st.session_state.train_data_size = train_data_size
                        st.session_state.conf_matrix = conf_matrix
                        st.session_state.training_feature_names = training_feature_names # Store training feature names
                        st.session_state.accuracy = accuracy
                        st.session_state.classification_report = report
                        st.session_state.cv_mean = cv_mean
                        st.session_state.cv_std = cv_std


                        st.success("Model trained successfully!")
                        st.metric("Test Accuracy", f"{accuracy:.2%}")
                        if not np.isnan(cv_mean):
                             st.metric("Cross Validation Score (mean)", f"{cv_mean:.2%} (+/- {cv_std*2:.2%})")
                        st.write(f"Training Data Size: {train_data_size[0]} rows, {train_data_size[1]} features")
                        st.write("Features Used:", training_feature_names)


                    except Exception as e:
                        st.error(f"An error occurred during model training: {e}")
                        # Reset relevant state if training fails
                        st.session_state.model = None
                        st.session_state.label_encoder = None
                        st.session_state.feature_importance = None
                        st.session_state.conf_matrix = None
                        st.session_state.training_feature_names = None


    # Tab 2: Query Analyzer
    with tab2:
        st.header("Query Analyzer")
        if st.session_state.model is None:
            st.warning("Please upload data and train the model in the 'Upload & Train' tab first.")
        else:
            st.subheader("Enter SQL Query to Recommend Index For:")
            query_text = st.text_area("Enter your SQL query:", height=150,
                                      placeholder="SELECT customer_id, COUNT(*) FROM orders WHERE order_date > '2024-01-01' GROUP BY customer_id ORDER BY COUNT(*) DESC LIMIT 10;")

            if st.button("Analyze Query", key="analyze_query"):
                if query_text:
                    # 1. Extract features from the new query using the updated function
                    input_features_dict = extract_features_from_query(query_text, sql_features_list)

                    # 2. Get the exact feature names the model was trained on
                    training_features = st.session_state.training_feature_names
                    if not training_features:
                         st.error("Training feature names not found in session state. Please retrain the model.")
                         st.stop()

                    # 3. Create DataFrame for prediction with correct columns and order
                    # Initialize with zeros
                    input_data = {feature_name: [0] for feature_name in training_features}
                    prediction_df = pd.DataFrame(input_data)

                    # 4. Fill the DataFrame with extracted feature values
                    matched_features = 0
                    for feature_name, value in input_features_dict.items():
                        if feature_name in prediction_df.columns:
                            prediction_df[feature_name] = value
                            matched_features += 1
                        # else: # Optional: Warn about features not used by model
                        #     st.warning(f"Extracted feature '{feature_name}' was not used during training.")

                    if matched_features == 0:
                         st.error("Could not match any extracted features to the model's training features.")
                         st.stop()

                    # Ensure column order matches exactly (redundant if created correctly, but safe)
                    prediction_df = prediction_df[training_features]

                    # 5. Predict
                    try:
                        prediction = st.session_state.model.predict(prediction_df)
                        predicted_label_index = prediction[0] # Get the first (and only) prediction index
                        index_type = st.session_state.label_encoder.inverse_transform([predicted_label_index])[0]
                        probabilities = st.session_state.model.predict_proba(prediction_df)[0]
                        confidence = probabilities[predicted_label_index] * 100 # Get probability of the predicted class

                        st.success(f"Recommended index type: **{index_type.upper()}** (Confidence: {confidence:.1f}%)")

                        # Display simple description based on class name (can be enhanced)
                        if index_type == 'btree':
                             st.info("Consider a B-Tree index, suitable for range queries and equality checks on ordered data.")
                        elif index_type == 'hash':
                             st.info("Consider a Hash index, best for exact equality lookups.")
                        elif index_type == 'gist':
                             st.info("Consider a GiST index, useful for geometric data, full-text search, or range types.")
                        elif index_type == 'reevers': # Assuming 'reevers' might mean reverse B-tree
                             st.info("Consider a REVERSE index (or Descending B-Tree), potentially useful for queries ordering by DESC.")
                        else:
                             st.info("Index type description not available.")


                        st.subheader("Extracted Query Features (Counts)")
                        features_df = pd.DataFrame([input_features_dict])
                        non_zero_features = features_df.loc[:, (features_df != 0).any(axis=0)] # Show columns with non-zero counts
                        st.dataframe(non_zero_features)

                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")
                        st.error(f"Model expected features: {training_features}")
                        st.error(f"Features provided: {prediction_df.columns.tolist()}")


                else:
                    st.error("Please enter a SQL query.")

    # Tab 3: Model Insights
    with tab3:
        st.header("Model Insights")
        if st.session_state.model is None:
            st.warning("Please upload data and train the model in the 'Upload & Train' tab first.")
        else:
            st.subheader("Model Performance Metrics")
            if st.session_state.accuracy is not None:
                 st.metric("Test Accuracy", f"{st.session_state.accuracy:.2%}")
            if st.session_state.cv_mean is not None and not np.isnan(st.session_state.cv_mean):
                 st.metric("Cross Validation Score (mean)", f"{st.session_state.cv_mean:.2%} (+/- {st.session_state.cv_std*2:.2%})")
            if st.session_state.train_data_size:
                 st.write(f"Training Data Size: {st.session_state.train_data_size[0]} rows, {st.session_state.train_data_size[1]} features")

            if st.session_state.classification_report:
                 st.subheader("Classification Report")
                 # Convert report dict to DataFrame for better display
                 report_df = pd.DataFrame(st.session_state.classification_report).transpose()
                 st.dataframe(report_df)


            st.subheader("Feature Importance")
            st.write("Shows which pre-calculated features most influence index recommendations:")
            display_feature_importance(st.session_state.feature_importance)

            st.subheader("Confusion Matrix")
            st.write("How the model performed on the test data (Actual vs. Predicted):")
            display_confusion_matrix(st.session_state.conf_matrix, st.session_state.label_encoder)

            # Removed Index Description section as it's not directly in the input CSV

if __name__ == "__main__":
    main()