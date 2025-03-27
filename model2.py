import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import re
# import sqlparse # No longer needed if features are pre-calculated or handled by regex
from sklearn.feature_extraction.text import TfidfVectorizer # Re-added for 'quiri' analysis
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import os

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="DB Index Optimizer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the features based on the CSV header (excluding 'quiri', 'target')
sql_features_list = [
    'select', 'insert', 'update', 'delete', 'create', 'drop', 'alter',
    'join', 'where', 'group by', 'order by', 'having', 'limit',
    'between', 'in', 'like', 'from', 'on', 'as', 'desc', 'count', 'sum', 'avg', 'distinct'
]

# Function to parse NEW SQL query and extract keyword count features
def extract_keyword_features_from_query(query_text, keywords):
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
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        try:
            count = len(re.findall(pattern, query_lower))
        except re.error as e:
            st.error(f"Regex error for keyword '{keyword}': {e}")
            count = 0
        features[f"{keyword.replace(' ', '_')}_count"] = count

    return features


# Function to preprocess data and train the model INCLUDING TF-IDF on 'quiri'
def train_model(df, feature_keyword_list, max_tfidf_features=50): # Added max_tfidf_features
    # Remove empty rows if any
    df = df.dropna(subset=['quiri', 'target'])
    df['quiri'] = df['quiri'].fillna('') # Ensure no NaN in text column
    df = df.fillna(0) # Fill any remaining NaN feature counts with 0

    # --- Feature Selection ---
    # 1. Select the pre-calculated keyword count features
    keyword_feature_cols = [f"{keyword.replace(' ', '_')}_count" for keyword in feature_keyword_list]
    missing_cols = [col for col in keyword_feature_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing keyword count columns in uploaded CSV: {', '.join(missing_cols)}")
        st.stop()
    keyword_features = df[keyword_feature_cols].reset_index(drop=True) # Reset index for safe concat

    # 2. Create TF-IDF Vectorizer for 'quiri' text
    vectorizer = TfidfVectorizer(max_features=max_tfidf_features,
                                 ngram_range=(1, 2), # Consider single words and pairs
                                 stop_words='english')
    try:
        tfidf_vectors = vectorizer.fit_transform(df['quiri'])
        tfidf_feature_names = ['tfidf_' + name for name in vectorizer.get_feature_names_out()]
        tfidf_features = pd.DataFrame(tfidf_vectors.toarray(), columns=tfidf_feature_names).reset_index(drop=True)
    except Exception as e:
        st.error(f"Error during TF-IDF vectorization: {e}")
        st.stop()

    # 3. Combine keyword counts and TF-IDF features
    features = pd.concat([keyword_features, tfidf_features], axis=1)

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
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Store feature names in the model - CRITICAL for prediction consistency
    # Includes both keyword counts and tfidf features
    model.feature_names_in_ = features.columns.tolist()

    # Evaluate performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)


    # Cross validation
    try:
        # Use a simplified model for CV if the main one is complex and slow
        cv_model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
        cv_scores = cross_val_score(cv_model, features, target_encoded, cv=5)
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
    conf_matrix = confusion_matrix(y_test, y_pred, labels=le.transform(le.classes_))

    # Return the fitted vectorizer as well
    return (model, le, vectorizer, X_test, y_test, accuracy, report, cv_mean, cv_std,
            feature_importance, train_data_size, conf_matrix, features.columns.tolist())


# Function to display feature importance chart (show top N)
def display_feature_importance(feature_importance, top_n=30):
    if feature_importance is None or feature_importance.empty:
        st.warning("Feature importance data is not available.")
        return
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3))) # Adjust height
    # Get top N features
    top_features = feature_importance.head(top_n).sort_values('Importance', ascending=True) # Ascending for horizontal bar
    ax.barh(top_features['Feature'], top_features['Importance'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances in Index Selection')
    plt.tight_layout() # Adjust layout
    st.pyplot(fig)

# Function to display confusion matrix
def display_confusion_matrix(conf_matrix, le):
    if conf_matrix is None or le is None:
        st.warning("Confusion matrix data is not available.")
        return
    fig, ax = plt.subplots(figsize=(max(6, len(le.classes_)*0.8), max(5, len(le.classes_)*0.6))) # Adjust size
    labels = le.classes_
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha='right') # Rotate labels if many classes
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

# Main Streamlit app
def main():
    st.title("🔍 Database Index Optimization Advisor")
    st.write("Upload pre-processed SQL feature data (keyword counts). Query text ('quiri') will be vectorized and added as features.")

    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.label_encoder = None
        st.session_state.vectorizer = None # Added vectorizer to state
        st.session_state.df_uploaded = None
        st.session_state.feature_importance = None
        st.session_state.train_data_size = None
        st.session_state.conf_matrix = None
        st.session_state.training_feature_names = None
        st.session_state.accuracy = None
        st.session_state.classification_report = None
        st.session_state.cv_mean = None
        st.session_state.cv_std = None

    tab1, tab2, tab3 = st.tabs(["Upload & Train", "Query Analyzer", "Model Insights"])

    # Tab 1: Upload and Train
    with tab1:
        st.header("Upload Pre-Processed Feature Data")
        st.markdown("Upload your CSV with keyword counts, `quiri` (raw query), and `target`. The `quiri` column will be vectorized using TF-IDF and combined with the counts.")
        uploaded_file = st.file_uploader("Upload your feature CSV file", type="csv")

        # Add TF-IDF parameter selection
        max_tfidf = st.slider("Max TF-IDF Features (from query text)", min_value=10, max_value=500, value=50, step=10)


        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df_uploaded = df
                st.subheader("Data Preview (First 5 Rows)")
                st.dataframe(df.head())
                st.write(f"Shape: {df.shape}")
                if 'target' in df.columns:
                     st.write("Target Distribution:")
                     st.dataframe(df['target'].value_counts())
                else:
                     st.warning("Target column ('target') not found in preview.")


            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                st.session_state.df_uploaded = None

            if st.session_state.df_uploaded is not None and st.button("Train Model"):
                with st.spinner(f"Training model with {max_tfidf} TF-IDF features..."):
                    try:
                        (model, le, vectorizer, X_test, y_test, accuracy, report, cv_mean, cv_std,
                        feature_importance, train_data_size, conf_matrix, training_feature_names) = train_model(st.session_state.df_uploaded, sql_features_list, max_tfidf_features=max_tfidf) # Pass max_tfidf

                        # Store results in session state
                        st.session_state.model = model
                        st.session_state.label_encoder = le
                        st.session_state.vectorizer = vectorizer # Store the fitted vectorizer
                        st.session_state.feature_importance = feature_importance
                        st.session_state.train_data_size = train_data_size
                        st.session_state.conf_matrix = conf_matrix
                        st.session_state.training_feature_names = training_feature_names # Store combined feature names
                        st.session_state.accuracy = accuracy
                        st.session_state.classification_report = report
                        st.session_state.cv_mean = cv_mean
                        st.session_state.cv_std = cv_std


                        st.success("Model trained successfully!")
                        st.metric("Test Accuracy", f"{accuracy:.2%}")
                        if not np.isnan(cv_mean):
                             st.metric("Cross Validation Score (mean)", f"{cv_mean:.2%} (+/- {cv_std*2:.2%})")
                        st.write(f"Training Data Size: {train_data_size[0]} rows, {train_data_size[1]} features (including TF-IDF)")
                        # st.write("Features Used:", training_feature_names) # Can be long, maybe omit


                    except Exception as e:
                        st.error(f"An error occurred during model training: {e}")
                        # Reset relevant state if training fails
                        st.session_state.model = None
                        st.session_state.label_encoder = None
                        st.session_state.vectorizer = None
                        st.session_state.feature_importance = None
                        st.session_state.conf_matrix = None
                        st.session_state.training_feature_names = None


    # Tab 2: Query Analyzer
    with tab2:
        st.header("Query Analyzer")
        if st.session_state.model is None or st.session_state.vectorizer is None:
            st.warning("Please upload data and train the model in the 'Upload & Train' tab first.")
        else:
            st.subheader("Enter SQL Query to Recommend Index For:")
            query_text = st.text_area("Enter your SQL query:", height=150,
                                      placeholder="SELECT customer_id, COUNT(*) FROM orders WHERE order_date > '2024-01-01' GROUP BY customer_id ORDER BY COUNT(*) DESC LIMIT 10;")

            if st.button("Analyze Query", key="analyze_query"):
                if query_text:
                    try:
                        # 1. Extract keyword count features
                        keyword_features_dict = extract_keyword_features_from_query(query_text, sql_features_list)

                        # 2. Vectorize the new query using the *fitted* vectorizer
                        query_vector = st.session_state.vectorizer.transform([query_text])
                        tfidf_feature_names = ['tfidf_' + name for name in st.session_state.vectorizer.get_feature_names_out()]
                        query_tfidf_df = pd.DataFrame(query_vector.toarray(), columns=tfidf_feature_names)

                        # 3. Get the exact feature names the model was trained on
                        training_features = st.session_state.training_feature_names
                        if not training_features:
                             st.error("Training feature names not found. Please retrain.")
                             st.stop()

                        # 4. Create DataFrame for prediction with correct columns and order
                        prediction_df = pd.DataFrame(0, index=[0], columns=training_features)

                        # 5. Fill the DataFrame with extracted feature values
                        combined_input_features = {}
                        combined_input_features.update(keyword_features_dict)
                        # Add TF-IDF features from the transformed vector
                        for col in query_tfidf_df.columns:
                            if col in prediction_df.columns: # Check if this TF-IDF feature was in training
                                combined_input_features[col] = query_tfidf_df[col].iloc[0]

                        # Populate the prediction DataFrame safely
                        for feature_name, value in combined_input_features.items():
                            if feature_name in prediction_df.columns:
                                prediction_df.loc[0, feature_name] = value
                            # else: # Feature extracted but not in training (shouldn't happen if logic is correct)
                            #     st.warning(f"Feature '{feature_name}' not found in training columns.")


                        # Ensure column order is identical to training (safety check)
                        prediction_df = prediction_df[training_features]

                        # 6. Predict
                        prediction = st.session_state.model.predict(prediction_df)
                        predicted_label_index = prediction[0]
                        index_type = st.session_state.label_encoder.inverse_transform([predicted_label_index])[0]
                        probabilities = st.session_state.model.predict_proba(prediction_df)[0]
                        confidence = probabilities[predicted_label_index] * 100

                        st.success(f"Recommended index type: **{index_type.upper()}** (Confidence: {confidence:.1f}%)")

                        # Display simple description based on class name
                        if index_type == 'btree':
                             st.info("Consider a B-Tree index, suitable for range queries (=, >, <, BETWEEN) and equality checks on ordered data. Good general-purpose index.")
                        elif index_type == 'hash':
                             st.info("Consider a Hash index, best ONLY for exact equality lookups (=). Generally faster than B-Tree for equality but less flexible.")
                        elif index_type == 'gist':
                             st.info("Consider a GiST index, useful for geometric data, full-text search, range types, or implementing custom data types.")
                        elif index_type == 'reevers': # Assuming 'reevers' might mean reverse B-tree
                             st.info("Consider a REVERSE index (or Descending B-Tree in some DBs), potentially useful if queries frequently sort by a column in DESC order.")
                        else:
                             st.info("Index type description not available.")


                        st.subheader("Extracted Query Features (Counts & Top TF-IDF)")
                        features_for_display = {}
                        # Show non-zero keyword counts
                        for k,v in keyword_features_dict.items():
                            if v > 0:
                                features_for_display[k] = v
                        # Show top N non-zero TF-IDF features
                        tfidf_series = query_tfidf_df.iloc[0]
                        top_tfidf = tfidf_series[tfidf_series > 0].nlargest(5) # Show top 5 relevant terms/ngrams
                        for k,v in top_tfidf.items():
                            features_for_display[k] = round(v, 3)

                        if features_for_display:
                             st.dataframe(pd.Series(features_for_display).reset_index().rename(columns={'index':'Feature', 0:'Value'}))
                        else:
                             st.write("No relevant keyword counts or TF-IDF features extracted (or query was empty).")


                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")
                        if 'training_feature_names' in st.session_state and st.session_state.training_feature_names:
                            st.error(f"Model expected features ({len(st.session_state.training_feature_names)}): {st.session_state.training_feature_names}")
                        if 'prediction_df' in locals():
                             st.error(f"Features provided ({len(prediction_df.columns)}): {prediction_df.columns.tolist()}")


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
                 report_df = pd.DataFrame(st.session_state.classification_report).transpose()
                 st.dataframe(report_df)


            st.subheader("Feature Importance")
            st.write("Shows which features (keyword counts & TF-IDF terms) most influence recommendations:")
            display_feature_importance(st.session_state.feature_importance, top_n=30) # Show top 30

            st.subheader("Confusion Matrix")
            st.write("How the model performed on the test data (Actual vs. Predicted):")
            display_confusion_matrix(st.session_state.conf_matrix, st.session_state.label_encoder)

if __name__ == "__main__":
    main()