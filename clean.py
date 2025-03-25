import pandas as pd
import re

# List of known index suffixes that denote variants
INDEX_SUFFIXES = ['_bitmap', '_btree', '_gist', '_hash', '_reverse']

def extract_base_query(query_name):
    """
    Removes known index suffixes from the query_name to extract the base query.
    If none of the suffixes are found, returns the original query_name.
    """
    base = query_name
    for suffix in INDEX_SUFFIXES:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break  # Remove only one suffix
    return base

def clean_data(input_path, output_path):
    # Load the CSV file
    df = pd.read_csv(input_path)
    
    # Drop rows where the 'error' column is nonempty (assuming nonempty error indicates a problem)
    df['error'] = df['error'].fillna("").astype(str)
    df = df[df['error'].str.strip() == ""]
    
    # Convert execution_time to numeric and drop rows that fail conversion
    df['execution_time'] = pd.to_numeric(df['execution_time'], errors='coerce')
    df = df.dropna(subset=['execution_time'])
    
    # Create a new column 'base_query' by stripping known suffixes from 'query_name'
    df['base_query'] = df['query_name'].apply(extract_base_query)
    
    # For each base_query group, select the row with the lowest execution_time
    best_indices = df.loc[df.groupby('base_query')['execution_time'].idxmin()]
    
    # Save the cleaned data to output CSV file
    best_indices.to_csv(output_path, index=False)
    
    return best_indices

if __name__ == '__main__':
    # Hardcoded file paths
    input_path = 'database_performance_results.csv'
    output_path = 'train.csv'
    
    cleaned_df = clean_data(input_path, output_path)
    print(f"Cleaned data saved to {output_path}")
    print(cleaned_df.head())
