import os
import glob
import time
import psutil
import pandas as pd
import psycopg2
import logging
from contextlib import contextmanager
import re

# ----------------------#
# Setup and Configuration
# ----------------------#

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simulation_log.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Database configuration for PostgreSQL (adjust for your environment)
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 5432,
    'user': 'root',          # Replace with your PostgreSQL username
    'password': 'root',      # Replace with your PostgreSQL password
    'database': 'educ_market'
}

# Paths to folders containing SQL query files (each folder represents an index type scenario)
QUERY_FOLDERS = [
    'sql_queries/btree',
    'sql_queries/hash',
    'sql_queries/bitmap',
    'sql_queries/reverse',
    'sql_queries/gist'
]

# Dictionary to provide human-friendly descriptions for index types
INDEX_DESCRIPTIONS = {
    'btree': 'B-Tree index for range queries and sorting',
    'hash': 'Hash index for equality comparisons',
    'bitmap': 'Bitmap index for low-cardinality columns',
    'reverse': 'Reverse index for reducing concurrency contention',
    'gist': 'GiST index for complex data types'
}

# Add this to the configuration section of simulation.py
INDEX_STATEMENTS = {
    'btree': [
        "CREATE INDEX btree_orders_date ON Orders (order_date)",
        "CREATE INDEX btree_orders_user ON Orders (user_id)",
        "CREATE INDEX btree_products_category ON Products (category_id)",
        "CREATE INDEX btree_orderitems_order ON OrderItems (order_id)",
        "CREATE INDEX btree_orderitems_product ON OrderItems (product_id)"
    ],
    'hash': [
        "CREATE INDEX hash_users_email ON Users USING HASH (email)",
        "CREATE INDEX hash_products_id ON Products USING HASH (product_id)",
        "CREATE INDEX hash_orders_id ON Orders USING HASH (order_id)"
    ],
    'bitmap': [
        "CREATE INDEX bitmap_orders_status ON Orders (status)",
        "CREATE INDEX bitmap_products_category ON Products (category_id)",
        "CREATE INDEX bitmap_categories_parent ON Categories (category_id)"
    ],
    'gist': [
        "CREATE INDEX gist_orders_date ON Orders USING GIST (order_date)",
        "CREATE INDEX gist_products_name ON Products USING GIST (name gist_trgm_ops)",
        "CREATE INDEX gist_users_address ON Users USING GIST (shipping_address gist_trgm_ops)"
    ],
    'reverse': [
        "CREATE INDEX rev_users_id ON Users USING REVERSE (user_id)",
        "CREATE INDEX rev_orders_date ON Orders USING REVERSE (order_date)",
        "CREATE INDEX rev_products_name ON Products USING REVERSE (name)"
    ]
}

# ----------------------#
# Helper Functions
# ----------------------#

@contextmanager
def database_connection():
    """Context manager for creating and closing a database connection."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def extract_index_info(query_file):
    """
    Returns the appropriate index statements based on the query file path.
    """
    # Extract index type from the path
    index_type = next((t for t in INDEX_STATEMENTS.keys() if t in query_file.lower()), None)
    if index_type:
        return INDEX_STATEMENTS[index_type]
    return []


def extract_query_from_file(query_file):
    """
    Extracts the SQL query from the file, ignoring comments.
    """
    with open(query_file, 'r') as file:
        lines = file.readlines()
    
    # Filter out comment lines and empty lines
    query_lines = [line.strip() for line in lines 
                  if line.strip() and not line.strip().startswith('--')]
    
    return ' '.join(query_lines)


def drop_all_non_primary_indexes(conn):
    """
    Drops all indexes except primary (and foreign key) indexes.
    This ensures that each simulation run starts with a clean slate.
    """
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE schemaname = 'public' 
              AND indexname NOT LIKE '%_pkey'
              AND indexname NOT LIKE '%_fkey'
        """)
        indexes = cursor.fetchall()
        for (index_name,) in indexes:
            try:
                logger.info(f"Dropping index {index_name}")
                cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
                conn.commit()
            except psycopg2.Error as e:
                logger.warning(f"Error dropping index {index_name}: {e}")
                conn.rollback()
    except psycopg2.Error as e:
        logger.error(f"Error retrieving indexes: {e}")
    finally:
        cursor.close()


def create_index(conn, index_statement):
    """
    Creates an index in the database based on the given SQL statement.
    Drops any existing index with the same name before creation.
    """
    if not index_statement:
        return False
    
    cursor = conn.cursor()
    try:
        # Parse index name and table name using regex
        match = re.search(r'INDEX\s+(\w+)\s+ON\s+(\w+)', index_statement, re.IGNORECASE)
        if not match:
            logger.warning(f"Could not parse index statement: {index_statement}")
            return False
        
        index_name, table_name = match.groups()
        
        # Drop existing index if present
        try:
            cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
            conn.commit()
        except psycopg2.Error as e:
            logger.warning(f"Error dropping existing index {index_name}: {e}")
            conn.rollback()
        
        # Create the new index
        logger.info(f"Creating index: {index_statement}")
        cursor.execute(index_statement)
        conn.commit()
        return True
    except psycopg2.Error as e:
        logger.error(f"Error creating index: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()


def execute_query_with_metrics(conn, query, query_name, index_type, index_statement):
    """
    Executes a given query and records performance metrics:
    - Execution time
    - CPU usage change
    - Memory usage change
    Also fetches query results to ensure complete execution.
    """
    cursor = conn.cursor()
    
    try:
        # Record initial metrics
        process = psutil.Process(os.getpid())
        start_cpu = process.cpu_percent(interval=0.1)
        start_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        
        # Execute query and measure time
        start_time = time.time()
        cursor.execute(query)
        results = cursor.fetchall()  # Ensure query execution completes
        end_time = time.time()
        
        # Record ending metrics
        end_cpu = process.cpu_percent(interval=0.1)
        end_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Calculate metrics
        execution_time = end_time - start_time
        cpu_change = end_cpu - start_cpu
        memory_change = end_memory - start_memory
        
        logger.info(f"[{query_name}] Execution Time: {execution_time:.2f} seconds")
        logger.info(f"[{query_name}] CPU Usage: {start_cpu:.2f}% -> {end_cpu:.2f}% (Change: {cpu_change:.2f}%)")
        logger.info(f"[{query_name}] Memory Usage: {start_memory:.2f} MB -> {end_memory:.2f} MB (Change: {memory_change:.2f} MB)")
        
        # Count SQL verbs - exclude comments by removing lines starting with '--'
        query_no_comments = '\n'.join(line for line in query.split('\n') if not line.strip().startswith('--'))
        sql_verbs_count = {
            'select': query_no_comments.lower().count('select'),
            'insert': query_no_comments.lower().count('insert'),
            'update': query_no_comments.lower().count('update'),
            'delete': query_no_comments.lower().count('delete'),
            'create': query_no_comments.lower().count('create'),
            'drop': query_no_comments.lower().count('drop'),
            'alter': query_no_comments.lower().count('alter'),
            'join': query_no_comments.lower().count('join'),
            'where': query_no_comments.lower().count('where'),
            'group by': query_no_comments.lower().count('group by'),
            'order by': query_no_comments.lower().count('order by'),
            'having': query_no_comments.lower().count('having'),
            'limit': query_no_comments.lower().count('limit'),
            'between': query_no_comments.lower().count('between'),
            'in': query_no_comments.lower().count('in'),
            'like': query_no_comments.lower().count('like')
        }
        
        return {
            'query_name': query_name,
            'index_type': index_type,
            'query': query,
            'index_description': INDEX_DESCRIPTIONS.get(index_type.lower(), 'Unknown index type'),
            'index_statement': index_statement,
            'execution_time': execution_time,
            'cpu_start': start_cpu,
            'cpu_end': end_cpu,
            'cpu_change': cpu_change,
            'memory_start': start_memory,
            'memory_end': end_memory,
            'memory_change': memory_change,
            'result_count': len(results) if results else 0,
            **sql_verbs_count  # Add SQL verbs count to the result
        }
    except psycopg2.Error as e:
        logger.error(f"[{query_name}] Query execution error: {e}")
        return {
            'query_name': query_name,
            'index_type': index_type,
            'query': query,
            'index_description': INDEX_DESCRIPTIONS.get(index_type.lower(), 'Unknown index type'),
            'index_statement': index_statement,
            'execution_time': -1,
            'cpu_start': -1,
            'cpu_end': -1,
            'cpu_change': -1,
            'memory_start': -1,
            'memory_end': -1,
            'memory_change': -1,
            'result_count': -1,
            'error': str(e),
            **{verb: 0 for verb in ['select', 'insert', 'update', 'delete', 'create', 'drop', 'alter', 'join', 'where', 'group by', 'order by', 'having', 'limit', 'between', 'in', 'like']}  # Default counts to 0
        }
    finally:
        cursor.close()


# ----------------------#
# Main Simulation Function
# ----------------------#

def run_simulation():
    """
    Main function that orchestrates the simulation:
    1. Sets up required PostgreSQL extensions.
    2. Iterates through each query folder (each representing a particular index type).
    3. For each query file:
       a. Extracts the query and index creation statement.
       b. Drops any non-primary indexes.
       c. Creates the index (if provided).
       d. Executes the query and measures performance.
       e. Optionally, tests alternative index scenarios.
    4. Aggregates and saves the results as CSV reports.
    """
    results = []
    
    # Ensure necessary PostgreSQL extensions are enabled
    try:
        with database_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")
            conn.commit()
            cursor.close()
    except Exception as e:
        logger.error(f"Error setting up PostgreSQL extensions: {e}")
    
    with database_connection() as conn:
        for folder in QUERY_FOLDERS:
            if not os.path.exists(folder):
                logger.warning(f"Folder not found: {folder}")
                continue
                
            index_type = os.path.basename(folder).lower()
            query_files = glob.glob(os.path.join(folder, "*.sql"))
            
            for query_file in query_files:
                query_name = os.path.basename(query_file).replace('.sql', '')
                logger.info(f"Processing query: {query_name} with index type: {index_type}")
                
                # Extract query
                query = extract_query_from_file(query_file)
                if not query:
                    logger.warning(f"Could not extract query from {query_file}")
                    continue
                
                # Clean up existing indexes
                drop_all_non_primary_indexes(conn)
                
                # Create all indexes for this type
                index_statements = INDEX_STATEMENTS.get(index_type, [])
                for index_statement in index_statements:
                    created = create_index(conn, index_statement)
                    if not created:
                        logger.warning(f"Failed to create index: {index_statement}")
                
                # Execute query and capture metrics
                result = execute_query_with_metrics(
                    conn, 
                    query, 
                    query_name, 
                    index_type, 
                    '; '.join(index_statements)  # Store all index statements used
                )
                results.append(result)
                
                # Test with other index types
                for alt_index_type in INDEX_STATEMENTS.keys():
                    if alt_index_type == index_type:
                        continue
                    
                    # Drop existing indexes
                    drop_all_non_primary_indexes(conn)
                    
                    # Create alternative indexes
                    alt_index_statements = INDEX_STATEMENTS.get(alt_index_type, [])
                    for index_statement in alt_index_statements:
                        created = create_index(conn, index_statement)
                        if not created:
                            logger.warning(f"Failed to create alternative index: {index_statement}")
                    
                    # Test query with alternative indexes
                    alt_result = execute_query_with_metrics(
                        conn, 
                        query, 
                        f"{query_name}_{alt_index_type}", 
                        alt_index_type, 
                        '; '.join(alt_index_statements)
                    )
                    results.append(alt_result)
    
    # ----------------------#
    # Aggregation and Reporting
    # ----------------------#
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    columns_order = [
        'query_name', 'index_type','query', 'index_description', 
        'index_statement', 'execution_time', 
        'cpu_start', 'cpu_end', 'cpu_change',
        'memory_start', 'memory_end', 'memory_change',
        'result_count'
    ]
    
    # Add SQL verbs count to columns order
    sql_verbs = ['select', 'insert', 'update', 'delete', 'create', 'drop', 'alter', 'join', 'where', 'group by', 'order by', 'having', 'limit', 'between', 'in', 'like']
    columns_order.extend(sql_verbs)
    
    # Add error column if it exists
    if 'error' in df.columns:
        columns_order.append('error')
    
    df = df[columns_order]
    
    # Save full results to CSV
    csv_path = 'database_performance_results.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Detailed results saved to {csv_path}")
    
    # Generate summary: Average metrics per query and index type
    summary_df = df.groupby(['index_type', 'query_name']).agg({
        'execution_time': 'mean',
        'cpu_change': 'mean',
        'memory_change': 'mean'
    }).reset_index()
    
    summary_path = 'performance_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Performance summary saved to {summary_path}")
    
    # Identify the best index type per query based on minimum execution time
    try:
        best_indexes = df.loc[df.groupby('query_name')['execution_time'].idxmin()]
        best_indexes = best_indexes[['query_name', 'index_type', 'execution_time']]
        best_indexes.columns = ['query_name', 'best_index_type', 'best_execution_time']
        best_path = 'best_indexes.csv'
        best_indexes.to_csv(best_path, index=False)
        logger.info(f"Best index configurations saved to {best_path}")
    except Exception as e:
        logger.error(f"Error generating best index summary: {e}")


# ----------------------#
# Entry Point
# ----------------------#

if __name__ == "__main__":
    logger.info("Starting database performance simulation")
    run_simulation()
    logger.info("Simulation completed")
