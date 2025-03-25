import psycopg2

def fetch_query_metrics(query):
    cursor = None
    connection = None
    try:
        # Connect to the database
        connection = psycopg2.connect(
            host='127.0.0.1',
            port=5432,
            user='root',          # Replace with your PostgreSQL username
            password='root',      # Replace with your PostgreSQL password
            database='educ_market'
        )
        cursor = connection.cursor()
        
        # Execute the query
        cursor.execute(query)
        
        # Fetch the results
        result = cursor.fetchone()
        
        # Assuming the query returns result_count, memory_usage, and cpu_usage
        if result:
            # Adjusting to handle potential changes in the result structure
            result_count = result[0] if len(result) > 0 else None
            memory_usage = result[1] if len(result) > 1 else None
            cpu_usage = result[2] if len(result) > 2 else None
            
            return {
                'result_count': result_count,
                'memory_usage': memory_usage,
                'cpu_usage': cpu_usage
            }
        else:
            return None

    except Exception as e:
        print(f"Error fetching query metrics: {e}")
        return None
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            
            connection.close()

# Example usage
query = """
SELECT
    o.order_id,
    o.user_id,
    o.order_date,
    o.total_amount,
    u.email,
    COUNT(oi.order_item_id) AS item_count,
    SUM(oi.quantity) AS total_quantity
FROM
    Orders o
JOIN
    Users u ON o.user_id = u.user_id
JOIN
    OrderItems oi ON o.order_id = oi.order_id
WHERE
    o.order_id = 12345
GROUP BY
    o.order_id, o.user_id, o.order_date, o.total_amount, u.email;
"""
metrics = fetch_query_metrics(query)
print(metrics)
