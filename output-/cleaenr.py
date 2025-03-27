import re
import csv
import json
import os # To get filename base

def normalize_query(query_str):
    """Cleans up the query string for consistent comparison."""
    if not query_str:
        return ""
    # Handle embedded SQL blocks within quiri tags
    if query_str.strip().startswith('```sql'):
         lines = [line.strip() for line in query_str.strip().split('\n')]
         # Remove ```sql and ``` lines
         lines = [line for line in lines if not line.startswith('```')]
         query_str = ' '.join(lines)

    # Remove leading/trailing whitespace
    query_str = query_str.strip()
    # Remove potential leading/trailing backticks or quotes
    query_str = query_str.strip('`\'"')
    # Replace multiple whitespace chars with a single space
    query_str = re.sub(r'\s+', ' ', query_str)
    # Remove trailing semicolon if present
    if query_str.endswith(';'):
        query_str = query_str[:-1]
    # Convert to lowercase
    query_str = query_str.lower()
    return query_str

def sql_to_single_line(sql_block):
    """Converts a multi-line SQL block to a normalized single line."""
    if not sql_block:
        return ""
    lines = [line.strip() for line in sql_block.strip().split('\n') if line.strip()]
    single_line = ' '.join(lines)
    return normalize_query(single_line)

def extract_queries_from_file(input_filename):
    """Extracts only query strings from the input file."""
    extracted_queries = []
    in_sql_block = False
    current_sql_block = ""
    last_sql_block = ""
    json_buffer = ""
    in_json_block = False
    last_processed_line_index = -1 # To avoid double processing

    # Regex patterns - simplified as we only need the query part now
    quiri_pattern = re.compile(r"^(?:quiri|query):\s*`(.+?)`(?:;)?", re.IGNORECASE)
    # Pattern to find preferindex (to ignore it if on the same line as quiri)
    preferindex_on_quiri_line = re.compile(r"preferindex:\s*.+$", re.IGNORECASE)

    with open(input_filename, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

        i = 0
        while i < len(lines):
            if i <= last_processed_line_index:
                i += 1
                continue

            line = lines[i].strip()
            original_line_index = i

            # --- Handle JSON blocks ---
            if line.startswith('{') and not in_json_block:
                in_json_block = True
                json_buffer = line
                if line.endswith('}'): # Single line JSON
                    try:
                        data = json.loads(json_buffer)
                        query = normalize_query(data.get('quiri', data.get('query', '')))
                        if query:
                            extracted_queries.append(query)
                            last_processed_line_index = i
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON: {json_buffer}")
                    in_json_block = False
                    json_buffer = ""
                    last_sql_block = ""
                i += 1
                continue

            if in_json_block:
                json_buffer += line
                if line.endswith('}'): # End of multi-line JSON
                    try:
                        start_brace = json_buffer.find('{')
                        if start_brace != -1:
                            data = json.loads(json_buffer[start_brace:])
                            query = normalize_query(data.get('quiri', data.get('query', '')))
                            if query:
                                extracted_queries.append(query)
                                last_processed_line_index = i
                        else:
                            print(f"Warning: JSON buffer malformed: {json_buffer[:50]}...")
                    except json.JSONDecodeError:
                        try: # Try last line only
                             data = json.loads(line)
                             query = normalize_query(data.get('quiri', data.get('query', '')))
                             if query:
                                extracted_queries.append(query)
                                last_processed_line_index = i
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON buffer ending in: {line}")
                    in_json_block = False
                    json_buffer = ""
                    last_sql_block = ""
                i += 1
                continue

            # --- Handle SQL blocks ---
            if line.startswith('```sql'):
                in_sql_block = True
                current_sql_block = ""
                last_sql_block = ""
                i += 1
                continue
            elif line.startswith('```') and in_sql_block:
                in_sql_block = False
                last_sql_block = current_sql_block # Store completed block
                # Don't add yet, check next line isn't a quiri for the same block
                i += 1
                continue
            elif in_sql_block:
                current_sql_block += line + "\n"
                i += 1
                continue

            # --- Handle quiri/query lines ---
            match_quiri = quiri_pattern.match(line)
            if match_quiri:
                query = normalize_query(match_quiri.group(1))
                if query:
                    extracted_queries.append(query)
                    last_processed_line_index = i
                # If a quiri line is found, it uses the query from there,
                # so discard any pending SQL block before it.
                last_sql_block = ""
                i += 1
                continue

            # --- Handle SQL block that had no associated quiri/JSON on subsequent lines ---
            if last_sql_block and i > original_line_index: # Moved past block end
                prev_line_idx = i - 1
                while prev_line_idx >= 0 and not lines[prev_line_idx].strip():
                    prev_line_idx -= 1
                # If the SQL block just ended and wasn't processed by a quiri line
                if prev_line_idx >= 0 and lines[prev_line_idx].strip().startswith('```') and prev_line_idx > last_processed_line_index:
                     query = sql_to_single_line(last_sql_block)
                     if query:
                         # Check if same query was JUST added by a quiri line (avoid double add)
                         if not extracted_queries or extracted_queries[-1] != query:
                              extracted_queries.append(query)
                              last_processed_line_index = prev_line_idx # Mark block end line as processed
                     last_sql_block = ""

            # If we processed something this iteration, continue loop
            if i > original_line_index:
                 continue
            else: # Didn't match anything, advance
                 last_sql_block = "" # Clear context if line was unparsed/unrelated
                 i += 1


        # Final check for any trailing SQL block
        if last_sql_block:
             query = sql_to_single_line(last_sql_block)
             if query:
                 if not extracted_queries or extracted_queries[-1] != query:
                     extracted_queries.append(query)

    # Remove exact duplicates resulted from parsing ambiguity
    unique_queries = []
    seen = set()
    for q in extracted_queries:
        if q not in seen:
            unique_queries.append(q)
            seen.add(q)

    return unique_queries


def extract_sql_features(query, keywords):
    """Counts occurrences of SQL keywords/clauses in a query string."""
    features = {}
    # Ensure query is lowercase for case-insensitive matching
    query_lower = query.lower()

    for keyword in keywords:
        # Use word boundaries (\b) to match whole words/phrases
        # Escape special regex characters in the keyword (like *)
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        try:
            count = len(re.findall(pattern, query_lower))
        except re.error as e:
            print(f"Regex error for keyword '{keyword}': {e}")
            count = 0 # Default to 0 on error
        features[f"{keyword.replace(' ', '_')}_count"] = count # Use _ for spaces in feature name

    return features

# --- Main Execution ---
input_files = ['reevers.txt', 'gist.txt', 'btree.txt', 'hash.txt']
output_csv_filename = 'sql_features_with_target.csv'

# Define the features (SQL verbs/clauses) to count
sql_features_list = [
    'select', 'insert', 'update', 'delete', 'create', 'drop', 'alter',
    'join', 'where', 'group by', 'order by', 'having', 'limit',
    'between', 'in', 'like', 'from', 'on', 'as', 'desc', 'count', 'sum', 'avg', 'distinct'
] # Added a few more common ones based on input data

all_rows_data = []
processed_queries = set() # Keep track of queries already processed to avoid full duplicates

for filename in input_files:
    if not os.path.exists(filename):
        print(f"Warning: File not found - {filename}")
        continue

    print(f"Processing file: {filename}...")
    # Derive target class from filename (e.g., 'reevers.txt' -> 'reevers')
    target_class = os.path.splitext(os.path.basename(filename))[0]

    queries = extract_queries_from_file(filename)
    print(f"  Found {len(queries)} unique queries.")

    for query in queries:
        if not query: # Skip empty queries
            continue

        # Check if this exact query has already been processed (from any file)
        # If you want duplicates across files, remove this check
        if query in processed_queries:
            # print(f"  Skipping duplicate query: {query[:100]}...") # Optional: log skipped duplicates
            continue
        processed_queries.add(query)


        # Extract features for the current query
        features = extract_sql_features(query, sql_features_list)

        # Prepare row for CSV
        row_dict = {'quiri': query}
        row_dict.update(features)
        row_dict['target'] = target_class
        all_rows_data.append(row_dict)

print(f"\nTotal unique queries processed across all files: {len(all_rows_data)}")

# --- Write the final CSV ---
if all_rows_data:
    # Define CSV header order
    csv_headers = ['quiri'] + [f"{feat}_count" for feat in sql_features_list] + ['target']

    print(f"Writing data to {output_csv_filename}...")
    try:
        with open(output_csv_filename, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(all_rows_data)
        print(f"Successfully wrote {len(all_rows_data)} rows to {output_csv_filename}")
    except IOError as e:
        print(f"Error writing CSV file: {e}")
else:
    print("No data extracted to write to CSV.")