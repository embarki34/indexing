import csv
import re
import random # Import the random module
import os

# Specify the full path to the input file
input_filename = 'd:/faker/output/btree.txt'
output_filename_with_modified_duplicates = 'extracted_queries_with_modified_duplicates.csv'
output_filename_no_duplicates = 'extracted_queries_no_duplicates.csv'

# Check if the input file exists
if not os.path.exists(input_filename):
    print(f"Error: Input file '{input_filename}' not found")
else:
    # Proceed with your processing logic
    print(f"Processing file: {input_filename}")

all_extracted_queries = [] # Store queries exactly as extracted initially
in_sql_block = False
current_query_lines = []

# Regex to find quiri:`...` or query:`...` lines
# Captures the content inside the backticks
quiri_regex = re.compile(r'^(?:quiri|query):\s*`(.*?)`.*$', re.IGNORECASE)
# Regex to find an existing LIMIT clause at the end of a query (case-insensitive)
# Allows for an optional semicolon at the very end.
limit_pattern = re.compile(r'(LIMIT\s+\d+)(\s*;)?\s*$', re.IGNORECASE)

print("Current Working Directory:", os.getcwd())

try:
    with open(input_filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            stripped_line = line.strip()

            # --- Multi-line SQL block handling ---
            if stripped_line.lower() == '```sql': # Make case-insensitive
                in_sql_block = True
                current_query_lines = [] # Reset for new block
                continue

            if stripped_line == '```' and in_sql_block:
                in_sql_block = False
                if current_query_lines:
                    full_query = "\n".join(current_query_lines).strip()
                    if full_query:
                        all_extracted_queries.append(full_query)
                continue

            if in_sql_block:
                current_query_lines.append(line.rstrip('\n'))
                continue

            # --- Single-line quiri/query handling (only if not in a block) ---
            match = quiri_regex.match(stripped_line)
            if match:
                query_content = match.group(1).strip()
                if query_content:
                    # Optional: remove trailing semicolon for consistency *before* adding
                    # if query_content.endswith(';'):
                    #     query_content = query_content[:-1].strip()
                    if query_content: # Check again
                        all_extracted_queries.append(query_content)
                continue

    # --- Process for duplicates and modification ---
    processed_queries_with_mods = []
    query_counts = {} # Dictionary to track counts of normalized queries

    for query in all_extracted_queries:
        # Normalize for duplicate checking (lowercase, strip, remove trailing ';')
        normalized_query = query.strip().lower()
        if normalized_query.endswith(';'):
            normalized_query = normalized_query[:-1].strip()

        if not normalized_query: # Skip empty queries
             continue

        if normalized_query in query_counts:
            # This is a duplicate
            query_counts[normalized_query] += 1
            random_limit = random.randint(5, 150) # Generate random limit (adjust range as needed)

            modified_query = query # Start with the original query formatting

            # Check if the original query already has a LIMIT clause at the end
            limit_match = limit_pattern.search(modified_query)

            if limit_match:
                # Replace existing LIMIT
                existing_limit_clause = limit_match.group(1)
                semicolon = limit_match.group(2) if limit_match.group(2) else ''
                # Replace only the number part
                modified_query = limit_pattern.sub(f'LIMIT {random_limit}{semicolon}', modified_query, count=1)
            else:
                # Append new LIMIT
                # Check if original ends with semicolon and insert before it
                if modified_query.rstrip().endswith(';'):
                     # Find the position of the last semicolon and insert before it
                     last_semicolon_pos = modified_query.rfind(';')
                     modified_query = modified_query[:last_semicolon_pos].rstrip() + f' LIMIT {random_limit};' + modified_query[last_semicolon_pos+1:]
                else:
                     modified_query = modified_query.rstrip() + f' LIMIT {random_limit}' # Append directly

            processed_queries_with_mods.append(modified_query)

        else:
            # First time seeing this query
            query_counts[normalized_query] = 1
            processed_queries_with_mods.append(query) # Add the original query


    # --- Write to CSV with modified duplicates ---
    with open(output_filename_with_modified_duplicates, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['sql']) # Write header
        for query in processed_queries_with_mods:
            writer.writerow([query])

    # --- Deduplicate the ORIGINAL list for the unique file ---
    seen_unique = set()
    unique_queries = []
    for query in all_extracted_queries:
         # Normalize slightly for better deduplication
         normalized_query = query.strip()
         # Decide if you want to normalize semicolons for uniqueness check
         # If yes:
         # if normalized_query.endswith(';'):
         #     normalized_query = normalized_query[:-1].strip()

         if normalized_query and normalized_query not in seen_unique:
             seen_unique.add(normalized_query)
             unique_queries.append(query) # Add the original formatting back

    # --- Write to CSV without duplicates (using original queries) ---
    with open(output_filename_no_duplicates, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['sql']) # Write header
        for query in unique_queries:
            writer.writerow([query])

    print(f"Successfully extracted and processed {len(processed_queries_with_mods)} SQL queries (duplicates modified) to {output_filename_with_modified_duplicates}")
    print(f"Successfully extracted {len(unique_queries)} unique SQL queries (original form) to {output_filename_no_duplicates}")

except FileNotFoundError:
    print(f"Error: Input file '{input_filename}' not found")
except Exception as e:
    print(f"An error occurred: {e}")