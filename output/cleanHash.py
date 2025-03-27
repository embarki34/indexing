import re
import csv
import json
from collections import defaultdict

def normalize_query(query_str):
    """Cleans up the query string for consistent comparison."""
    if not query_str:
        return ""
    # Remove leading/trailing whitespace
    query_str = query_str.strip()
    # Remove potential leading/trailing backticks or quotes if they slipped through
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

def clean_index_value(index_val):
    """Removes 'preferindex:' prefix and cleans up the index value."""
    if not index_val:
        return ''
    # Remove prefix if present (case-insensitive)
    if index_val.lower().startswith('preferindex:'):
        index_val = index_val[len('preferindex:'):]
    # Strip whitespace and trailing junk
    return index_val.strip().rstrip(';,')

def extract_data(input_filename):
    """Extracts query and index pairs from the input file."""
    extracted_pairs = []
    in_sql_block = False
    current_sql_block = ""
    last_sql_block = ""
    json_buffer = ""
    in_json_block = False
    last_processed_line_index = -1 # To avoid double processing

    # Regex patterns
    # Captures quiri/query and optional preferindex on the same line
    quiri_pattern = re.compile(r"^(?:quiri|query):\s*`(.+?)`(?:;)?\s*(?:preferindex:\s*(.+))?$", re.IGNORECASE)
    # Captures standalone preferindex lines (allowing potential leading/trailing ';')
    preferindex_pattern = re.compile(r"^\s*(?:;|preferindex:)\s*(.+?)\s*;?$", re.IGNORECASE)
    # Captures quiri/query line without index, for checking next line
    quiri_only_pattern = re.compile(r"^(?:quiri|query):\s*`(.+?)`(?:;)?$", re.IGNORECASE)

    with open(input_filename, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

        i = 0
        while i < len(lines):
            if i <= last_processed_line_index: # Skip if already processed (e.g., by lookahead)
                 i += 1
                 continue

            line = lines[i].strip()
            original_line_index = i

            # --- Handle JSON blocks ---
            if line.startswith('{') and not in_json_block:
                in_json_block = True
                json_buffer = line
                # Simple check for single-line JSON
                if line.endswith('}'):
                    try:
                        data = json.loads(json_buffer)
                        query = normalize_query(data.get('quiri', data.get('query', '')))
                        index = clean_index_value(data.get('preferindex', '')) # Clean index value
                        if query:
                            extracted_pairs.append((query, index))
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
                if line.endswith('}'):
                    try:
                        start_brace = json_buffer.find('{')
                        if start_brace != -1:
                            data = json.loads(json_buffer[start_brace:])
                            query = normalize_query(data.get('quiri', data.get('query', '')))
                            index = clean_index_value(data.get('preferindex', '')) # Clean index value
                            if query:
                                extracted_pairs.append((query, index))
                                last_processed_line_index = i
                        else:
                            print(f"Warning: JSON buffer malformed, starting '{json_buffer[:50]}...'")
                    except json.JSONDecodeError:
                        try:
                            data = json.loads(line)
                            query = normalize_query(data.get('quiri', data.get('query', '')))
                            index = clean_index_value(data.get('preferindex', '')) # Clean index value
                            if query:
                                extracted_pairs.append((query, index))
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
                # Don't immediately add, wait to see if next line is preferindex or quiri
                i += 1
                continue
            elif in_sql_block:
                current_sql_block += line + "\n"
                i += 1
                continue

            # --- Handle quiri/query lines with optional index ---
            match_quiri = quiri_pattern.match(line)
            if match_quiri:
                query = normalize_query(match_quiri.group(1))
                # Group 2 might be None if preferindex wasn't present
                index_val = match_quiri.group(2) or ''
                index = clean_index_value(index_val) # Clean index value
                if query:
                    extracted_pairs.append((query, index))
                    last_processed_line_index = i
                last_sql_block = "" # quiri line overrides SQL block context
                i += 1
                continue

             # --- Handle quiri/query lines where index might be on the next line ---
            match_quiri_only = quiri_only_pattern.match(line)
            if match_quiri_only:
                query = normalize_query(match_quiri_only.group(1))
                index = ''
                 # Check next line for preferindex
                if i + 1 < len(lines):
                    next_line_strip = lines[i+1].strip()
                    # Check if next line *starts* with preferindex or just ; preferindex
                    if next_line_strip.lower().startswith(('preferindex:', '; preferindex:')):
                         next_line_match = preferindex_pattern.match(next_line_strip)
                         if next_line_match:
                             index = clean_index_value(next_line_match.group(1)) # Clean index value
                             i += 1 # Consume the index line as well
                if query:
                    extracted_pairs.append((query, index))
                    last_processed_line_index = i
                last_sql_block = "" # quiri line overrides SQL block context
                i += 1
                continue


            # --- Handle standalone preferindex lines potentially following an SQL block ---
            match_index = preferindex_pattern.match(line)
            if match_index and last_sql_block:
                 # Check if the previous line was the end of the SQL block
                 # Or if there were only blank lines between SQL end and this index line
                 prev_non_blank_index = original_line_index -1
                 while prev_non_blank_index >= 0 and not lines[prev_non_blank_index].strip():
                      prev_non_blank_index -= 1

                 if prev_non_blank_index >= 0 and lines[prev_non_blank_index].strip().startswith('```'):
                     query = sql_to_single_line(last_sql_block)
                     index = clean_index_value(match_index.group(1)) # Clean index value
                     if query:
                         extracted_pairs.append((query, index))
                         last_processed_line_index = i
                     last_sql_block = "" # SQL block used
                     i += 1
                     continue
                 else: # Index line doesn't seem related to the last SQL block
                      last_sql_block = ""


            # --- Handle SQL block that had no associated index/quiri on subsequent lines ---
            if last_sql_block and i > original_line_index: # Ensure we've moved past the block end
                 # Check if the block *ended* on the previous line index
                 prev_line_idx = i -1
                 while prev_line_idx >= 0 and not lines[prev_line_idx].strip():
                    prev_line_idx -=1
                 # If the last non-blank line ended the SQL block and wasn't processed
                 if prev_line_idx >= 0 and lines[prev_line_idx].strip().startswith('```') and prev_line_idx > last_processed_line_index:
                     query = sql_to_single_line(last_sql_block)
                     if query:
                         # Avoid adding if a quiri already added it
                         is_already_added = any(p[0] == query for p in extracted_pairs[-5:])
                         if not is_already_added:
                             extracted_pairs.append((query, '')) # No index found
                             last_processed_line_index = prev_line_idx # Mark block end line as processed
                     last_sql_block = ""

            # If we processed something this iteration, continue loop
            if i > original_line_index:
                 continue
            # Otherwise, if nothing matched or happened, ensure we advance
            else:
                 last_sql_block = "" # Clear context if line was unparsed/unrelated
                 i += 1


        # Final check for any trailing SQL block without index
        if last_sql_block:
             query = sql_to_single_line(last_sql_block)
             if query:
                 is_already_added = any(p[0] == query for p in extracted_pairs[-5:])
                 if not is_already_added:
                      extracted_pairs.append((query, ''))

    return extracted_pairs


def write_csv(data, filename, include_all_occurrences=False):
    """Writes the extracted data to a CSV file.

    Args:
        data: List of (query, index) tuples.
        filename: Name of the output CSV file.
        include_all_occurrences: If True, writes all found pairs, including
                                 identical duplicates. If False, writes only
                                 unique pairs based on query and normalized index.
    """
    processed_data = []
    seen_pairs = set() # For unique pairs version

    # Normalize index names function for consistent comparison
    def normalize_index_name(index_val):
        if not index_val: return ''
        norm = index_val.lower().strip()
        if 'btree' in norm or 'b-tree' in norm:
            match = re.match(r'(.+?)\s*\(b[-]?tree\)', norm)
            if match: return f"{match.group(1).strip()}(btree)"
            else: return 'btree'
        if 'hash' in norm:
             match = re.match(r'(.+?)\s*\(hash\)', norm)
             if match: return f"{match.group(1).strip()}(hash)"
             else: return 'hash'
        if norm.startswith('index on'): return norm # Keep complex definitions
        # Simplified: treat bare column names or simple descriptions differently than standard types
        if re.match(r'^[a-z0-9_() ]+$', norm) and not norm in ['btree', 'hash']:
            # It's likely a column name or simple description, keep it lowercased/stripped
            return norm
        # Fallback for unrecognized patterns or explicit types
        return norm

    if include_all_occurrences:
        # Write all pairs exactly as found (after extraction cleaning)
        # No duplicate handling or modification needed here per new requirement
        processed_data = [[query, index] for query, index in data]
    else: # Unique pairs only
        for query, index in data:
            # Use normalized index for uniqueness check
            norm_index_val = normalize_index_name(index)
            pair_key = (query, norm_index_val)

            if pair_key not in seen_pairs:
                # Write the original extracted index value
                processed_data.append([query, index])
                seen_pairs.add(pair_key)

    with open(filename, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['quiri', 'index']) # Header
        writer.writerows(processed_data)
    print(f"Successfully wrote {len(processed_data)} rows to {filename}")


# --- Main Execution ---
input_filename = 'hach.txt'
# Renamed output files to reflect the new logic
output_filename_all_occurrences = 'hach_extracted_queries_all_occurrences.csv'
output_filename_no_duplicates = 'hach_extracted_queries_no_duplicates.csv'

all_extracted_data = extract_data(input_filename)

# Create CSV with *all* found occurrences (includes identical duplicates)
write_csv(all_extracted_data, output_filename_all_occurrences, include_all_occurrences=True)

# Create CSV with unique pairs only (based on query and normalized index)
write_csv(all_extracted_data, output_filename_no_duplicates, include_all_occurrences=False)