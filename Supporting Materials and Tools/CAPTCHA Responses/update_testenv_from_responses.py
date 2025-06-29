#!/usr/bin/env python3
# filepath: /Users/thomas/Library/Mobile Documents/com~apple~CloudDocs/epo-cfs/classification-ui/update_testenv_from_responses.py

import pickle
import pandas as pd
import sqlite3
import json
import numpy as np

def load_testenv(file_path='testenv.pkl'):
    """Load the testenv DataFrame from the specified file."""
    print(f"Loading testenv data from {file_path}...")
    with open(file_path, 'rb') as f:
        testenv = pickle.load(f)
    
    print(f"Loaded testenv with {len(testenv.index)} rows")
    return testenv

def load_responses_from_db(db_path='captcha_responses.db'):
    """Load responses from the SQLite database."""
    print(f"Loading responses from {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # Query to get the relevant fields from the responses table
    query = """
    SELECT r.row_index, r.selected_sdg, r.response_type, r.relevant_sdgs, r.sdg_totals
    FROM responses r
    """
    
    # Load responses into a pandas DataFrame
    responses_df = pd.read_sql_query(query, conn)
    
    conn.close()
    
    print(f"Loaded {len(responses_df)} responses from the database")
    return responses_df

def update_testenv_from_responses(testenv, responses_df, output_path='testenv_new.pkl'):
    """
    Update testenv with ground truth labels from the responses database.
    
    Logic:
    1. For responses with empty 'selected_sdg':
       - For each SDG in the 'sdg_present' array:
         - Increment sdgXX.n by 1
         - Update sdgXX.gt = current_gt * (new_n - 1) / new_n
         
    2. For responses with a numeric 'selected_sdg':
       - If response_type is 'incorrect', ignore it
       - Otherwise:
         - Increment the corresponding sdgXX.n by 1
         - Update sdgXX.gt = (current_gt * (new_n - 1) + 1) / new_n
    """
    print("Updating testenv with captcha responses...")
    
    # Counter for tracking updates
    updates_none_selected = 0
    updates_sdg_selected = 0
    responses_ignored = 0
    
    # Convert string JSON columns to Python objects
    responses_df['relevant_sdgs'] = responses_df['relevant_sdgs'].apply(json.loads)
    responses_df['sdg_totals'] = responses_df['sdg_totals'].apply(json.loads)
    
    # Process each response
    for _, row in responses_df.iterrows():
        row_index = row['row_index']
        selected_sdg = row['selected_sdg']
        response_type = row['response_type']
        
        # Make sure row_index is within valid range
        if row_index < 0 or row_index >= len(testenv.index):
            print(f"Warning: Invalid row_index {row_index}, skipping")
            continue
        
        # Case 1: No SDG was selected
        if pd.isna(selected_sdg):
            # Get the sdg_present array for this row in testenv
            sdg_present = testenv['sdg_present'].iloc[row_index]
            
            # Process each SDG in the sdg_present array
            if isinstance(sdg_present, list) and len(sdg_present) > 0:
                for sdg_num in sdg_present:
                    # Format SDG number with leading zero if needed
                    sdg_key = f"sdg{int(sdg_num):02d}"
                    
                    # Get current values, handle None/NaN
                    current_gt = testenv.loc[row_index, f'{sdg_key}.gt']
                    current_n = testenv.loc[row_index, f'{sdg_key}.n']
                    
                    if pd.isna(current_gt) or current_gt is None:
                        current_gt = 0
                    if pd.isna(current_n) or current_n is None:
                        current_n = 0
                    
                    # Increment n
                    new_n = current_n + 1
                    
                    # Update gt using the formula: new_gt = current_gt * (new_n - 1) / new_n
                    if new_n > 0:
                        new_gt = current_gt * (new_n - 1) / new_n
                    else:
                        new_gt = 0
                    
                    # Update testenv
                    testenv.loc[row_index, f'{sdg_key}.gt'] = new_gt
                    testenv.loc[row_index, f'{sdg_key}.n'] = new_n
                
                updates_none_selected += 1
        
        # Case 2: A specific SDG was selected
        elif selected_sdg is not None and selected_sdg != "":
            # Skip if response_type is 'incorrect'
            if response_type == 'incorrect':
                responses_ignored += 1
                continue
            
            # Format the selected SDG number with leading zero
            sdg_key = f"sdg{int(selected_sdg):02d}"
            
            # Get current values, handle None/NaN
            current_gt = testenv.loc[row_index, f'{sdg_key}.gt']
            current_n = testenv.loc[row_index, f'{sdg_key}.n']
            
            if pd.isna(current_gt) or current_gt is None:
                current_gt = 0
            if pd.isna(current_n) or current_n is None:
                current_n = 0
            
            # Increment n
            new_n = current_n + 1
            
            # Update gt using formula: new_gt = (current_gt * (new_n - 1) + 1) / new_n
            if new_n > 0:
                new_gt = (current_gt * (new_n - 1) + 1) / new_n
            else:
                new_gt = 1  # Should never happen as new_n should be at least 1
            
            # Update testenv
            testenv.loc[row_index, f'{sdg_key}.gt'] = new_gt
            testenv.loc[row_index, f'{sdg_key}.n'] = new_n
            
            updates_sdg_selected += 1
    
    # Print update statistics
    print(f"\nUpdate Statistics:")
    print(f"  Responses with no SDG selected (updates to all relevant SDGs): {updates_none_selected}")
    print(f"  Responses with specific SDG selected: {updates_sdg_selected}")
    print(f"  Responses ignored (incorrect): {responses_ignored}")
    print(f"  Total responses processed: {len(responses_df)}")
    
    # Save the updated testenv back to file
    print(f"\nSaving updated testenv to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(testenv, f)
    
    print(f"Update complete. Saved as '{output_path}'")
    return testenv

def main():
    # Load the testenv DataFrame
    testenv = load_testenv()
    
    # Load responses from the database
    responses_df = load_responses_from_db()
    
    # Update testenv with the responses data
    updated_testenv = update_testenv_from_responses(testenv, responses_df)
    
    print("\nProcess completed successfully.")

if __name__ == "__main__":
    main()
