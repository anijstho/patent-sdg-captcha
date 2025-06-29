from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import pickle
import time
import os
import numpy as np
import sqlite3
import json
import uuid
from datetime import datetime

app = Flask(__name__, static_folder='../build')
CORS(app, resources={r"*": {"origins": ["https://captcha.thomasanijs.com", "http://captcha.thomasanijs.com"]}}, supports_credentials=True)  # Enable CORS for all routes

# Database setup
DB_PATH = '/var/www/captcha-ui/captcha_responses.db'

def init_db():
    """Initialize the SQLite database with the required schema"""
    try:
        # Create the database and tables if they don't exist
        with sqlite3.connect(DB_PATH) as conn:
            # Read schema.sql file
            with open('/var/www/captcha-ui/src/schema.sql', 'r') as f:
                schema = f.read()
            
            # Execute the schema SQL
            conn.executescript(schema)
            conn.commit()
            print("Database initialized successfully")
            return True
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

# Initialize the database when the server starts
db_initialized = init_db()

# Load the dataframe once when server starts
print("Loading dataframe...")
start_time = time.time()
try:
    # Updated to use testenv.pkl instead of dummy_env.pkl
    with open('/var/www/captcha-ui/testenv.pkl', 'rb') as f:
        df = pickle.load(f)
    print(f"Dataframe loaded in {time.time() - start_time:.2f} seconds")
    print(f"Dataframe shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"Error loading dataframe: {e}")
    df = pd.DataFrame()  # Empty dataframe as fallback

# Cache for frequently requested data
from functools import lru_cache

@lru_cache(maxsize=32)
def get_cached_filtered_data(filter_str):
    # This function caches results based on the filter string
    # Used for expensive filtering operations
    return "Cached result for " + filter_str

@app.route('/api/data', methods=['GET'])
def get_data():
    """Get a sample of the dataframe data"""
    try:
        limit = request.args.get('limit', default=10, type=int)
        result = df.head(limit).to_dict(orient='records')
        return jsonify({
            'success': True,
            'data': result,
            'total': len(df)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/data/paginated', methods=['GET'])
def get_paginated_data():
    """Get paginated data from the dataframe"""
    try:
        page = request.args.get('page', default=1, type=int)
        page_size = request.args.get('size', default=50, type=int)
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        result = df.iloc[start_idx:end_idx].to_dict(orient='records')
        return jsonify({
            'success': True,
            'data': result,
            'total': len(df),
            'page': page,
            'pageSize': page_size
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/data/filter', methods=['POST'])
def filter_data():
    """Filter dataframe based on criteria in request JSON"""
    try:
        filters = request.json
        if not filters:
            return jsonify({
                'success': False,
                'error': 'No filters provided'
            }), 400
            
        filtered_df = df
        
        # Apply filters
        for column, value in filters.items():
            if column in df.columns:
                filtered_df = filtered_df[filtered_df[column] == value]
        
        result = filtered_df.head(100).to_dict(orient='records')
        return jsonify({
            'success': True,
            'data': result,
            'total': len(filtered_df),
            'filtered_from': len(df)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/columns', methods=['GET'])
def get_columns():
    """Get the column names from the dataframe"""
    try:
        return jsonify({
            'success': True,
            'columns': df.columns.tolist()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get a summary of the dataframe"""
    try:
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            'missing_values': df.isna().sum().to_dict(),
            'sample': df.head(5).to_dict(orient='records')
        }
        return jsonify({
            'success': True,
            'summary': summary
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/random_sentence', methods=['GET'])
def get_random_sentence():
    """Get a random row from the dataframe with 'abstract.sentence' and full row data"""
    try:
        if len(df) == 0:
            return jsonify({
                'success': False,
                'error': 'Dataframe is empty'
            }), 500
            
        # Check if 'abstract.sentence' column exists
        if 'abstract.sentence' not in df.columns:
            return jsonify({
                'success': False,
                'error': "'abstract.sentence' column not found in dataframe"
            }), 500
            
        # Get a random row index
        random_index = np.random.randint(0, len(df))
        
        # Get the row data
        random_row = df.iloc[random_index].to_dict()
        
        # Get the sentence
        sentence = random_row.get('abstract.sentence', 'No sentence available')
        
        # Get SDG vector if available
        sdg_vector = None
        if 'sdg_vector' in random_row:
            sdg_vector = random_row['sdg_vector']
            # Convert to list if it's numpy array or other format
            if hasattr(sdg_vector, 'tolist'):
                sdg_vector = sdg_vector.tolist()
        
        # Get SDG totals if available
        sdg_totals = None
        if 'sdg_totals' in random_row:
            sdg_totals = random_row['sdg_totals']
            # Convert to list if it's numpy array or other format
            if hasattr(sdg_totals, 'tolist'):
                sdg_totals = sdg_totals.tolist()
        
        # Get max total if available
        max_total = None
        if 'max_total' in random_row:
            max_total = float(random_row['max_total'])
        
        return jsonify({
            'success': True,
            'sentence': sentence,
            'row_data': random_row,
            'sdg_vector': sdg_vector,
            'sdg_totals': sdg_totals,
            'max_total': max_total,
            'row_index': int(random_index)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Serve React static files in production
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/session', methods=['POST'])
def create_session():
    """Create a new user session and return a session ID"""
    try:
        session_id = str(uuid.uuid4())
        user_agent = request.headers.get('User-Agent', '')
        ip_address = request.remote_addr
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, user_agent, ip_address) VALUES (?, ?, ?)",
                (session_id, user_agent, ip_address)
            )
            conn.commit()
        
        return jsonify({
            'success': True,
            'session_id': session_id
        })
    except Exception as e:
        print(f"Error creating session: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/response', methods=['POST'])
def save_response():
    """Save a user's response to the database"""
    try:
        data = request.json
        required_fields = ['session_id', 'row_index', 'sentence', 'response_type', 'streak', 'response_time_ms']
        
        # Check if all required fields are present
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f"Missing required field: {field}"
                }), 400
        
        # Extract data from request
        session_id = data['session_id']
        row_index = data['row_index'] 
        sentence = data['sentence']
        selected_sdg = data.get('selected_sdg')  # Can be None
        relevant_sdgs = json.dumps(data.get('relevant_sdgs', []))
        sdg_totals = json.dumps(data.get('sdg_totals', []))
        max_total = data.get('max_total', 0)
        is_correct = data.get('is_correct')
        response_type = data['response_type']  # "correct", "not_sure", "incorrect"
        streak = data['streak']
        response_time_ms = data['response_time_ms']
        
        # Insert response into database
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO responses 
                (session_id, row_index, sentence, selected_sdg, relevant_sdgs, sdg_totals, 
                max_total, is_correct, response_type, streak, response_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, row_index, sentence, selected_sdg, relevant_sdgs, sdg_totals,
                max_total, is_correct, response_type, streak, response_time_ms
            ))
            conn.commit()
        
        return jsonify({
            'success': True,
            'message': 'Response saved successfully'
        })
    except Exception as e:
        print(f"Error saving response: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about user responses"""
    try:
        # Optional session_id parameter to filter by session
        session_id = request.args.get('session_id')
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Basic stats query - can be expanded based on requirements
            query = """
                SELECT 
                    COUNT(*) as total_responses,
                    SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_responses,
                    SUM(CASE WHEN response_type = 'not_sure' THEN 1 ELSE 0 END) as not_sure_responses,
                    SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) as incorrect_responses,
                    MAX(streak) as max_streak,
                    AVG(response_time_ms) as avg_response_time_ms
                FROM responses
            """
            
            # Add session filter if provided
            if session_id:
                query += " WHERE session_id = ?"
                cursor.execute(query, (session_id,))
            else:
                cursor.execute(query)
                
            result = dict(cursor.fetchone())
            
            # Get distribution of responses by SDG
            cursor.execute("""
                SELECT 
                    selected_sdg, 
                    COUNT(*) as count 
                FROM responses 
                WHERE selected_sdg IS NOT NULL
                GROUP BY selected_sdg
                ORDER BY selected_sdg
            """)
            
            sdg_distribution = {
                row['selected_sdg']: row['count'] 
                for row in cursor.fetchall()
            }
            
            # Add none selections count
            cursor.execute("""
                SELECT COUNT(*) as none_count
                FROM responses
                WHERE selected_sdg IS NULL
            """)
            
            none_count = cursor.fetchone()['none_count']
            
            result['sdg_distribution'] = sdg_distribution
            result['none_selections'] = none_count
            
            return jsonify({
                'success': True,
                'stats': result
            })
    except Exception as e:
        print(f"Error getting stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Change port from 5000 to 5001 to avoid conflict with AirPlay Receiver
    app.run(debug=True, host='0.0.0.0', port=5001)
