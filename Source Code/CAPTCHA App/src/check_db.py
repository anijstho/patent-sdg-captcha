#!/usr/bin/env python3

import sqlite3
import json
from datetime import datetime
import sys

DB_PATH = '/var/www/captcha-ui/captcha_responses.db'

def print_sessions():
    """Print all sessions in the database"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions")
            sessions = cursor.fetchall()
            
            print(f"\n=== Sessions ({len(sessions)}) ===")
            for session in sessions:
                created_at = datetime.fromisoformat(session['created_at'])
                print(f"Session ID: {session['session_id']}")
                print(f"Created: {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"User Agent: {session['user_agent']}")
                print(f"IP: {session['ip_address']}")
                print("-" * 50)
    except Exception as e:
        print(f"Error getting sessions: {e}")

def print_responses(limit=10):
    """Print responses in the database"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM responses ORDER BY created_at DESC LIMIT ?", (limit,))
            responses = cursor.fetchall()
            
            print(f"\n=== Recent Responses ({len(responses)}) ===")
            for resp in responses:
                created_at = datetime.fromisoformat(resp['created_at'])
                
                # Parse JSON strings
                relevant_sdgs = json.loads(resp['relevant_sdgs']) if resp['relevant_sdgs'] else []
                sdg_totals = json.loads(resp['sdg_totals']) if resp['sdg_totals'] else []
                
                # Format response data
                print(f"Response ID: {resp['id']}")
                print(f"Session ID: {resp['session_id']}")
                print(f"Created: {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Row Index: {resp['row_index']}")
                print(f"Selected SDG: {resp['selected_sdg'] or 'None'}")
                print(f"Response Type: {resp['response_type']}")
                print(f"Is Correct: {resp['is_correct']}")
                print(f"Streak: {resp['streak']}")
                print(f"Response Time: {resp['response_time_ms']} ms")
                print("-" * 50)
    except Exception as e:
        print(f"Error getting responses: {e}")

def print_stats():
    """Print statistics from the database"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get overall statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_responses,
                    SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_responses,
                    SUM(CASE WHEN response_type = 'not_sure' THEN 1 ELSE 0 END) as not_sure_responses,
                    SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) as incorrect_responses,
                    MAX(streak) as max_streak,
                    AVG(response_time_ms) as avg_response_time_ms
                FROM responses
            """)
            
            stats = cursor.fetchone()
            
            print("\n=== Response Statistics ===")
            print(f"Total Responses: {stats['total_responses']}")
            print(f"Correct Responses: {stats['correct_responses']} ({get_percentage(stats['correct_responses'], stats['total_responses'])}%)")
            print(f"Not Sure Responses: {stats['not_sure_responses']} ({get_percentage(stats['not_sure_responses'], stats['total_responses'])}%)")
            print(f"Incorrect Responses: {stats['incorrect_responses']} ({get_percentage(stats['incorrect_responses'], stats['total_responses'])}%)")
            print(f"Max Streak: {stats['max_streak']}")
            print(f"Average Response Time: {stats['avg_response_time_ms']:.1f} ms")
            
            # Get SDG distribution
            print("\n=== SDG Selection Distribution ===")
            cursor.execute("""
                SELECT 
                    selected_sdg, 
                    COUNT(*) as count 
                FROM responses 
                WHERE selected_sdg IS NOT NULL
                GROUP BY selected_sdg
                ORDER BY selected_sdg
            """)
            
            sdg_counts = cursor.fetchall()
            
            # Get "None" selections count
            cursor.execute("""
                SELECT COUNT(*) as none_count
                FROM responses
                WHERE selected_sdg IS NULL
            """)
            
            none_count = cursor.fetchone()['none_count']
            
            # Display the counts
            if none_count:
                print(f"None: {none_count}")
            
            for row in sdg_counts:
                print(f"SDG {row['selected_sdg']}: {row['count']}")
                
    except Exception as e:
        print(f"Error getting statistics: {e}")

def get_percentage(part, total):
    """Calculate and format a percentage"""
    if not total:
        return "0.0"
    return f"{(part / total) * 100:.1f}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "sessions":
            print_sessions()
        elif command == "responses":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            print_responses(limit)
        elif command == "stats":
            print_stats()
        else:
            print("Unknown command. Use: sessions, responses, or stats")
    else:
        print_sessions()
        print_responses(5)
        print_stats()
