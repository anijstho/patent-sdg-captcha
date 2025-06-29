#!/usr/bin/env python3
from flask import Flask, render_template, request, Response, jsonify, stream_with_context
import subprocess
import sys
import os
import time
import threading
import queue
import select
import uuid

# Set environment variable to indicate browser terminal is being used
os.environ['BROWSER_TERMINAL'] = 'True'

app = Flask(__name__)

# Store active processing sessions
active_sessions = {}

def process_input_text_stream(text_input, output_queue):
    """Process text input through sdg_matrix.py and stream results to the queue"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Path to the sdg_matrix.py script
        sdg_matrix_path = os.path.join(script_dir, 'sdg_matrix.py')
        
        # Execute the script and set up pipes for streaming
        process = subprocess.Popen(
            [sys.executable, sdg_matrix_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered for more responsive output
            env={**os.environ, 
                'SDG_TEST_MODE': os.environ.get('SDG_TEST_MODE', 'False'),
                'SDG_VERBOSE_MODE': 'False',  # Always use minimal output in web app
                'SDG_PROCESSING_UPDATES': 'True',  # Always show processing status
                'PYTHONUNBUFFERED': '1'}  # Force Python to be unbuffered
        )
        
        # Send the text input to the script
        process.stdin.write(text_input)
        process.stdin.close()
        
        # Set up non-blocking reads
        for fd in [process.stdout, process.stderr]:
            if fd:
                fd.flush()
        
        # Stream both stdout and stderr
        out_eof = err_eof = False
        stderr_buffer = []
        last_flush_time = time.time()
        
        while not (out_eof and err_eof):
            # Use select to check which streams have data
            ready_to_read, _, _ = select.select(
                [stream for stream in [process.stdout, process.stderr] if stream and not (
                    out_eof and stream == process.stdout or 
                    err_eof and stream == process.stderr
                )], 
                [], [], 0.1
            )
            
            # Force flush at least every 0.2 seconds even if no new data
            current_time = time.time()
            if current_time - last_flush_time > 0.2:
                output_queue.put(('flush', None))
                last_flush_time = current_time
            
            for stream in ready_to_read:
                line = stream.readline()
                if not line:
                    if stream == process.stdout:
                        out_eof = True
                    else:
                        err_eof = True
                    continue
                
                if stream == process.stdout:
                    # Send stdout directly to the client AND flush immediately
                    line_content = line.rstrip()
                    if line_content:  # Only send non-empty lines
                        output_queue.put(('output', line_content))
                        # Send a flush command to ensure browser renders immediately
                        output_queue.put(('flush', None))
                else:
                    # Collect stderr lines for later display
                    stderr_buffer.append(line.rstrip())
            
            # If no streams are ready, sleep a bit to avoid CPU spinning
            if not ready_to_read:
                time.sleep(0.01)
        
        # After stdout is complete, send any accumulated stderr
        if stderr_buffer:
            output_queue.put(('output', '\n\nErrors/Warnings:'))
            for line in stderr_buffer:
                output_queue.put(('output', line))
        
        # Signal completion
        output_queue.put(('complete', None))
        
    except Exception as e:
        output_queue.put(('output', f"Error processing text: {str(e)}"))
        output_queue.put(('complete', None))

def process_input_text(text_input):
    """Process text input synchronously (used for non-streaming API)"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Path to the sdg_matrix.py script
        sdg_matrix_path = os.path.join(script_dir, 'sdg_matrix.py')
        
        # Execute the script and capture its output
        process = subprocess.Popen(
            [sys.executable, sdg_matrix_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, 
                'SDG_TEST_MODE': os.environ.get('SDG_TEST_MODE', 'False'),
                'SDG_VERBOSE_MODE': 'False'}  # Always use minimal output in web app
        )
        
        # Send the text input to the script and get output
        stdout, stderr = process.communicate(input=text_input)
        
        # Combine stdout and stderr for display
        result = stdout
        if stderr:
            result += "\n\nErrors/Warnings:\n" + stderr
        
        return result
    except Exception as e:
        return f"Error processing text: {str(e)}"

@app.route('/', methods=['GET'])
def index():
    """Main page route - only renders the template, no processing"""
    return render_template('index.html', result=None)

@app.route('/process', methods=['POST'])
def process():
    """API endpoint for processing text input via AJAX"""
    text_input = request.form.get('text_input', '')
    
    if not text_input:
        return jsonify({"result": "No input provided"})
    
    # Process the input
    result = process_input_text(text_input)
    
    # Return just the result as JSON
    return jsonify({"result": result})

@app.route('/init_stream', methods=['POST'])
def init_stream():
    """Initialize a streaming session and return a session ID"""
    text_input = request.form.get('text_input', '')
    
    if not text_input:
        return jsonify({"error": "No input provided"})
    
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    
    # Create a queue for inter-thread communication
    output_queue = queue.Queue()
    
    # Start processing in a separate thread
    processing_thread = threading.Thread(
        target=process_input_text_stream, 
        args=(text_input, output_queue)
    )
    processing_thread.daemon = True  # Thread will be terminated when the main thread exits
    processing_thread.start()
    
    # Store the session
    active_sessions[session_id] = {
        'queue': output_queue,
        'thread': processing_thread,
        'created_at': time.time()
    }
    
    # Clean up old sessions (older than 10 minutes)
    current_time = time.time()
    for sid in list(active_sessions.keys()):
        if current_time - active_sessions[sid]['created_at'] > 600:  # 10 minutes
            del active_sessions[sid]
    
    return jsonify({"session_id": session_id})

@app.route('/stream/<session_id>', methods=['GET'])
def stream(session_id):
    """Stream processing results via Server-Sent Events for a given session ID"""
    if session_id not in active_sessions:
        return Response("event: error_message\ndata: Session not found or expired\n\n", 
                       content_type='text/event-stream')
    
    # Get the session's queue
    output_queue = active_sessions[session_id]['queue']
    
    # Set CORS headers to allow streaming from any domain
    # This helps with various browser security policies
    headers = {
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',  # Disable proxy buffering
        'Access-Control-Allow-Origin': '*'
    }
    
    # Generator function to yield SSE events
    def generate():
        try:
            while True:
                try:
                    # Wait for a message from the processing thread
                    msg_type, msg_content = output_queue.get(timeout=30)  # 30-second timeout
                    
                    if msg_type == 'output' and msg_content is not None:
                        # Send output line as an SSE event
                        yield f"event: output\ndata: {msg_content}\n\n"
                    elif msg_type == 'flush':
                        # Send a flush ping to force browser to render
                        yield f"event: flush\ndata: null\n\n"
                    elif msg_type == 'complete':
                        # Signal completion
                        yield "event: complete\ndata: null\n\n"
                        # Clean up the session
                        if session_id in active_sessions:
                            del active_sessions[session_id]
                        break
                    
                    output_queue.task_done()
                    
                except queue.Empty:
                    # Send a keepalive event every 30 seconds
                    yield "event: keepalive\ndata: null\n\n"
        except GeneratorExit:
            # Client disconnected, clean up if needed
            if session_id in active_sessions:
                del active_sessions[session_id]
        except Exception as e:
            # Handle any other errors
            yield f"event: error_message\ndata: {str(e)}\n\n"
            if session_id in active_sessions:
                del active_sessions[session_id]
    
    return Response(stream_with_context(generate()), 
                   content_type='text/event-stream',
                   headers=headers)

if __name__ == '__main__':
    # Set Maximum buffer size to smaller value to send data more quickly
    # Default is usually 8192 bytes, we'll set to 1 for line buffering
    app.config['RESPONSE_BUFFER_SIZE'] = 1
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
