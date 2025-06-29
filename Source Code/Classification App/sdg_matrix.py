#!/usr/bin/env python3
"""
SDG Matrix Generator

This script generates a 16x9 binary matrix representing whether a given text matches
the SDG queries from different models (auckland, aurora, elsevier, sdgo, sdsn, siris, sdgbert, mistral, ensemble).

The 'ensemble' column represents predictions from XGBoost models trained on the outputs of the other models.

This implementation is completely standalone with no dependencies on query_search.py.
"""

import sys
import os
import re
import io
import ast
import time
import uuid
import pickle
import logging
import warnings
import urllib3
import json
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
from elasticsearch import Elasticsearch, helpers
from lark import Lark, Token, Transformer
from mistralai.client import MistralClient

# Try to import tqdm for progress bars, provide fallback if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Constants

# Constants
MODELS = ['auckland', 'aurora', 'elsevier', 'sdgo', 'sdsn', 'siris', 'sdgbert', 'mistral', 'ensemble']
NUM_SDGS = 16
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# SDG names for display
SDG_NAMES = [
    "No poverty",
    "Zero hunger",
    "Good health and well-being",
    "Quality education",
    "Gender equality",
    "Clean water and sanitation",
    "Affordable and clean energy",
    "Decent work and economic growth",
    "Industry, innovation and infrastructure",
    "Reduced inequalities",
    "Sustainable cities and communities",
    "Responsible consumption and production",
    "Climate action",
    "Life below water",
    "Life on land",
    "Peace, justice and strong institutions"
]

# Test mode flag - set to True to skip external API calls
TEST_MODE = os.environ.get('SDG_TEST_MODE', 'False').lower() in ('true', 't', '1')

# Verbose mode flag - set to False to reduce terminal output
VERBOSE_MODE = os.environ.get('SDG_VERBOSE_MODE', 'False').lower() in ('true', 't', '1')

# Mode for processing status updates - only model progress indicators
PROCESSING_UPDATES = True

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Helper function to ensure flushed output
def print_flush(*args, **kwargs):
    """Print and immediately flush the output for real-time browser updates"""
    # Detect if we're running in a browser environment
    browser_env = os.environ.get('BROWSER_TERMINAL', 'False').lower() in ('true', 't', '1')
    
    # Print the text
    print(*args, **kwargs)
    
    # Force immediate flush
    sys.stdout.flush()
    
    # For browser environments, add a small delay for updates to render
    if browser_env:
        time.sleep(0.02)  # Small delay to ensure browser updates

# AST Node classes for evaluation
class Node:
    def __repr__(self):
        attrs = [f"{k}={repr(v)}" for k, v in vars(self).items()]
        return f"{self.__class__.__name__}({', '.join(attrs)})"

class Term(Node):
    def __init__(self, value):
        self.value = value

class BinaryOp(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

class UnaryOp(Node):
    def __init__(self, operand):
        self.operand = operand

class And(BinaryOp): pass
class Or(BinaryOp): pass
class Not(UnaryOp): pass

class Proximity(BinaryOp):
    def __init__(self, left, right, distance):
        super().__init__(left, right)
        self.distance = distance

# Define Lark grammar for parsing Scopus-style queries
query_parser = Lark(r"""
    start: expr
    
    expr: or_expr
    
    or_expr: and_expr
           | or_expr "OR" and_expr     -> or_op
    
    and_expr: prox_expr
            | and_expr "AND" prox_expr -> and_op
    
    prox_expr: not_expr
             | prox_expr proximity_op not_expr -> prox_op
    
    not_expr: "NOT" primary            -> not_op
            | primary
    
    primary: atom
           | "(" expr ")"              -> group
           | "((" expr "))"            -> double_group  // Handle double parentheses common in aurora queries
    
    proximity_op: "W/" NUMBER          -> prox_no_space
                | "W/ " NUMBER         -> prox_with_space
    
    atom: WORD                         -> word
        | QUOTED_STRING                -> quoted
        | CURLY_PHRASE                 -> curly
        | ANGLE_PHRASE                 -> angle
    
    CURLY_PHRASE: "{" /[^}]*/ "}"
    ANGLE_PHRASE: "<" /[^>]*/ ">"
    QUOTED_STRING: "\"" /[^"]*/ "\""
    WORD: /[a-zA-Z0-9\-*?]+/
    NUMBER: /\d+/
    
    %import common.WS
    %ignore WS
""", start='start')

# Transformer class for converting Lark parse tree to AST
class ASTTransformer(Transformer):
    def start(self, items):
        return items[0]
        
    def expr(self, items):
        return items[0]
        
    def group(self, items):
        return items[0]
        
    def double_group(self, items):
        return items[0]
        
    def or_op(self, items):
        return Or(items[0], items[1])
        
    def and_op(self, items):
        return And(items[0], items[1])
        
    def not_op(self, items):
        return Not(items[0])
        
    def prox_op(self, items):
        left = items[0]
        proximity_op = items[1]
        right = items[2]
        
        # Extract distance from proximity operator
        if isinstance(proximity_op, list) and len(proximity_op) > 0:
            distance = int(proximity_op[0])
        else:
            distance = 1  # Default distance
            
        return Proximity(left, right, distance)
        
    def prox_no_space(self, items):
        return [items[0].value]
        
    def prox_with_space(self, items):
        return [items[0].value]
        
    def or_expr(self, items):
        return items[0]
        
    def and_expr(self, items):
        return items[0]
        
    def prox_expr(self, items):
        return items[0]
        
    def not_expr(self, items):
        return items[0]
        
    def primary(self, items):
        return items[0]
        
    def word(self, items):
        return Term(items[0].value)
        
    def quoted(self, items):
        return Term(items[0].value)
        
    def curly(self, items):
        return Term(items[0].value)
        
    def angle(self, items):
        return Term(items[0].value)


def check_query_in_string(text_content, query_input):
    """
    Checks if a query is present in the given string with no terminal output.
    This is a standalone implementation that doesn't depend on query_search.py.
    
    Args:
        text_content (str): The text content to search in.
        query_input (str): Either the path to a .txt file containing the query OR the query string itself.
        
    Returns:
        bool: True if the query matches the text content, False otherwise.
    """
    # Save all original logging configurations
    original_loggers = {}
    loggers_to_silence = [
        'elasticsearch', 
        'elasticsearch.trace', 
        'urllib3', 
        'urllib3.connectionpool',
        'elastic_transport.transport',
        'elastic_transport',
        '__main__'
    ]
    
    # Completely suppress all output
    null_output = io.StringIO()
    
    try:
        # Silence all relevant loggers
        for logger_name in loggers_to_silence:
            logger = logging.getLogger(logger_name)
            original_loggers[logger_name] = (logger.level, logger.propagate, logger.handlers.copy() if logger.handlers else [])
            logger.setLevel(logging.CRITICAL + 100)  # Set to higher than CRITICAL
            logger.propagate = False
            logger.handlers = []
            logger.addHandler(logging.NullHandler())
        
        # Disable deprecation warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # Disable urllib3 warnings
        import urllib3
        urllib3.disable_warnings()
        
        # Disable Elasticsearch tracer
        es_tracer = logging.getLogger('elasticsearch.trace')
        es_tracer.propagate = False
        es_tracer.setLevel(logging.CRITICAL + 100)
        
        # Redirect stdout and stderr
        with redirect_stdout(null_output), redirect_stderr(null_output):
            # Create a quiet Elasticsearch client
            es = Elasticsearch(
                ["http://localhost:9200"],
                http_auth=("elastic", "1YKSRDyc"),
                request_timeout=30,
                verify_certs=False
            )
            
            # Determine if query_input is a file path or a direct query string
            if os.path.exists(query_input) and query_input.endswith('.txt'):
                # Read query from file
                try:
                    with open(query_input, 'r', encoding='utf-8') as f:
                        query_string = f.read().strip()
                    is_complex_query = len(query_string) > 500 or "aurora" in query_input.lower()
                except Exception:
                    return False
            else:
                # Use the string directly as query
                query_string = query_input.strip()
                is_complex_query = len(query_string) > 500
            
            # Create a temporary index name
            index_name = f"temp_search_{int(time.time())}"
            
            try:
                # Create the index
                settings = {
                    "settings": {
                        "analysis": {
                            "analyzer": {
                                "custom_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "filter": ["lowercase", "asciifolding"],
                                    "char_filter": ["html_strip"]
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "content": {
                                "type": "text",
                                "analyzer": "custom_analyzer",
                                "search_analyzer": "custom_analyzer"
                            },
                            "segment_id": {"type": "keyword"}
                        }
                    }
                }
                
                # Delete if exists and create index
                if es.indices.exists(index=index_name):
                    es.indices.delete(index=index_name)
                es.indices.create(index=index_name, body=settings)
                
                # Index the text
                doc = {
                    '_index': index_name,
                    '_id': str(uuid.uuid4()),
                    'content': text_content,
                    'segment_id': "segment_1"
                }
                helpers.bulk(es, [doc])
                es.indices.refresh(index=index_name)
                
                # Parse query and convert to Elasticsearch DSL
                try:
                    # Parse the query using Lark parser and transform to AST
                    ast = parse_query(query_string)
                    
                    # Decide based on complexity
                    if is_complex_query:
                        # For complex queries, use the simplified approach with term extraction
                        important_terms = extract_all_terms(ast)
                        
                        # Build a simple query with OR clauses
                        query_dsl = {
                            "bool": {
                                "should": []
                            }
                        }
                        
                        # Add terms as separate should clauses
                        for term in important_terms:
                            if term.strip():
                                if term.startswith('"') and term.endswith('"'):
                                    # Handle phrases
                                    content = term[1:-1].strip()
                                    query_dsl["bool"]["should"].append(
                                        {"match_phrase": {"content": content}}
                                    )
                                elif '*' in term:
                                    # Handle wildcards
                                    query_dsl["bool"]["should"].append(
                                        {"wildcard": {"content": term}}
                                    )
                                else:
                                    # Handle regular terms
                                    query_dsl["bool"]["should"].append(
                                        {"match": {"content": term}}
                                    )
                    else:
                        # For simpler queries, use the full AST to DSL conversion
                        query_dsl = ast_to_elasticsearch_dsl(ast)
                except Exception:
                    # Fallback to simple query_string if parsing completely fails
                    query_dsl = {
                        "query_string": {
                            "query": query_string,
                            "default_field": "content", 
                            "default_operator": "AND"
                        }
                    }
                
                # Execute search
                search_body = {"query": query_dsl}
                results = es.search(index=index_name, body=search_body)
                
                # Extract match status
                total = results.get("hits", {}).get("total", {})
                if isinstance(total, dict):  # Elasticsearch 7.x
                    total_hits = total.get("value", 0)
                else:  # Elasticsearch <7.0
                    total_hits = total
                
                return total_hits > 0
                
            except Exception:
                return False
            
            finally:
                # Clean up: Delete the temporary index
                if es.indices.exists(index=index_name):
                    try:
                        es.indices.delete(index=index_name)
                    except:
                        pass  # Ignore cleanup errors
    
    except Exception:
        return False
    
    finally:
        # Restore original logging configurations
        for logger_name, (level, propagate, handlers) in original_loggers.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            logger.propagate = propagate
            logger.handlers = handlers


# Function to preprocess the query before parsing
def preprocess_query(query):
    """Clean and normalize query for parsing"""
    if not query or not query.strip():
        return ""
    
    # Clean up whitespace
    query = ' '.join(query.split())
    
    # Replace "AND NOT" with just "NOT" to treat them the same
    query = re.sub(r'\bAND\s+NOT\b', 'NOT', query, flags=re.IGNORECASE)
    
    # Handle double parentheses which are common in Aurora queries
    query = query.replace("))", ") )").replace("((", "( (")
    
    # Fix "x NOT y" -> "x AND (NOT y)"
    i = 0
    result = ""
    in_quotes = False
    in_curly = False
    in_angle = False
    
    while i < len(query):
        char = query[i]
        
        # Track quotes and brackets
        if char == '"' and not in_curly and not in_angle:
            in_quotes = not in_quotes
            result += char
        elif char == '{' and not in_quotes and not in_angle:
            in_curly = True
            result += char
        elif char == '}' and in_curly:
            in_curly = False
            result += char
        elif char == '<' and not in_quotes and not in_curly:
            in_angle = True
            result += char
        elif char == '>' and in_angle:
            in_angle = False
            result += char
        # Handle binary NOT (not in quotes/brackets)
        elif (not in_quotes and not in_curly and not in_angle and
              i > 0 and query[i-1] == ' ' and 
              i + 3 < len(query) and query[i:i+3] == 'NOT' and 
              (i + 3 == len(query) or query[i+3] == ' ')):
            result += "AND (NOT"
            i += 2  # Skip the 'NO' in 'NOT', will add the 'T' next iteration
        else:
            result += char
        i += 1
    
    # Balance parentheses
    open_count = result.count('(')
    close_count = result.count(')')
    
    if open_count > close_count:
        result += ')' * (open_count - close_count)
    elif close_count > open_count:
        result = '(' * (close_count - open_count) + result
    
    return result

# Parse the query using Lark and transform to AST
def parse_query(query):
    """Parse a query string and return an AST"""
    try:
        # First preprocess the query
        processed_query = preprocess_query(query)
        
        # Parse with Lark
        try:
            tree = query_parser.parse(processed_query)
            
            # Transform to AST using the custom transformer
            transformer = ASTTransformer()
            ast = transformer.transform(tree)
            
            return ast
        except Exception as e:
            # If the query is too complex, try to break it down into chunks
            # Find the highest level OR operators and split at them
            # This helps with extremely long queries like aurora_13.txt
            chunks = re.split(r'\s+OR\s+', processed_query, flags=re.IGNORECASE)
            
            if len(chunks) <= 1:
                # If no top-level OR, try breaking at AND
                chunks = re.split(r'\s+AND\s+', processed_query, flags=re.IGNORECASE)
            
            # Parse each chunk separately
            if len(chunks) > 1:
                chunk_asts = []
                for chunk in chunks:
                    # Recursively parse each chunk
                    try:
                        chunk_ast = parse_query(chunk)
                        chunk_asts.append(chunk_ast)
                    except Exception:
                        # If a chunk fails, just represent it as a Term
                        chunk_asts.append(Term(chunk))
                
                # Combine the chunks with OR or AND operators
                root = chunk_asts[0]
                for chunk_ast in chunk_asts[1:]:
                    if 'OR' in processed_query:
                        root = Or(root, chunk_ast)
                    else:
                        root = And(root, chunk_ast)
                
                return root
            
            # If we can't split into chunks, just use a Term node
            raise e
            
    except Exception as e:
        # Return a simple term as fallback for the entire query
        return Term(query)

# Convert AST to Elasticsearch query DSL
def ast_to_elasticsearch_dsl(node):
    """Convert AST to Elasticsearch Query DSL structure"""
    if isinstance(node, Term):
        term = node.value.strip()
        
        # Handle different bracket styles
        if len(term) >= 2:
            if term[0] == '{' and term[-1] == '}':
                # Exact match with curly braces
                content = term[1:-1].strip()
                return {"match_phrase": {"content": content}}
                
            elif term[0] == '<' and term[-1] == '>':
                # Exact match with angle brackets
                content = term[1:-1].strip()
                return {"match_phrase": {"content": content}}
                
            elif term[0] == '"' and term[-1] == '"':
                # Exact match with quotes
                content = term[1:-1].strip()
                return {"match_phrase": {"content": content}}
        
        # Handle wildcards appropriately
        if '*' in term or '?' in term:
            return {"wildcard": {"content": term}}
            
        # Default term query
        return {"match": {"content": term}}
    
    elif isinstance(node, And):
        left = ast_to_elasticsearch_dsl(node.left)
        right = ast_to_elasticsearch_dsl(node.right)
        return {"bool": {"must": [left, right]}}
    
    elif isinstance(node, Or):
        left = ast_to_elasticsearch_dsl(node.left)
        right = ast_to_elasticsearch_dsl(node.right)
        return {"bool": {"should": [left, right]}}
    
    elif isinstance(node, Not):
        operand = ast_to_elasticsearch_dsl(node.operand)
        return {"bool": {"must_not": operand}}
    
    elif isinstance(node, Proximity):
        # For proximity searches, we need to use span_near
        left_term = extract_term_value(node.left)
        right_term = extract_term_value(node.right)
        
        if left_term and right_term:
            return {
                "span_near": {
                    "clauses": [
                        {"span_term": {"content": left_term}},
                        {"span_term": {"content": right_term}}
                    ],
                    "slop": node.distance,
                    "in_order": False
                }
            }
        else:
            # Fallback if we couldn't extract clean terms
            return {
                "bool": {
                    "must": [
                        ast_to_elasticsearch_dsl(node.left),
                        ast_to_elasticsearch_dsl(node.right)
                    ]
                }
            }
    
    else:
        return {"match_all": {}}

def extract_term_value(node):
    """Extract clean term value from an AST node for proximity searches"""
    if isinstance(node, Term):
        term = node.value.strip()
        
        # Clean terms from various bracket styles
        if len(term) >= 2:
            if term[0] == '{' and term[-1] == '}':
                return term[1:-1].strip()
            elif term[0] == '<' and term[-1] == '>':
                return term[1:-1].strip()
            elif term[0] == '"' and term[-1] == '"':
                return term[1:-1].strip()
                
        return term
    return None

def extract_all_terms(node, terms=None):
    """
    Extract all important terms from an AST node recursively.
    This is used for complex queries like aurora_13.txt to create a flattened query structure.
    """
    if terms is None:
        terms = []
    
    if isinstance(node, Term):
        term = node.value.strip()
        
        # Clean and normalize terms
        if len(term) >= 2:
            if term[0] == '{' and term[-1] == '}':
                # Convert curly braces to quotes
                term = f'"{term[1:-1].strip()}"'
            elif term[0] == '<' and term[-1] == '>':
                # Convert angle brackets to quotes
                term = f'"{term[1:-1].strip()}"'
            # Keep quotes as is
        
        # Add the term if it's not already in the list
        if term and term not in terms:
            terms.append(term)
            
    elif isinstance(node, And) or isinstance(node, Or):
        # Process both sides of binary operators
        extract_all_terms(node.left, terms)
        extract_all_terms(node.right, terms)
        
    elif isinstance(node, Not):
        # Just collect the terms from the operand
        extract_all_terms(node.operand, terms)
        
    elif isinstance(node, Proximity):
        # For proximity, extract both sides and also create a combined term
        extract_all_terms(node.left, terms)
        extract_all_terms(node.right, terms)
        
        # Try to create a combined term for better proximity matching
        try:
            left_term = extract_term_value(node.left)
            right_term = extract_term_value(node.right)
            
            if left_term and right_term:
                combined = f'"{left_term} {right_term}"~{node.distance}'
                if combined not in terms:
                    terms.append(combined)
        except:
            pass
    
    return terms


# SDGBert classification functionality
def classify_with_sdgbert(text):
    """
    Classify text using SDGBert model without dependency on sdgbert.py
    
    Args:
        text (str): The text to classify
        
    Returns:
        list: A list of 16 scores, one for each SDG
    """
    try:
        # We're using a try-except block because the transformers library might not be installed
        from transformers import pipeline
        import numpy as np
        
        # Initialize the model (will download it if not already present)
        # Using silent=True to reduce output noise
        model = pipeline(
            "text-classification", 
            model="sadickam/sdg-classification-bert", 
            tokenizer="sadickam/sdg-classification-bert", 
            top_k=16, 
            truncation=True
        )
        
        # Get the predictions
        predictions = model(text)
        pred_list = [0] * 16  # Initialize a list to hold the scores for each SDG

        for pred in predictions[0]:
            label = pred['label'].lower()
            sdg = int(label[3:])
            pred_list[sdg-1] = pred['score']
        
        return pred_list
        
    except ImportError:
        # If transformers is not installed, return all zeros
        print("Warning: transformers library not found. SDGBert classification not available.", file=sys.stderr)
        print("Install with: pip install transformers torch", file=sys.stderr)
        return [0] * 16
    except Exception as e:
        # Handle any other errors
        print(f"Error with SDGBert classification: {e}", file=sys.stderr)
        return [0] * 16


def generate_sdg_matrix(text_input):
    """
    Generate a 16x9 binary matrix for SDGs x query models.
    
    Args:
        text_input (str): The input text to classify against SDG queries.
        
    Returns:
        numpy.ndarray: A 16x9 binary matrix where rows represent SDGs (1-16) 
                       and columns represent query models in alphabetical order 
                       (auckland, aurora, elsevier, sdgo, sdsn, siris, sdgbert, mistral, ensemble).
                       The 'ensemble' column represents predictions from XGBoost models.
    """
    # Initialize the matrix with zeros
    matrix = np.zeros((NUM_SDGS, len(MODELS)), dtype=int)
    
    # Create progress description
    print_flush("Starting SDG analysis...\n")
    
    # For each model and SDG, check if the text matches the query
    active_models = [m for m in MODELS if m != 'ensemble']
    for model_idx, model_name in enumerate(MODELS):
        if model_name == 'ensemble':
            # Skip ensemble processing for now - we'll handle it after all other models are processed
            continue
        elif model_name == 'sdgbert':
            # Handle SDGBert model separately
            print_flush(f"[07/10] Processing {model_name.upper()} model...")
            
            try:
                # Skip SDGBert if in test mode
                if TEST_MODE:
                    print("Test mode: Skipping SDGBert model", file=sys.stderr) if VERBOSE_MODE else None
                    print_flush("SDGBert model: SKIPPED (test mode)\n")
                    continue
                    
                try:
                    # Get predictions from SDGBert
                    sdgbert_scores = classify_with_sdgbert(text_input)
                    
                    # Binarize the scores: values > 0.95 become 1, else 0
                    match_count = 0
                    for sdg_idx, score in enumerate(sdgbert_scores):
                        if score > 0.95:
                            matrix[sdg_idx, model_idx] = 1
                            match_count += 1
                    
                    # No output line for SDGBert model completion
                except Exception as e:
                    if VERBOSE_MODE:
                        print(f"Error processing sdgbert: {e}", file=sys.stderr)
                    print_flush(f"SDGBert model: ERROR\n")
            except Exception as e:
                if VERBOSE_MODE:
                    print(f"Error in SDGBert model handling: {e}", file=sys.stderr)
                print_flush(f"SDGBert model: ERROR (general error)\n")
        elif model_name == 'mistral':
            # Handle Mistral model
            print_flush(f"[08/10] Processing {model_name.upper()} model...")
            
            try:
                # Skip Mistral API calls if in test mode
                if TEST_MODE:
                    if VERBOSE_MODE:
                        print("Test mode: Skipping Mistral API calls", file=sys.stderr)
                    print_flush("Mistral model: SKIPPED (test mode)\n")
                    continue
                    
                # Import the mistral module
                api_key = "GET YOURS AT https://console.mistral.ai/"
                model = "mistral-small-latest"

                try:
                    client = MistralClient(api_key=api_key)
                except Exception as e:
                    if VERBOSE_MODE:
                        print(f"Failed to initialize Mistral client: {e}", file=sys.stderr)
                    print_flush(f"Mistral model: ERROR (client initialization failed)\n")
                    continue  # Skip Mistral processing

                def describe_abstract(abstract):
                    try:
                        SYSTEM_PROMPT = """Describe the application of the invention described in the patent abstract, including any environmental, social and economic aspects. The summary should be in a single sentence, and should not exceed 50 words. Do not mention that the summary concerns an invention or patent, and avoid technical details and demonstrative pronouns. The summary should be clear and concise, and should be suitable for a general audience."""
                        USER_PROMPT = abstract
                        messages = [
                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT
                            },
                            {
                                "role": "user",
                                "content": USER_PROMPT
                            },
                        ]
                        chat_response = client.chat(
                            model=model,
                            messages=messages,
                        )
                        return chat_response.choices[0].message.content
                    except Exception as e:
                        print(f"Error in describe_abstract: {e}", file=sys.stderr)
                        return abstract[:100]  # Return truncated abstract as fallback

                def classify_mistral(abstract):
                    SYSTEM_PROMPT = """Report a maximum of six SDGs that are most relevant to the text. See below for SDG descriptions.
                    Use Python list format and report only the SDG number(s). If you detect no SDGs, report [0].
                    SDG 1: End poverty in all its forms everywhere
                    SDG 2: End hunger, achieve food security and improved nutrition and promote sustainable agriculture
                    SDG 3: Ensure healthy lives and promote well-being for all at all ages
                    SDG 4: Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all
                    SDG 5: Achieve gender equality and empower all women and girls
                    SDG 6: Ensure availability and sustainable management of water and sanitation for all
                    SDG 7: Ensure access to affordable, reliable, sustainable and modern energy for all
                    SDG 8: Promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all
                    SDG 9: Build resilient infrastructure, promote inclusive and sustainable industrialization and foster innovation
                    SDG 10: Reduce inequality within and among countries
                    SDG 11: Make cities and human settlements inclusive, safe, resilient and sustainable
                    SDG 12: Ensure sustainable consumption and production patterns
                    SDG 13: Take urgent action to combat climate change and its impacts
                    SDG 14: Conserve and sustainably use the oceans, seas and marine resources for sustainable development
                    SDG 15: Protect, restore and promote sustainable use of terrestrial ecosystems, sustainably manage forests, combat desertification, halt and reverse land degradation, and halt biodiversity loss
                    SDG 16: Promote peaceful and inclusive societies for sustainable development, provide access to justice for all and build effective, accountable and inclusive institutions at all levels"""
                    USER_PROMPT = describe_abstract(abstract)
                    messages = [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": USER_PROMPT
                        },
                    ]
                    chat_response = client.chat(
                        model=model,
                        messages=messages,
                    )
                    # Extract relevant SDG numbers from the response
                    response_text = chat_response.choices[0].message.content
                    try:
                        # Try to evaluate the response text as a Python list
                        import ast
                        sdg_nums = ast.literal_eval(response_text)
                        if not isinstance(sdg_nums, list):
                            sdg_nums = [0]  # Default if not a list
                    except:
                        # Fall back to regex pattern matching to find numbers
                        import re
                        sdg_nums = [int(num) for num in re.findall(r'\b([1-9]|1[0-6])\b', response_text)]
                        if not sdg_nums:
                            sdg_nums = [0]  # Default if no numbers found
                    
                    return sdg_nums
                
                try:
                    # Classify using Mistral
                    sdg_list = classify_mistral(text_input)
                    
                    # Convert to binary matrix: if the SDG is in the list, set to 1
                    for sdg_num in sdg_list:
                        if 1 <= sdg_num <= NUM_SDGS:  # Ensure valid SDG number
                            matrix[sdg_num - 1, model_idx] = 1
                    
                    # No output line for Mistral model completion
                except Exception as e:
                    if VERBOSE_MODE:
                        print(f"Error in mistral processing: {e}", file=sys.stderr)
                    print_flush(f"Mistral model: ERROR\n")
                    # Fall back to zeros for this model column if there's an error
                    pass
            except ImportError:
                if VERBOSE_MODE:
                    print("Warning: mistral.py module not found. Mistral classification not available.", file=sys.stderr)
                print_flush(f"Mistral model: ERROR (module not found)\n")
            except Exception as e:
                if VERBOSE_MODE:
                    print(f"Error processing mistral: {e}", file=sys.stderr)
                print_flush(f"Mistral model: ERROR (general error)\n")
        else:
            # Handle file-based models (auckland, aurora, etc.)
            model_counter = model_idx + 1  # Calculate appropriate counter (1-6)
            print_flush(f"[{model_counter:02d}/10] Processing {model_name.upper()} model...")
            sys.stdout.flush()  # Force immediate flush
            
            match_count = 0
            
            for sdg_num in range(1, NUM_SDGS + 1):
                # Format the SDG number with leading zero if needed
                sdg_str = f"{sdg_num:02d}"
                
                # Path to the query file for this model and SDG
                query_file = os.path.join(SCRIPT_DIR, model_name, f"{model_name}_{sdg_str}.txt")
                
                # Check if the query is in the text
                if os.path.exists(query_file):
                    try:
                        matches = check_query_in_string(text_input, query_file)
                        if matches:
                            matrix[sdg_num - 1, model_idx] = 1
                            match_count += 1
                    except Exception as e:
                        if VERBOSE_MODE:
                            print(f"Error processing {model_name} SDG{sdg_str}: {e}", file=sys.stderr)
            
            matched_sdgs = [f"SDG{i+1:02d}" for i in range(NUM_SDGS) if matrix[i, model_idx] == 1]
            # No output line for model completion
            if matched_sdgs and VERBOSE_MODE:
                print_flush(f"  Matched SDGs: {', '.join(matched_sdgs)}\n")
            else:
                print_flush("\n")
    
    # Process ensemble models after all other models have been processed
    print_flush(f"[09/10] Processing ENSEMBLE model...")
    sys.stdout.flush()  # Force immediate flush
    
    # Don't show "Processing" message for ensemble in processing updates mode
    match_count = 0
    try:
        import pickle
        import xgboost
        ensemble_idx = MODELS.index('ensemble')
        
        # For each SDG, load its specific ensemble model and make a prediction
        for sdg_idx in range(NUM_SDGS):
            sdg_num = sdg_idx + 1
            sdg_str = f"{sdg_num:02d}"
            
            # Path to the model file
            model_path = os.path.join(SCRIPT_DIR, 'ensemble', f'sdg{sdg_str}_testenv_model.pkl')
            
            if os.path.exists(model_path):
                try:
                    # Load the XGBoost model
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Get the input features for this SDG (all other model predictions)
                    # Exclude the ensemble column and extract just this SDG's row
                    input_features = matrix[sdg_idx, :ensemble_idx].reshape(1, -1)
                    
                    # Make prediction
                    prediction = model.predict(input_features)[0]
                    
                    # Set the ensemble model's prediction
                    matrix[sdg_idx, ensemble_idx] = prediction
                    if prediction == 1:
                        match_count += 1
                except Exception as e:
                    if VERBOSE_MODE:
                        print(f"Error using ensemble model for SDG{sdg_str}: {e}", file=sys.stderr)
        
        # Generate list of matched SDGs from ensemble (still needed for later use)
        matched_sdgs = [f"SDG{i+1:02d}" for i in range(NUM_SDGS) if matrix[i, ensemble_idx] == 1]
        # No output line for ensemble model completion
    except ImportError:
        if VERBOSE_MODE:
            print("Error: xgboost library not found. Ensemble classification not available.", file=sys.stderr)
        print_flush("Ensemble model: ERROR (library not found)\n")
    except Exception as e:
        if VERBOSE_MODE:
            print(f"Error processing ensemble models: {e}", file=sys.stderr)
        print_flush("Ensemble model: ERROR\n")
    
    # Combine results from different models based on custom rules
    print_flush("\n[10/10] Combining model results...")
    sys.stdout.flush()  # Force immediate display
    
    # Create a final verdict for each SDG
    final_sdgs = []
    
    # Get indices for the models we need
    ensemble_idx = MODELS.index('ensemble')
    mistral_idx = MODELS.index('mistral')
    sdgo_idx = MODELS.index('sdgo')
    
    for sdg_idx in range(NUM_SDGS):
        sdg_num = sdg_idx + 1
        
        # Different logic for different SDGs
        if sdg_num == 1 or sdg_num == 5:  # SDG 1 and 5 based on Mistral
            if matrix[sdg_idx, mistral_idx] == 1:
                final_sdgs.append(sdg_num)
        elif sdg_num == 10:  # SDG 10 based on SDGO
            if matrix[sdg_idx, sdgo_idx] == 1:
                final_sdgs.append(sdg_num)
        else:  # All other SDGs based on Ensemble
            if matrix[sdg_idx, ensemble_idx] == 1:
                final_sdgs.append(sdg_num)
    
    # Display the final results with SDG names
    if final_sdgs:
        print_flush("\n\nSDGs DETECTED:")
        for sdg_num in final_sdgs:
            sdg_idx = sdg_num - 1
            sdg_label = f"SDG{sdg_num:02d}"
            print_flush(f"{sdg_label}: {SDG_NAMES[sdg_idx]}")
    else:
        print_flush("\n\nNO SDGs DETECTED")
    
    return matrix


def matrix_to_json(matrix):
    """
    Convert the matrix to a JSON-friendly dictionary.
    
    The matrix has dimensions 16x9, representing the 16 SDGs and 9 models
    (auckland, aurora, elsevier, sdgo, sdsn, siris, sdgbert, mistral, ensemble).
    """
    result = {
        "models": MODELS,
        "sdgs": [f"SDG{i+1:02d}" for i in range(NUM_SDGS)],
        "matrix": matrix.tolist()
    }
    
    # In verbose mode, print the JSON representation
    if VERBOSE_MODE:
        print("\nJSON Representation:")
        print(json.dumps(result, indent=2))
        
    return result


def pretty_print_matrix(matrix):
    """Print the matrix in a human-readable format."""
    if not VERBOSE_MODE:
        # In non-verbose mode, don't print the matrix
        return
        
    print("\nSDG Matrix Results:")
    print("=" * 90)  # Increased width to accommodate the extra column
    header = "SDG".ljust(7) + " " + " ".join(model.ljust(8) for model in MODELS)
    print(header)
    print("-" * 90)  # Increased width to accommodate the extra column
    
    for sdg_num in range(1, NUM_SDGS + 1):
        row = f"SDG {sdg_num:02d}".ljust(7) + " "
        for model_idx in range(len(MODELS)):
            value = matrix[sdg_num - 1, model_idx]
            row += f"{value}".center(8)
        print(row)
    print("=" * 90)  # Increased width to accommodate the extra column


if __name__ == "__main__":
    # Parse command line arguments
    input_file = None
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    try:
        # Get text from file or stdin
        if input_file:
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            if VERBOSE_MODE:
                print("Enter text to classify (Ctrl+D to finish):")
            text = sys.stdin.read()
        
        # Generate and process the matrix
        matrix = generate_sdg_matrix(text)
        
        # Only show the matrix and JSON in verbose mode
        if VERBOSE_MODE:
            pretty_print_matrix(matrix)
            print("\nJSON representation:")
            print(json.dumps(matrix_to_json(matrix), indent=2))
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
