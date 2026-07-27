import ast
import re
import os

__all__ = [
    "SecurityViolationError",
    "validate_safe_python_code",
    "validate_read_only_sql",
    "extract_text_content"
]

class SecurityViolationError(Exception):
    """Custom exception raised when code or queries violate security policies."""
    pass

# --- Configuration & Constants ---

# Can be overridden by environment variables for different deployment tiers
FORBIDDEN_MODULES = set(os.environ.get(
    "FORBIDDEN_PYTHON_MODULES", 
    "os,sys,subprocess,shutil,databricks,pg8000,sqlalchemy,requests,urllib"
).split(","))

FORBIDDEN_FUNCTIONS = {'eval', 'exec', 'open', '__import__'}

# Unified list of destructive SQL patterns using regex word boundaries (\b).
# This prevents blocking valid column names like 'drop_off_time' or 'update_date'.
FORBIDDEN_SQL_MUTATIONS = [
    r'\bdrop\s+table\b',
    r'\bdrop\s+database\b',
    r'\bdelete\s+from\b',
    r'\btruncate\s+table\b',
    r'\bupdate\s+[a-z0-9_]+\s+set\b',
    r'\binsert\s+into\b',
    r'\balter\s+table\b',
    r'\bcreate\s+table\b'
]

# --- Python AST Security Validator ---

class SecurityValidator(ast.NodeVisitor):
    """Walks the Abstract Syntax Tree to identify forbidden imports or function calls."""
    def __init__(self):
        self.violations = []

    def visit_Import(self, node):
        for alias in node.names:
            base_module = alias.name.split('.')[0]
            if base_module in FORBIDDEN_MODULES:
                self.violations.append(f"Importing module '{alias.name}' is strictly forbidden.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            base_module = node.module.split('.')[0]
            if base_module in FORBIDDEN_MODULES:
                self.violations.append(f"Importing from module '{node.module}' is strictly forbidden.")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_FUNCTIONS:
                self.violations.append(f"Calling built-in function '{node.func.id}()' is strictly forbidden.")
        self.generic_visit(node)


# --- Core Validation Functions ---

def validate_safe_python_code(code: str) -> None:
    """
    Validates Python code for security using AST parsing and targeted Regex.
    Raises SecurityViolationError if the code is blocked.
    """
    # 1. AST-based check for imports and dangerous built-ins
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SecurityViolationError(f"SyntaxError in generated code: {str(e)}")
    
    validator = SecurityValidator()
    validator.visit(tree)
    
    if validator.violations:
        raise SecurityViolationError(" | ".join(validator.violations))
        
    # 2. Regex-based check for destructive SQL commands inside strings
    code_lower = code.lower()
    for pattern in FORBIDDEN_SQL_MUTATIONS:
        if re.search(pattern, code_lower):
            raise SecurityViolationError(
                f"Code contains forbidden SQL mutation pattern matching: {pattern}"
            )

def validate_read_only_sql(query: str) -> None:
    """
    Validates that a SQL query is strictly read-only to prevent destructive operations.
    Raises SecurityViolationError if the query is blocked.
    """
    clean_query = query.strip().lower()
    
    # 1. Check valid starting keywords
    if not (clean_query.startswith("select") or clean_query.startswith("with") or clean_query.startswith("explain")):
        raise SecurityViolationError("Only read-only queries starting with SELECT, WITH, or EXPLAIN are allowed.")
    
    # 2. Check unified forbidden mutation patterns
    for pattern in FORBIDDEN_SQL_MUTATIONS:
        if re.search(pattern, clean_query):
            raise SecurityViolationError(
                f"Destructive SQL statement matching pattern '{pattern}' is not permitted."
            )

# --- General Content Extractors ---

def extract_text_content(message) -> str:
    """
    Safely extracts the text string from a ChatCompletionMessage regardless of
    whether content is a plain string or a Gemini-style list of content blocks.
    """
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Grab the first block with type='text' and return its text value
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
    return str(content)