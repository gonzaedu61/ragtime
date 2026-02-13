from typing import List, Dict, Any, Optional
import re
import os
import glob

# ---------------------------------------------------------
def expand_files_pattern(dir, pattern) -> List[str]:
    """
    Resolves the given pattern into a list of file paths.
    Supports:
    - exact filename
    - wildcard patterns (*.pdf)
    - regex (if document_name starts with 're:')
    """

    # Case 1: regex pattern
    if pattern.startswith("re:"):
        regex = re.compile(pattern[3:])
        files = [
            os.path.join(dir, f)
            for f in os.listdir(dir)
            if regex.match(f)
        ]
        return files

    # Case 2: wildcard pattern
    wildcard_path = os.path.join(dir, pattern)
    files = glob.glob(wildcard_path)
    if files:
        return files

    # Case 3: exact file
    exact_path = os.path.join(dir, pattern)
    if os.path.exists(exact_path):
        return [exact_path]

    raise FileNotFoundError(f"No files match: {pattern}")
