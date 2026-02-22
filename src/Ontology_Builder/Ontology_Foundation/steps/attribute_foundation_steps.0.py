import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utilities import Simple_Progress_Bar


def step_attributes_1_extract(builder):
    """
    Placeholder for attribute extraction.
    You can later expand this to scan entities/relationships/processes.
    """
    step_dir = builder._ensure_step_dir(builder.attributes_dir, 1)

    # For now, nothing to extract — but structure is ready.
    # You can later add:
    # - LLM-based attribute extraction
    # - Derived attributes from relationships
    # - Normalization rules

    return
