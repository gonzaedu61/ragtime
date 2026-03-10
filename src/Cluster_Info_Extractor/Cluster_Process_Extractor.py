from Cluster_Info_Extractor import Cluster_Info_Extractor
from dotenv import load_dotenv

load_dotenv()


class Cluster_Process_Extractor:
    """
    Wrapper class that encapsulates the cluster process extraction workflow.
    It initializes the LLM, VectorDB, and Cluster_Info_Extractor, and exposes
    a single method to process a hierarchy file.
    """

    def __init__(self,
                 llm,
                 vectordb,
                 output_folder,
                 verbose=True,
                 show_progress_bar=True,
                 max_concurrent_llm_calls=1,
                 log_prompts=True,
                 branch_id=None):
        """
        Parameters are passed directly to Cluster_Info_Extractor.
        """

        # Store parameters
        self.llm = llm
        self.vectordb = vectordb
        self.output_folder = output_folder
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar
        self.max_concurrent_llm_calls = max_concurrent_llm_calls
        self.log_prompts = log_prompts
        self.branch_id = branch_id

        # Constants
        self.INFO_TYPE = "process_b"

        # Prompt templates
        self.LEAF_PROMPT = """
You are analyzing a set of text chunks that belong to the same topic.

TASK:
Identify and describe the the process covering the full input text, with the following information:
- Proces name: A short label (max 6 words). YOU MUST ALWAYS PROVIDE A LABEL.
- Process description: A 4-8 sentences explaining what the process is about
- Process steps: An ordered list of the key steps composing the process. Give them a sequential numeric id (the list index), a short label and a short one-line description.
- IMPORTANT: If the text does not seem to refer to a process, then DO NOT INFERE or CREATE FACTS. Just return all field blanks

TEXTS:
{text}

OUTPUT LANGUAGE: German

FORMAT RULES:
- Respond ONLY with valid JSON.
- Do NOT output multiple JSON objects.
- Do NOT output text before or after the JSON.
- The FIRST character must be '{'. The LAST must be '}'.
- Return a single JSON for the process.
- The JSON must exactly follow this structure:
{
  "process_name": "...",
  "process_description": "...",
  "process_steps": [{id, name, description}, ... ],
}
"""

        self.INTERNAL_PROMPT = """
You are analyzing a list of json items, each one of them describing a process.

Each input process item in the list has this information:
- Process name
- Process description
- Process steps (an ordered list of the key steps composing the process)
- JSON item structure:
{
  "process_name": "...",
  "process_description": "...",
  "process_steps": [{id, name, description}, ... ],
}

AGREGATION TASK:
- For each child process item, pick the steps and agregate them into one single step description
- The collection of the resulting aggregated steps from each child process will become the list of steps for a new aggregated process (the parent process) encompasing them.

OUTPUT TASK: Produce a json for the newly created parent process with the following information:
- Process description: A 4-8 sentences explaining what this parent process is about, from the combination of the child process descriptions
- Proces name: A short label (max 6 words). YOU MUST ALWAYS PROVIDE A LABEL based on the generated description.
- Process steps: An ordered list of the key steps composing the process (the aggregated steps of each input child process). Give them a sequential numeric id (the list index), a short label and a short one-line description.

OUTPUT LANGUAGE: German

COVERAGE RULE:
If the process is unclear, too small, or ambiguous, you MUST STILL provide the best possible descriptive label, warning about the quality of the given details. Never return an empty process. All input child processes should map to at least one parent process

INPUT:
{json_list}

FORMAT RULES:
- Respond ONLY with valid JSON.
- Do NOT output multiple JSON objects.
- Do NOT output text before or after the JSON.
- The FIRST character must be '{'. The LAST must be '}'.
- Return a single JSON for the process.
- The JSON must exactly follow this structure:
{
  "process_name": "...",
  "process_description": "...",
  "process_steps": [{id, name, description}, ... ],
}
"""

        # Initialize the underlying extractor
        self.extractor = Cluster_Info_Extractor(
            self.llm,
            self.vectordb,
            self.LEAF_PROMPT,
            self.INTERNAL_PROMPT,
            info_type=self.INFO_TYPE,
            output_folder=self.output_folder,
            verbose=self.verbose,
            show_progress_bar=self.show_progress_bar,
            max_concurrent_llm_calls=self.max_concurrent_llm_calls,
            log_prompts=self.log_prompts,
            branch_id=self.branch_id
        )

    def process_hierarchy(self, hierarchy_file_path):
        return self.extractor.process_hierarchy_file(hierarchy_file_path)
