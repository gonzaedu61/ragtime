import os
import json

class Prompt_Loader:
    def __init__(self, project_root: str):
        """
        project_root is the folder where:
            - prompts/
            - steps/
            - utils/
            - ontology_foundation_builder.py
        live together.
        """
        self.prompts_dir = os.path.join(project_root, "prompts")

    def load(self, relative_path: str) -> str:
        """
        Load a prompt template from prompts/<relative_path>.
        """
        path = os.path.join(self.prompts_dir, relative_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def fill(self, template: str, **kwargs) -> str:
        """
        Safely fill a prompt template with JSON-serializable values.
        """
        safe = {}
        for k, v in kwargs.items():
            if isinstance(v, (dict, list)):
                safe[k] = json.dumps(v, ensure_ascii=False, indent=2)
            else:
                safe[k] = str(v)
        return template.format(**safe)
