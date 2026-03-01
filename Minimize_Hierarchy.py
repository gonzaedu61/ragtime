import json

def transform_node(node):
    # Remove cutoff and depth if present
    del node["cutoff"]
    del node["depth"]

    # Process clusters list
    if "clusters" in node:
        for cluster in node["clusters"]:
            # Remove metadatas if present
            del cluster["metadatas"]

            children = cluster["children"]

            if children is not None:
                # children is not null → remove ids
                if "ids" in cluster:
                    del cluster["ids"]
               
                # Recurse into children 
                transform_node(children)
            else:
                # children is null → rename ids → chunk_ids
                if "ids" in cluster:
                  cluster["chunk_ids"] = cluster["ids"]
                  del cluster["ids"]

    return node


# ---- Example usage ----

with open("Labeled_Topics_Hierarchy (full).json", "r") as f:
    data = json.load(f)

cleaned = transform_node(data)

with open("Labeled_Topics_Hierarchy.cleaned.json", "w") as f:
    json.dump(cleaned, f, indent=2)
