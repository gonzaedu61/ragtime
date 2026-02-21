import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import json


class Topic_Hierarchy_Builder:

    # ---------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------
    def __init__(
        self,
        vector_db,
        initial_cutoff=0.5,
        min_cluster_size=5,
        cutoff_decay=0.85,
        min_cutoff=0.15,
        max_depth=6,
        metadata_keys=None,
        metadata_weight=0.1,
        postprocess_rules=None,
        verbose=False
    ):
        self.db = vector_db

        self.initial_cutoff = initial_cutoff
        self.min_cluster_size = min_cluster_size
        self.cutoff_decay = cutoff_decay
        self.min_cutoff = min_cutoff
        self.max_depth = max_depth

        self.metadata_keys = metadata_keys or []
        self.metadata_weight = metadata_weight

        self.postprocess_rules = postprocess_rules or []
        self.verbose = verbose

    # ---------------------------------------------------------
    # Metadata → vector encoding
    # ---------------------------------------------------------
    def encode_metadata(self, metadatas):
        if not self.metadata_keys:
            return np.zeros((len(metadatas), 1))

        vocab = {key: {} for key in self.metadata_keys}

        # Build vocab
        for meta in metadatas:
            for key in self.metadata_keys:
                val = meta.get(key, None)
                if isinstance(val, list):
                    for item in val:
                        if item not in vocab[key]:
                            vocab[key][item] = len(vocab[key])
                else:
                    if val not in vocab[key]:
                        vocab[key][val] = len(vocab[key])

        # Encode
        encoded = []
        for meta in metadatas:
            vec = []
            for key in self.metadata_keys:
                size = len(vocab[key])
                one_hot = np.zeros(size)
                val = meta.get(key, None)

                if isinstance(val, list):
                    for item in val:
                        idx = vocab[key].get(item)
                        if idx is not None:
                            one_hot[idx] = 1
                else:
                    idx = vocab[key].get(val)
                    if idx is not None:
                        one_hot[idx] = 1

                vec.extend(one_hot)
            encoded.append(vec)

        return np.array(encoded)

    # ---------------------------------------------------------
    # Combine embeddings + metadata vectors
    # ---------------------------------------------------------
    def combine_vectors(self, embeddings, metadata_vectors):
        metadata_vectors = metadata_vectors * self.metadata_weight
        return np.concatenate([embeddings, metadata_vectors], axis=1)

    # ---------------------------------------------------------
    # Compute centroid
    # ---------------------------------------------------------
    def compute_centroid(self, embeddings):
        return np.mean(embeddings, axis=0)

    # ---------------------------------------------------------
    # Recursive clustering (single unified path)
    # ---------------------------------------------------------
    def recursive_cluster(
        self,
        embeddings,
        ids,
        metadatas,
        cutoff,
        depth=0,
        path=""
    ):
        if len(embeddings) < self.min_cluster_size:
            return None
        if cutoff < self.min_cutoff:
            return None
        if depth > self.max_depth:
            return None

        # Build augmented vectors
        metadata_vectors = self.encode_metadata(metadatas)
        combined_vectors = self.combine_vectors(embeddings, metadata_vectors)

        # Run clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=cutoff,
            metric="cosine",
            linkage="average"
        )
        labels = clustering.fit_predict(combined_vectors)

        # Group items
        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, {
                "embeddings": [],
                "ids": [],
                "metadatas": []
            })
            clusters[label]["embeddings"].append(embeddings[idx])
            clusters[label]["ids"].append(ids[idx])
            clusters[label]["metadatas"].append(metadatas[idx])

        # Build node
        node = {
            "cutoff": cutoff,
            "depth": depth,
            "clusters": []
        }

        # Process clusters
        for label, group in clusters.items():
            group_embeddings = np.array(group["embeddings"])
            child_path = f"{path}.{label}" if path else str(label)

            child_node = {
                "cluster_id": child_path,
                "size": len(group["ids"]),
                "ids": list(group["ids"]),
                "metadatas": group["metadatas"],
                "children": None
            }

            # Recurse
            if len(group["ids"]) >= self.min_cluster_size:
                child_node["children"] = self.recursive_cluster(
                    embeddings=group_embeddings,
                    ids=child_node["ids"],
                    metadatas=child_node["metadatas"],
                    cutoff=cutoff * self.cutoff_decay,
                    depth=depth + 1,
                    path=child_path
                )

            node["clusters"].append(child_node)

        node["clusters_count"] = len(node["clusters"])
        return node

    # ---------------------------------------------------------
    # Redundant-level merging (full mode)
    # ---------------------------------------------------------
    def merge_redundant_levels(self, node):
        merged = []
        for cluster in node["clusters"]:
            child = cluster["children"]

            if (
                child is not None
                and child["clusters_count"] == 1
                and set(cluster["ids"]) == set(child["clusters"][0]["ids"])
            ):
                cluster["children"] = child["clusters"][0]["children"]

            if cluster["children"] is not None:
                self.merge_redundant_levels(cluster["children"])

            merged.append(cluster)

        node["clusters"] = merged
        node["clusters_count"] = len(merged)

    # ---------------------------------------------------------
    # Minimal tree transformation
    # ---------------------------------------------------------
    def to_minimal_tree(self, node):
        """
        Convert full tree → minimal tree:
            - remove ids, metadatas
            - keep cluster_id, size, children
            - leaf nodes get chunk_ids
        """
        minimal = {"clusters": []}

        for cluster in node["clusters"]:
            new_c = {
                "cluster_id": cluster["cluster_id"],
                "size": cluster["size"],
                "children": None
            }

            if cluster["children"] is None:
                new_c["chunk_ids"] = cluster["ids"]
            else:
                new_c["children"] = self.to_minimal_tree(cluster["children"])

            minimal["clusters"].append(new_c)

        minimal["clusters_count"] = len(minimal["clusters"])
        return minimal

    # ---------------------------------------------------------
    # Public API: Build ontology
    # ---------------------------------------------------------
    def build(self, minimal=False):
        embeddings, ids, metadatas = self.db.get_for_clustering(self.metadata_keys)
        embeddings = np.array(embeddings)

        # Build full tree
        tree = self.recursive_cluster(
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
            cutoff=self.initial_cutoff
        )

        # Merge redundant levels
        self.merge_redundant_levels(tree)

        # Minimal mode: strip tree
        if minimal:
            return self.to_minimal_tree(tree)

        return tree

    # ---------------------------------------------------------
    # Save to JSON
    # ---------------------------------------------------------
    def save(self, hierarchy, filename="topic_hierarchy.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(hierarchy, f, indent=2, ensure_ascii=False)
