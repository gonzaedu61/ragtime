import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import json


class Semantic_Ontology_Builder:

    # ---------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------
    def __init__(
        self,
        vector_db,
        initial_cutoff=0.45,
        min_cluster_size=5,
        cutoff_decay=0.75,
        min_cutoff=0.10,
        max_depth=6,
        metadata_keys=None,
        metadata_weight=0.25,
        postprocess_rules=None
    ):
        """
        vector_db must implement:
            get_all() -> embeddings, ids, documents, metadatas

        metadata_keys: list of metadata fields to encode into vectors
        metadata_weight: scaling factor for metadata vector
        postprocess_rules: list of functions(cluster) -> modified cluster
        """
        self.db = vector_db

        self.initial_cutoff = initial_cutoff
        self.min_cluster_size = min_cluster_size
        self.cutoff_decay = cutoff_decay
        self.min_cutoff = min_cutoff
        self.max_depth = max_depth

        self.metadata_keys = metadata_keys or []
        self.metadata_weight = metadata_weight

        # Post-processing rules (splitting, merging, constraints)
        self.postprocess_rules = postprocess_rules or []

    # ---------------------------------------------------------
    # Metadata → vector encoding
    # ---------------------------------------------------------
    def encode_metadata(self, metadatas):
        """
        Convert metadata dicts into numeric vectors.
        Simple one-hot encoding for categorical fields.
        """
        if not self.metadata_keys:
            return np.zeros((len(metadatas), 1))

        # Build vocabulary for each metadata key
        vocab = {key: {} for key in self.metadata_keys}

        # Assign integer IDs for each categorical value
        for meta in metadatas:
            for key in self.metadata_keys:
                val = meta.get(key, None)
                if val not in vocab[key]:
                    vocab[key][val] = len(vocab[key])

        # Encode metadata
        encoded = []
        for meta in metadatas:
            vec = []
            for key in self.metadata_keys:
                size = len(vocab[key])
                one_hot = np.zeros(size)
                val = meta.get(key, None)
                idx = vocab[key].get(val, None)
                if idx is not None:
                    one_hot[idx] = 1
                vec.extend(one_hot)
            encoded.append(vec)

        return np.array(encoded)

    # ---------------------------------------------------------
    # Combine embeddings + metadata vectors
    # ---------------------------------------------------------
    def combine_vectors(self, embeddings, metadata_vectors):
        """
        Concatenate embedding + scaled metadata vector.
        """
        metadata_vectors = metadata_vectors * self.metadata_weight
        return np.concatenate([embeddings, metadata_vectors], axis=1)

    # ---------------------------------------------------------
    # Utility: compute centroid
    # ---------------------------------------------------------
    def compute_centroid(self, embeddings):
        return np.mean(embeddings, axis=0)

    # ---------------------------------------------------------
    # Utility: label cluster
    # ---------------------------------------------------------
    def label_cluster(self, centroid, documents, top_k=3):
        sims = cosine_similarity([centroid], documents)[0]
        top_indices = sims.argsort()[-top_k:][::-1]
        sample_texts = [documents[i][:120] for i in top_indices]
        return " | ".join(sample_texts)

    # ---------------------------------------------------------
    # Apply metadata-based post-processing rules
    # ---------------------------------------------------------
    def apply_postprocess_rules(self, cluster):
        """
        Each rule receives the cluster dict and returns a modified cluster.
        """
        for rule in self.postprocess_rules:
            cluster = rule(cluster)
        return cluster

    # ---------------------------------------------------------
    # Recursive clustering
    # ---------------------------------------------------------
    def recursive_cluster(
        self,
        embeddings,
        ids,
        documents,
        metadatas,
        cutoff,
        depth=0
    ):
        if len(embeddings) < self.min_cluster_size:
            return None

        if cutoff < self.min_cutoff:
            return None

        if depth > self.max_depth:
            return None

        # Metadata-aware vector augmentation
        metadata_vectors = self.encode_metadata(metadatas)
        combined_vectors = self.combine_vectors(embeddings, metadata_vectors)

        # Run clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=cutoff,
            affinity="cosine",
            linkage="average"
        )

        labels = clustering.fit_predict(combined_vectors)

        # Group items by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, {
                "embeddings": [],
                "ids": [],
                "documents": [],
                "metadatas": []
            })
            clusters[label]["embeddings"].append(embeddings[idx])
            clusters[label]["ids"].append(ids[idx])
            clusters[label]["documents"].append(documents[idx])
            clusters[label]["metadatas"].append(metadatas[idx])

        # Build hierarchy node
        node = {
            "cutoff": cutoff,
            "depth": depth,
            "clusters": []
        }

        # Process each cluster
        for label, group in clusters.items():

            # Apply metadata-based post-processing rules
            group = self.apply_postprocess_rules(group)

            group_embeddings = np.array(group["embeddings"])
            centroid = self.compute_centroid(group_embeddings)

            cluster_label = self.label_cluster(
                centroid,
                group["documents"]
            )

            child_node = {
                "cluster_id": label,
                "label": cluster_label,
                "size": len(group["ids"]),
                "ids": group["ids"],
                "metadatas": group["metadatas"],
                "children": None
            }

            # Recurse if cluster is large enough
            if len(group["ids"]) >= self.min_cluster_size:
                child_node["children"] = self.recursive_cluster(
                    embeddings=group_embeddings,
                    ids=group["ids"],
                    documents=group["documents"],
                    metadatas=group["metadatas"],
                    cutoff=cutoff * self.cutoff_decay,
                    depth=depth + 1
                )

            node["clusters"].append(child_node)

        return node

    # ---------------------------------------------------------
    # Public API: Build ontology
    # ---------------------------------------------------------
    def build(self):
        embeddings, ids, documents, metadatas = self.db.get_all()
        embeddings = np.array(embeddings)

        return self.recursive_cluster(
            embeddings=embeddings,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            cutoff=self.initial_cutoff
        )

    # ---------------------------------------------------------
    # Public API: Save to JSON
    # ---------------------------------------------------------
    def save(self, hierarchy, filename="topic_hierarchy.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(hierarchy, f, indent=2, ensure_ascii=False)
