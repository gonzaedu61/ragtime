import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import json
import sys


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
        self.verbose = verbose

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

                try:
                    # Multi-label support: expand lists
                    if isinstance(val, list):
                        for item in val:
                            if item not in vocab[key]:
                                vocab[key][item] = len(vocab[key])
                    else:
                        if val not in vocab[key]:
                            vocab[key][val] = len(vocab[key])

                except TypeError:
                    print("\n🔥 ERROR: Unhashable metadata value encountered during vocab build")
                    print(f"   metadata key: {key}")
                    print(f"   raw value: {val}")
                    print(f"   type: {type(val)}")
                    print("   full metadata object:", meta)
                    raise

        # Encode metadata
        encoded = []
        for meta in metadatas:
            vec = []
            for key in self.metadata_keys:
                size = len(vocab[key])
                one_hot = np.zeros(size)
                val = meta.get(key, None)

                try:
                    if isinstance(val, list):
                        for item in val:
                            idx = vocab[key].get(item, None)
                            if idx is not None:
                                one_hot[idx] = 1
                    else:
                        idx = vocab[key].get(val, None)
                        if idx is not None:
                            one_hot[idx] = 1

                except TypeError:
                    print("\n🔥 ERROR: Unhashable metadata value encountered during encoding")
                    print(f"   metadata key: {key}")
                    print(f"   raw value: {val}")
                    print(f"   type: {type(val)}")
                    print("   full metadata object:", meta)
                    raise

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
    def label_cluster(self, centroid, embeddings, documents, top_k=3):
        sims = cosine_similarity([centroid], embeddings)[0]
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
        depth=0,
        path=""
    ):
        if self.verbose:
            print(f"[Depth {depth}] Starting clustering with {len(embeddings)} items, cutoff={cutoff:.4f}")

        # Stop conditions
        if len(embeddings) < self.min_cluster_size:
            if self.verbose:
                print(f"[Depth {depth}] Too few items ({len(embeddings)}). Stopping.")
            return None

        if cutoff < self.min_cutoff:
            if self.verbose:
                print(f"[Depth {depth}] Cutoff {cutoff:.4f} below min_cutoff. Stopping.")
            return None

        if depth > self.max_depth:
            if self.verbose:
                print(f"[Depth {depth}] Max depth reached. Stopping.")
            return None

        # Metadata-aware vector augmentation
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

        if self.verbose:
            n_clusters = len(set(labels))
            print(f"[Depth {depth}] Formed {n_clusters} clusters")

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

        # Build node
        node = {
            "cutoff": cutoff,
            "depth": depth,
            "clusters": []
        }

        # Process each cluster
        for label, group in clusters.items():

            if self.verbose:
                print(f"[Depth {depth}] Processing cluster {label} (size={len(group['ids'])})")

            # Apply post-processing rules
            if self.postprocess_rules:
                if self.verbose:
                    print(f"[Depth {depth}] Applying {len(self.postprocess_rules)} postprocess rules")
                group = self.apply_postprocess_rules(group)

            group_embeddings = np.array(group["embeddings"])
            centroid = self.compute_centroid(group_embeddings)

            # Label cluster
            cluster_label = self.label_cluster(
                centroid,
                group_embeddings,
                group["documents"]
            )

            # Build hierarchical path ID
            child_path = f"{path}.{label}" if path != "" else str(label)

            child_node = {
                "cluster_id": child_path,
                "label": cluster_label,
                "size": int(len(group["ids"])),
                "ids": list(group["ids"]),
                "metadatas": group["metadatas"],
                "children": None
            }

            # Recurse if cluster is large enough
            if len(group["ids"]) >= self.min_cluster_size:
                if self.verbose:
                    print(f"[Depth {depth}] Recursing into cluster {label}")
                child_node["children"] = self.recursive_cluster(
                    embeddings=group_embeddings,
                    ids=group["ids"],
                    documents=group["documents"],
                    metadatas=group["metadatas"],
                    cutoff=cutoff * self.cutoff_decay,
                    depth=depth + 1,
                    path=child_path
                )

            node["clusters"].append(child_node)

        # Add clusters_count field
        node["clusters_count"] = len(node["clusters"])

        if self.verbose:
            print(f"[Depth {depth}] Completed depth {depth}")


        # --- MERGE REDUNDANT LEVELS ---
        merged_clusters = []
        for cluster in node["clusters"]:
            child = cluster["children"]

            # Check if child exists and is redundant
            if (
                child is not None
                and child["clusters_count"] == 1
                and set(cluster["ids"]) == set(child["clusters"][0]["ids"])
            ):
                # Replace this cluster's children with the grandchild
                cluster["children"] = child["clusters"][0]["children"]
                # Optionally: update label to something more stable
                cluster["label"] = child["clusters"][0]["label"]

            merged_clusters.append(cluster)

        node["clusters"] = merged_clusters
        node["clusters_count"] = len(merged_clusters)

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


    # ---------------------------------------------------------
    # Public API: Pretty Print Tree
    # ---------------------------------------------------------
    def print_tree(self, node, indent=0):
        """
        Pretty-print the hierarchy tree using metadata fields instead of text snippets.
        Shows cluster_id, size, and for each item: document_name, pages, chunk_id.
        """

        if node is None:
            print(" " * indent + "(empty)")
            return

        # Print each cluster
        for cluster in node["clusters"]:
            cid = cluster["cluster_id"]
            size = cluster["size"]

            print(" " * indent + f"- [{cid}] size={size}")

            # Print metadata summary for each item in the cluster
            for meta in cluster["metadatas"]:
                doc = meta.get("document_name", "unknown")
                pages = meta.get("pages", [])
                chunk = meta.get("chunk_id", "n/a")

                print(" " * (indent + 2) + f"- {doc} | pages={pages} | chunk={chunk}")

            # Recurse into children
            if cluster["children"] is not None:
                self.print_tree(cluster["children"], indent + 4)

