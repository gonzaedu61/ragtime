import json
import numpy as np


class Hierarchy_Enricher:
    def __init__(self, vdb_client=None):
        """
        vdb_client: object with method get_embedding(chunk_id) -> np.array-like
        """
        self.vdb = vdb_client

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def enrich(self, hierarchy_path, retrieved_path, output_path):
        """
        Load hierarchy and retrieved chunks from files,
        enrich the hierarchy with drift metrics, compute global stats,
        and write the result to output_path.
        """
        with open(hierarchy_path, "r", encoding="utf-8") as f:
            hierarchy = json.load(f)

        # Unwrap root if file has a top-level "clusters" list
        if isinstance(hierarchy, dict) and "clusters" in hierarchy and isinstance(hierarchy["clusters"], list):
            if len(hierarchy["clusters"]) == 0:
                raise ValueError("Hierarchy has an empty 'clusters' list.")
            hierarchy = hierarchy["clusters"][0]

        with open(retrieved_path, "r", encoding="utf-8") as f:
            retrieved = json.load(f)

        retrieved_index = self._index_retrieved(retrieved)

        # Enrich all clusters with per-node metrics
        self._enrich_node(hierarchy, retrieved_index)

        # Compute global drift metrics
        global_stats = self.compute_aggregate_drift(hierarchy)
        drift_by_depth = self.compute_drift_by_depth(hierarchy)

        # Attach global metrics at the root
        hierarchy["_global_drift_statistics"] = global_stats
        hierarchy["_drift_by_depth"] = drift_by_depth

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(hierarchy, f, indent=2)

        return hierarchy

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------
    def _index_retrieved(self, retrieved_json):
        """
        Convert retrieved clusters into:
            { cluster_id: [chunk_id1, chunk_id2, ...] }
        """
        indexed = {}
        for entry in retrieved_json:
            cid = entry.get("cluster_id")
            chunks = entry.get("retrieved_chunks", [])

            # Extract chunk IDs correctly
            chunk_ids = [
                c["chunk_id"] for c in chunks
                if isinstance(c, dict) and "chunk_id" in c
            ]

            if cid is not None:
                indexed[cid] = chunk_ids

        return indexed

    def _collect_leaf_chunks(self, node):
        """
        Recursively collect all chunk_ids from descendant leaves.
        A leaf is defined as node with children == None and having "chunk_ids".
        """
        if node.get("children") is None:
            return node.get("chunk_ids", []) or []

        collected = []
        for child in node["children"].get("clusters", []):
            collected.extend(self._collect_leaf_chunks(child))
        return collected

    def _compute_overlap_distance(self, original, matching):
        """
        Jaccard distance between original and matching chunk ID sets.
        """
        if not original and not matching:
            return None
        A, B = set(original), set(matching)
        if not A and not B:
            return None
        return 1 - len(A & B) / len(A | B)

    def _compute_semantic_shift(self, original, matching):
        """
        Cosine distance between centroids of original and matching embeddings.
        Returns None if embeddings cannot be computed.
        """
        if not self.vdb or not original or not matching:
            return None

        orig_embs = [self.vdb.get_embedding(cid) for cid in original]
        match_embs = [self.vdb.get_embedding(cid) for cid in matching]

        # Filter out missing embeddings
        orig_embs = [e for e in orig_embs if e is not None]
        match_embs = [e for e in match_embs if e is not None]

        if not orig_embs or not match_embs:
            return None

        c_orig = np.mean(np.array(orig_embs), axis=0)
        c_match = np.mean(np.array(match_embs), axis=0)

        denom = (np.linalg.norm(c_orig) * np.linalg.norm(c_match))
        if denom == 0:
            return None

        cos_sim = float(np.dot(c_orig, c_match) / denom)
        return 1 - cos_sim

    def _enrich_node(self, node, retrieved_index):
        """
        Recursively enrich a cluster node with:
        - matching_chunks
        - overlap_distance
        - semantic_shift
        """
        cid = node.get("cluster_id")

        # Add matching chunks
        matching = retrieved_index.get(cid, []) or []
        node["matching_chunks"] = matching

        # Collect original chunks (leaf or aggregated)
        original = self._collect_leaf_chunks(node)

        # Compute deviation metrics
        node["overlap_distance"] = self._compute_overlap_distance(original, matching)
        node["semantic_shift"] = self._compute_semantic_shift(original, matching)

        # Recurse into children
        children = node.get("children")
        if children and "clusters" in children:
            for child in children["clusters"]:
                self._enrich_node(child, retrieved_index)

    # ---------------------------------------------------------
    # Global drift metrics
    # ---------------------------------------------------------
    def compute_aggregate_drift(self, hierarchy_root):
        """
        Compute aggregate drift statistics across the entire hierarchy.
        Returns a dict with summary metrics.
        """
        overlap_values = []
        semantic_values = []
        weighted_overlap = []
        weighted_semantic = []
        total_size = 0

        def traverse(node):
            nonlocal total_size

            od = node.get("overlap_distance")
            ss = node.get("semantic_shift")
            size = len(self._collect_leaf_chunks(node))

            if size > 0:
                total_size += size

                if od is not None:
                    overlap_values.append(od)
                    weighted_overlap.append(od * size)

                if ss is not None:
                    semantic_values.append(ss)
                    weighted_semantic.append(ss * size)

            children = node.get("children")
            if children and "clusters" in children:
                for child in children["clusters"]:
                    traverse(child)

        traverse(hierarchy_root)

        # If no valid data, return empty stats
        if total_size == 0 and not overlap_values and not semantic_values:
            return {
                "mean_overlap": None,
                "mean_semantic_shift": None,
                "weighted_mean_overlap": None,
                "weighted_mean_semantic_shift": None,
                "max_overlap": None,
                "max_semantic_shift": None,
                "percentiles_overlap": {},
                "percentiles_semantic_shift": {},
            }

        def safe_mean(values):
            return float(np.mean(values)) if values else None

        def safe_max(values):
            return float(np.max(values)) if values else None

        def safe_percentiles(values):
            if not values:
                return {}
            return {
                "25": float(np.percentile(values, 25)),
                "50": float(np.percentile(values, 50)),
                "75": float(np.percentile(values, 75)),
                "95": float(np.percentile(values, 95)),
            }

        weighted_mean_overlap = (
            float(sum(weighted_overlap) / total_size)
            if weighted_overlap and total_size > 0
            else None
        )
        weighted_mean_semantic = (
            float(sum(weighted_semantic) / total_size)
            if weighted_semantic and total_size > 0
            else None
        )

        return {
            "mean_overlap": safe_mean(overlap_values),
            "mean_semantic_shift": safe_mean(semantic_values),
            "weighted_mean_overlap": weighted_mean_overlap,
            "weighted_mean_semantic_shift": weighted_mean_semantic,
            "max_overlap": safe_max(overlap_values),
            "max_semantic_shift": safe_max(semantic_values),
            "percentiles_overlap": safe_percentiles(overlap_values),
            "percentiles_semantic_shift": safe_percentiles(semantic_values),
        }

    # ---------------------------------------------------------
    # Drift by depth
    # ---------------------------------------------------------
    def compute_drift_by_depth(self, hierarchy_root):
        """
        Compute mean drift metrics grouped by depth in the hierarchy.
        Returns: { depth: { mean_overlap, mean_semantic_shift, cluster_count } }
        """
        drift_by_depth = {}

        def traverse(node, depth):
            od = node.get("overlap_distance")
            ss = node.get("semantic_shift")

            if depth not in drift_by_depth:
                drift_by_depth[depth] = {"overlap": [], "semantic": []}

            if od is not None:
                drift_by_depth[depth]["overlap"].append(od)
            if ss is not None:
                drift_by_depth[depth]["semantic"].append(ss)

            children = node.get("children")
            if children and "clusters" in children:
                for child in children["clusters"]:
                    traverse(child, depth + 1)

        traverse(hierarchy_root, 0)

        summary = {}
        for depth, values in drift_by_depth.items():
            overlaps = values["overlap"]
            semantics = values["semantic"]

            mean_overlap = float(np.mean(overlaps)) if overlaps else None
            mean_semantic = float(np.mean(semantics)) if semantics else None

            summary[depth] = {
                "mean_overlap": mean_overlap,
                "mean_semantic_shift": mean_semantic,
                "cluster_count": len(overlaps) + len(semantics) - min(len(overlaps), len(semantics)),
            }

        return summary
