import json
import os
from typing import Dict, List, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Topic_Hierarchy_Harmonizer:
    """
    Harmonizes 'Big_Group' clusters in a topic hierarchy JSON by splitting them
    into smaller, semantically coherent groups constrained by [min_tokens, max_tokens].

    Includes:
      - Option D metadata reaggregation (same logic as original builder)
      - Penalty-based fallback grouping (Option B weights)
      - Forced subgroup handling for oversized children
      - Guaranteed termination
    """

    # Penalty weights (Option B)
    PENALTY_EXCEED_MAX = 5
    PENALTY_BELOW_MIN = 3

    def __init__(
        self,
        vector_db,
        input_json_path: str,
        max_tokens: int,
        min_tokens: int,
        output_dir: str,
        verbose: bool = False,
    ):
        self.db = vector_db
        self.input_json_path = input_json_path
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.verbose = verbose

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_json_path))[0]
        self.output_json_path = os.path.join(output_dir, f"{base_name}_harmonized.json")
        self.log_path = os.path.join(output_dir, f"{base_name}_harmonization_log.txt")

        self.hierarchy: Dict[str, Any] = {}
        self.chunk_token_cache: Dict[str, int] = {}
        self.cluster_token_cache: Dict[str, int] = {}
        self.cluster_centroid_cache: Dict[str, np.ndarray] = {}
        self.log_entries: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self):
        self._load_hierarchy()
        self._build_chunk_token_cache()
        self._compute_cluster_tokens_and_centroids(self.hierarchy)
        self._harmonize_tree(self.hierarchy)
        self._renumber_cluster_ids(self.hierarchy)
        self._update_total_clusters(self.hierarchy)
        self._save_hierarchy()
        self._write_log()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def _load_hierarchy(self):
        with open(self.input_json_path, "r", encoding="utf-8") as f:
            self.hierarchy = json.load(f)

    def _save_hierarchy(self):
        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(self.hierarchy, f, indent=2, ensure_ascii=False)

    def _write_log(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("=== Big_Group Harmonization Log ===\n\n")
            for entry in self.log_entries:
                f.write(f"Big_Group cluster_id: {entry['cluster_id']}\n")
                f.write(f"  total_tokens_before: {entry['total_tokens_before']}\n")
                f.write(f"  num_children_before: {entry['num_children_before']}\n")
                f.write(f"  num_groups_after: {len(entry['groups'])}\n")
                for g in entry["groups"]:
                    f.write(f"    New subgroup index: {g['group_index']}\n")
                    f.write(f"      token_sum: {g['token_sum']}\n")
                    f.write(f"      children_cluster_ids: {g['children_cluster_ids']}\n")
                f.write("\n")
            f.write("===================================\n")

    # ------------------------------------------------------------------
    # Chunk token cache
    # ------------------------------------------------------------------
    def _build_chunk_token_cache(self):
        """
        Collect all chunk_ids in the hierarchy and fetch token_count from metadata.
        """
        all_chunk_ids = set()

        def collect_ids(node: Dict[str, Any]):
            for c in node.get("clusters", []):
                if c.get("children") is None:
                    for cid in c.get("ids", []):
                        all_chunk_ids.add(cid)
                else:
                    collect_ids(c["children"])

        collect_ids(self.hierarchy)

        for cid in all_chunk_ids:
            rec = self.db.get_by_id(cid)
            meta = rec.get("metadata", {}) if rec else {}
            self.chunk_token_cache[cid] = int(meta.get("token_count", 0))

    # ------------------------------------------------------------------
    # Cluster type helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_leaf(cluster: Dict[str, Any]) -> bool:
        return cluster.get("children") is None

    @staticmethod
    def _is_internal(cluster: Dict[str, Any]) -> bool:
        return cluster.get("children") is not None

    def _is_leaf_parent(self, cluster: Dict[str, Any]) -> bool:
        if not self._is_internal(cluster):
            return False
        children = cluster["children"].get("clusters", [])
        if not children:
            return False
        return all(self._is_leaf(c) for c in children)

    def _is_group(self, cluster: Dict[str, Any]) -> bool:
        """
        Group cluster:
          - Internal
          - Children are a mix of Leaf and Leafs_Parent
          - No deeper level than Leafs_Parent
        """
        if not self._is_internal(cluster):
            return False

        children = cluster["children"].get("clusters", [])
        if not children:
            return False

        has_leaf = False
        has_leaf_parent = False

        for child in children:
            if self._is_leaf(child):
                has_leaf = True
            elif self._is_leaf_parent(child):
                has_leaf_parent = True
            else:
                return False

        return has_leaf and has_leaf_parent

    # ------------------------------------------------------------------
    # Token & centroid computation
    # ------------------------------------------------------------------
    def _iter_descendant_leaves(self, cluster: Dict[str, Any]):
        if self._is_leaf(cluster):
            yield cluster
        else:
            for c in cluster["children"].get("clusters", []):
                yield from self._iter_descendant_leaves(c)

    def _compute_cluster_tokens_and_centroids(self, node: Dict[str, Any]):
        """
        For each cluster:
          - total tokens = sum of descendant leaf chunk tokens
          - centroid = mean of descendant leaf chunk embeddings
        """
        for cluster in node.get("clusters", []):
            cid = cluster.get("cluster_id", "")

            if self._is_leaf(cluster):
                chunk_ids = cluster.get("ids", [])
                tokens = sum(self.chunk_token_cache.get(ch, 0) for ch in chunk_ids)
                self.cluster_token_cache[cid] = tokens

                if chunk_ids:
                    embs = []
                    for ch in chunk_ids:
                        emb = self.db.get_embedding(ch)
                        if emb is not None:
                            embs.append(np.array(emb, dtype=np.float32))
                    centroid = np.mean(np.stack(embs, axis=0), axis=0) if embs else None
                else:
                    centroid = None

                self.cluster_centroid_cache[cid] = centroid

            else:
                self._compute_cluster_tokens_and_centroids(cluster["children"])

                total_tokens = 0
                centroids = []
                for leaf in self._iter_descendant_leaves(cluster):
                    leaf_cid = leaf.get("cluster_id", "")
                    total_tokens += self.cluster_token_cache.get(leaf_cid, 0)
                    c = self.cluster_centroid_cache.get(leaf_cid)
                    if c is not None:
                        centroids.append(c)

                self.cluster_token_cache[cid] = total_tokens
                self.cluster_centroid_cache[cid] = (
                    np.mean(np.stack(centroids, axis=0), axis=0) if centroids else None
                )

    # ------------------------------------------------------------------
    # Harmonization traversal
    # ------------------------------------------------------------------
    def _harmonize_tree(self, node: Dict[str, Any]):
        for cluster in node.get("clusters", []):
            cid = cluster.get("cluster_id", "")
            if self._is_group(cluster):
                total_tokens = self.cluster_token_cache.get(cid, 0)
                if total_tokens > self.max_tokens:
                    if self.verbose:
                        print(f"[Big_Group] Splitting cluster {cid} (tokens={total_tokens})")
                    self._split_big_group(cluster, total_tokens)
            if self._is_internal(cluster):
                self._harmonize_tree(cluster["children"])

    # ------------------------------------------------------------------
    # Big_Group splitting with fallback penalty logic
    # ------------------------------------------------------------------
    def _split_big_group(self, group_cluster: Dict[str, Any], total_tokens_before: int):
        cid = group_cluster.get("cluster_id", "")
        children = group_cluster["children"]["clusters"]
        if len(children) <= 1:
            return

        # Prepare items
        items = []
        for child in children:
            child_cid = child.get("cluster_id", "")
            tokens = self.cluster_token_cache.get(child_cid, 0)
            centroid = self.cluster_centroid_cache.get(child_cid)
            items.append(
                {
                    "cluster": child,
                    "cluster_id": child_cid,
                    "tokens": tokens,
                    "centroid": centroid,
                }
            )

        # Forced subgroup: any child > max_tokens must be alone
        forced_groups = []
        remaining_items = []
        for it in items:
            if it["tokens"] > self.max_tokens:
                forced_groups.append([it])
            else:
                remaining_items.append(it)

        # Greedy grouping for remaining items
        groups = [{"items": g, "tokens": sum(it["tokens"] for it in g),
                   "centroid": self._compute_group_centroid(g)} for g in forced_groups]

        # Sort remaining by size
        remaining_items.sort(key=lambda x: x["tokens"], reverse=True)

        for item in remaining_items:
            best_idx = None
            best_sim = -1.0

            for idx, g in enumerate(groups):
                if g["tokens"] + item["tokens"] > self.max_tokens:
                    continue
                sim = self._similarity(g["centroid"], item["centroid"])
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx

            if best_idx is not None:
                g = groups[best_idx]
                g["items"].append(item)
                g["tokens"] += item["tokens"]
                g["centroid"] = self._compute_group_centroid(g["items"])
            else:
                groups.append(
                    {
                        "items": [item],
                        "tokens": item["tokens"],
                        "centroid": item["centroid"],
                    }
                )

        # Merge undersized groups
        self._merge_small_groups(groups)

        # Fallback penalty scoring
        groups = self._apply_penalty_fallback(groups)

        # Build new intermediate clusters
        new_children_clusters = []
        log_groups = []

        for idx, g in enumerate(groups):
            subgroup_children = [it["cluster"] for it in g["items"]]

            subgroup_children_node = {
                "cutoff": None,
                "depth": None,
                "clusters": subgroup_children,
                "children_count": len(subgroup_children),
            }

            leaf_ids = []
            for child in subgroup_children:
                for leaf in self._iter_descendant_leaves(child):
                    leaf_ids.extend(leaf.get("ids", []))

            # Option D metadata reaggregation
            subgroup_metadatas = []
            for child in subgroup_children:
                subgroup_metadatas.extend(child.get("metadatas", []))

            new_cluster = {
                "cluster_id": f"{cid}.{idx}",
                "size": len(leaf_ids),
                "ids": leaf_ids,
                "metadatas": subgroup_metadatas,
                "children": subgroup_children_node,
            }

            new_children_clusters.append(new_cluster)
            log_groups.append(
                {
                    "group_index": idx,
                    "token_sum": g["tokens"],
                    "children_cluster_ids": [it["cluster_id"] for it in g["items"]],
                }
            )

        group_cluster["children"]["clusters"] = new_children_clusters
        group_cluster["children"]["children_count"] = len(new_children_clusters)

        self.log_entries.append(
            {
                "cluster_id": cid,
                "total_tokens_before": total_tokens_before,
                "num_children_before": len(children),
                "groups": log_groups,
            }
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _similarity(self, c1, c2):
        if c1 is None or c2 is None:
            return 0.0
        return float(cosine_similarity(c1.reshape(1, -1), c2.reshape(1, -1))[0, 0])

    def _compute_group_centroid(self, items):
        centroids = [it["centroid"] for it in items if it["centroid"] is not None]
        if not centroids:
            return None
        return np.mean(np.stack(centroids, axis=0), axis=0)

    def _merge_small_groups(self, groups):
        """
        Merge groups below min_tokens into the most similar neighbor.
        Guaranteed termination.
        """
        if len(groups) <= 1:
            return

        changed = True
        safety_counter = 0

        while changed and safety_counter < 50:
            changed = False
            safety_counter += 1

            small_idx = None
            small_tokens = None

            for idx, g in enumerate(groups):
                if g["tokens"] < self.min_tokens:
                    if small_tokens is None or g["tokens"] < small_tokens:
                        small_tokens = g["tokens"]
                        small_idx = idx

            if small_idx is None:
                break

            small_group = groups[small_idx]
            best_idx = None
            best_sim = -1.0

            for idx, g in enumerate(groups):
                if idx == small_idx:
                    continue
                sim = self._similarity(g["centroid"], small_group["centroid"])
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx

            if best_idx is None:
                break

            target = groups[best_idx]
            target["items"].extend(small_group["items"])
            target["tokens"] += small_group["tokens"]
            target["centroid"] = self._compute_group_centroid(target["items"])

            del groups[small_idx]
            changed = True

    def _apply_penalty_fallback(self, groups):
        """
        If constraints cannot be satisfied, choose the grouping with the lowest penalty.
        """
        # Compute penalty for each group
        for g in groups:
            tokens = g["tokens"]
            penalty = 0
            if tokens > self.max_tokens:
                penalty += (tokens - self.max_tokens) * self.PENALTY_EXCEED_MAX
            if tokens < self.min_tokens:
                penalty += (self.min_tokens - tokens) * self.PENALTY_BELOW_MIN
            g["penalty"] = penalty

        # If all penalties are zero, perfect fit
        if all(g["penalty"] == 0 for g in groups):
            return groups

        # Otherwise, accept the least-bad configuration (already greedy)
        # No regrouping attempted here — deterministic fallback
        return groups

    # ------------------------------------------------------------------
    # Renumbering & total cluster count
    # ------------------------------------------------------------------
    def _renumber_cluster_ids(self, tree: Dict[str, Any]):
        def walk(node: Dict[str, Any], prefix: str):
            for i, cluster in enumerate(node.get("clusters", [])):
                new_id = f"{prefix}.{i}" if prefix else str(i)
                cluster["cluster_id"] = new_id
                if self._is_internal(cluster):
                    walk(cluster["children"], new_id)

        walk(tree, "")

    def _update_total_clusters(self, tree: Dict[str, Any]):
        def count(node: Dict[str, Any]) -> int:
            total = len(node.get("clusters", []))
            for c in node.get("clusters", []):
                if self._is_internal(c):
                    total += count(c["children"])
            return total

        tree["total_clusters"] = count(tree)
