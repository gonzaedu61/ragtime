import json
import csv
import os
from collections import defaultdict


class Hierarchy_CSV_Generator:
    """
    Generates two CSV files:
      1. chunks.csv   – one row per chunk (leaf-level)
      2. clusters.csv – one row per cluster node (all levels)

    Inputs:
      - hierarchy_json_path: path to the JSON hierarchy file
      - vector_db: an object exposing .get_by_id(chunk_id)
      - chunks_csv_filename: output filename for chunk-level CSV
      - clusters_csv_filename: output filename for cluster-level CSV
    """

    def __init__(self, hierarchy_json_path, vector_db,
                 chunks_csv_filename, clusters_csv_filename):

        self.hierarchy_json_path = hierarchy_json_path
        self.vector_db = vector_db
        self.chunks_csv_filename = chunks_csv_filename
        self.clusters_csv_filename = clusters_csv_filename

        # Output directory = same as JSON hierarchy
        self.output_dir = os.path.dirname(os.path.abspath(hierarchy_json_path))

        # Internal accumulators
        self.all_clusters = []          # list of cluster dicts
        self.leaf_chunks = []           # list of (cluster_id, chunk_id)
        self.cluster_token_sums = {}    # cluster_id → token sum
        self.cluster_children_map = {}  # cluster_id → list of child cluster_ids
        self.cluster_parent_map = {}    # cluster_id → parent cluster_id

        # document_cluster flags
        self.document_cluster = defaultdict(lambda: False)

        # NEW: document names per cluster
        self.cluster_documents = defaultdict(set)

        # NEW: token contribution per document per cluster
        self.cluster_doc_token_sums = defaultdict(lambda: defaultdict(int))


    # ------------------------------------------------------------
    # PUBLIC ENTRYPOINT
    # ------------------------------------------------------------
    def generate(self):
        tree = self._load_json()
        self._walk_tree(tree, parent_id=None)

        # First pass: detect low-level parent clusters
        self._detect_document_clusters_pass1()

        # Second pass: detect leaf clusters with document_cluster siblings
        self._detect_document_clusters_pass2()

        # Write chunks CSV (collects leaf-level token counts + documents)
        self._write_chunks_csv()

        # Aggregate token counts + document names upward
        self._aggregate_token_counts_upwards()

        # Write clusters CSV
        self._write_clusters_csv()


    # ------------------------------------------------------------
    # LOAD JSON
    # ------------------------------------------------------------
    def _load_json(self):
        with open(self.hierarchy_json_path, "r", encoding="utf-8") as f:
            return json.load(f)


    # ------------------------------------------------------------
    # TREE WALKING
    # ------------------------------------------------------------
    def _walk_tree(self, node, parent_id):
        clusters = node.get("clusters", [])
        for cluster in clusters:
            cid = cluster["cluster_id"]
            children = cluster.get("children")

            # Register cluster
            self.all_clusters.append(cluster)

            # Track parent
            if parent_id is not None:
                self.cluster_parent_map[cid] = parent_id

            # Track children
            if children is None:
                self.cluster_children_map[cid] = []
            else:
                child_ids = [c["cluster_id"] for c in children["clusters"]]
                self.cluster_children_map[cid] = child_ids

            # Leaf node → collect chunk ids
            if children is None:
                for chunk_id in cluster.get("ids", []):
                    self.leaf_chunks.append((cid, chunk_id))

            # Recurse
            if children is not None:
                self._walk_tree(children, parent_id=cid)


    # ------------------------------------------------------------
    # DOCUMENT CLUSTER DETECTION — PASS 1
    # ------------------------------------------------------------
    def _detect_document_clusters_pass1(self):
        for cluster in self.all_clusters:
            cid = cluster["cluster_id"]
            children = self.cluster_children_map.get(cid, [])

            if not children:
                continue

            all_leaf = all(len(self.cluster_children_map[ch]) == 0 for ch in children)

            if all_leaf:
                self.document_cluster[cid] = True


    # ------------------------------------------------------------
    # DOCUMENT CLUSTER DETECTION — PASS 2
    # ------------------------------------------------------------
    def _detect_document_clusters_pass2(self):
        for cluster in self.all_clusters:
            cid = cluster["cluster_id"]
            children = self.cluster_children_map.get(cid, [])

            if children:
                continue

            parent = self.cluster_parent_map.get(cid)
            if parent is None:
                continue

            siblings = self.cluster_children_map[parent]
            if any(self.document_cluster[sib] for sib in siblings):
                self.document_cluster[cid] = True


    # ------------------------------------------------------------
    # AGGREGATE TOKEN_COUNT + DOCUMENT NAMES UPWARD
    # ------------------------------------------------------------
    def _aggregate_token_counts_upwards(self):

        def depth(cid):
            d = 0
            while cid in self.cluster_parent_map:
                cid = self.cluster_parent_map[cid]
                d += 1
            return d

        sorted_clusters = sorted(
            self.cluster_children_map.keys(),
            key=lambda cid: depth(cid),
            reverse=True
        )

        for cid in sorted_clusters:
            children = self.cluster_children_map.get(cid, [])
            for child in children:

                # Token aggregation
                self.cluster_token_sums[cid] = \
                    self.cluster_token_sums.get(cid, 0) + \
                    self.cluster_token_sums.get(child, 0)

                # Document aggregation
                self.cluster_documents[cid].update(
                    self.cluster_documents.get(child, set())
                )

                # Document token aggregation
                for doc, tok in self.cluster_doc_token_sums[child].items():
                    self.cluster_doc_token_sums[cid][doc] += tok


    # ------------------------------------------------------------
    # WRITE CHUNKS CSV
    # ------------------------------------------------------------
    def _write_chunks_csv(self):
        path = os.path.join(self.output_dir, self.chunks_csv_filename)

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")

            writer.writerow([
                "cluster_id",
                "chunk_id",
                "chunk_type",
                "document_name",
                "pages",
                "heading_path",
                "image_paths",
                "token_count"
            ])

            for cluster_id, chunk_id in self.leaf_chunks:

                record = self.vector_db.get_by_id(chunk_id)
                if record is None:
                    continue

                meta = record["metadata"]

                chunk_type = meta.get("chunk_type", "")
                document_name = meta.get("document_name", "")
                pages = "\n".join(str(p) for p in meta.get("pages", []))
                heading_path = "\n".join(meta.get("heading_path", []))
                image_paths = "\n".join(meta.get("image_paths", []))
                token_count = meta.get("token_count", 0)

                # Token aggregation
                self.cluster_token_sums[cluster_id] = \
                    self.cluster_token_sums.get(cluster_id, 0) + token_count

                # Document aggregation
                if document_name:
                    self.cluster_documents[cluster_id].add(document_name)
                    self.cluster_doc_token_sums[cluster_id][document_name] += token_count

                writer.writerow([
                    cluster_id,
                    chunk_id,
                    chunk_type,
                    document_name,
                    pages,
                    heading_path,
                    image_paths,
                    token_count
                ])


    # ------------------------------------------------------------
    # WRITE CLUSTERS CSV
    # ------------------------------------------------------------
    def _write_clusters_csv(self):
        path = os.path.join(self.output_dir, self.clusters_csv_filename)

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")

            writer.writerow([
                "cluster_id",
                "size",
                "children",
                "summary",
                "label",
                "token_count",
                "document_cluster",
                "Source_Documents"
            ])

            for cluster in self.all_clusters:
                cid = cluster["cluster_id"]
                size = cluster.get("size", 0)
                children = len(self.cluster_children_map.get(cid, []))
                summary = cluster.get("summary", "")
                label = cluster.get("label", "")
                token_sum = self.cluster_token_sums.get(cid, 0)
                doc_cluster_flag = self.document_cluster[cid]

                # Build Source_Documents with percentages
                docs = []
                for doc in sorted(self.cluster_documents.get(cid, [])):
                    doc_tokens = self.cluster_doc_token_sums[cid].get(doc, 0)
                    pct = (doc_tokens / token_sum * 100) if token_sum > 0 else 0
                    docs.append(f"{doc} ({pct:.1f}%)")

                source_docs = "\n".join(docs)

                writer.writerow([
                    cid,
                    size,
                    children,
                    summary,
                    label,
                    token_sum,
                    doc_cluster_flag,
                    source_docs
                ])
