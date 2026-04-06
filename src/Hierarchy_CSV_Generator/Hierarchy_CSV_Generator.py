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

    def __init__(self, hierarchy_json_path, output_dir, vector_db,
                 chunks_csv_filename, clusters_csv_filename):

        self.output_dir = output_dir
        self.hierarchy_json_path = hierarchy_json_path
        self.vector_db = vector_db
        self.chunks_csv_filename = chunks_csv_filename
        self.clusters_csv_filename = clusters_csv_filename

        # Internal accumulators
        self.all_clusters = []
        self.leaf_chunks = []           # list of (cluster_id, chunk_id)
        self.cluster_token_sums = {}    # cluster_id → token sum
        self.cluster_children_map = {}  # cluster_id → list of child cluster_ids
        self.cluster_parent_map = {}    # cluster_id → parent cluster_id

        # document_cluster flags
        self.document_cluster = defaultdict(lambda: False)

        # Document names per cluster
        self.cluster_documents = defaultdict(set)

        # Token contribution per document per cluster
        self.cluster_doc_token_sums = defaultdict(lambda: defaultdict(int))

        # Text class per cluster
        self.cluster_text_classes = defaultdict(set)

        # Token contribution per text_class per cluster
        self.cluster_textclass_token_sums = defaultdict(lambda: defaultdict(int))


    # ------------------------------------------------------------
    # PUBLIC ENTRYPOINT
    # ------------------------------------------------------------
    def generate(self):
        tree = self._load_json()
        self._walk_tree(tree, parent_id=None)

        self._detect_document_clusters_pass1()

        self._write_chunks_csv()

        self._aggregate_token_counts_upwards()

        self._write_clusters_csv()
        self._write_cluster_category_json_files()


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

            self.all_clusters.append(cluster)

            if parent_id is not None:
                self.cluster_parent_map[cid] = parent_id

            # children = null → leaf
            if children is None:
                self.cluster_children_map[cid] = []
            else:
                child_ids = [c["cluster_id"] for c in children["clusters"]]
                self.cluster_children_map[cid] = child_ids

            # LEAF NODE: collect chunk_ids
            if children is None:
                chunk_list = cluster.get("chunk_ids", [])
                for chunk_id in chunk_list:
                    self.leaf_chunks.append((cid, chunk_id))

            # RECURSE
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

            has_leaf_child = any(len(self.cluster_children_map[ch]) == 0 for ch in children)
            if has_leaf_child:
                self.document_cluster[cid] = True

    # ------------------------------------------------------------
    # AGGREGATE TOKEN_COUNT + DOCUMENT NAMES + TEXT CLASS UPWARD
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

                for doc, tok in self.cluster_doc_token_sums[child].items():
                    self.cluster_doc_token_sums[cid][doc] += tok

                # Text class aggregation
                self.cluster_text_classes[cid].update(
                    self.cluster_text_classes.get(child, set())
                )

                for cls, tok in self.cluster_textclass_token_sums[child].items():
                    self.cluster_textclass_token_sums[cid][cls] += tok


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
                "token_count",
                "chunk_text_class"
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
                text_class = meta.get("chunk_text_class", "")

                # Token aggregation
                self.cluster_token_sums[cluster_id] = \
                    self.cluster_token_sums.get(cluster_id, 0) + token_count

                # Document aggregation
                if document_name:
                    self.cluster_documents[cluster_id].add(document_name)
                    self.cluster_doc_token_sums[cluster_id][document_name] += token_count

                # Text class aggregation
                if text_class:
                    self.cluster_text_classes[cluster_id].add(text_class)
                    self.cluster_textclass_token_sums[cluster_id][text_class] += token_count

                writer.writerow([
                    cluster_id,
                    chunk_id,
                    chunk_type,
                    document_name,
                    pages,
                    heading_path,
                    image_paths,
                    token_count,
                    text_class
                ])


    # ------------------------------------------------------------
    # WRITE CLUSTERS CSV
    # ------------------------------------------------------------
    def _write_clusters_csv(self):

        def pad_cluster_id(cid):
            parts = str(cid).split(".")
            padded = ["{:02d}".format(int(p)) for p in parts]
            return ".".join(padded)

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
                "isLeaf",
                "allChildLeaf",
                "allChildInternal",
                "hasLeafSibling",
                "hasInternalSibling", 
                "parent",
                "level",
                "Source_Documents",
                "text_class"
            ])

            for cluster in self.all_clusters:
                cid = cluster["cluster_id"]
                size = cluster.get("size", 0)
                children = len(self.cluster_children_map.get(cid, []))

                children_list = self.cluster_children_map.get(cid, [])
                is_leaf = (len(children_list) == 0)
                all_child_leaf = all(len(self.cluster_children_map[ch]) == 0 for ch in children_list)
                all_child_internal = all(len(self.cluster_children_map[ch]) > 0 for ch in children_list)

                # Has leaf sibling
                parent = self.cluster_parent_map.get(cid)
                if parent is None:
                    has_leaf_sibling = False
                else:
                    siblings = self.cluster_children_map[parent]
                    has_leaf_sibling = any(
                        len(self.cluster_children_map[sib]) == 0
                        for sib in siblings
                        if sib != cid
                    )

                # Has internal sibling
                if parent is None:
                    has_internal_sibling = False
                else:
                    siblings = self.cluster_children_map[parent]
                    has_internal_sibling = any(
                        len(self.cluster_children_map[sib]) > 0
                        for sib in siblings
                        if sib != cid
                    )


                def compute_level(x):
                    lvl = 0
                    while x in self.cluster_parent_map:
                        x = self.cluster_parent_map[x]
                        lvl += 1
                    return lvl
                level = compute_level(cid)





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

                # Build text_class with percentages
                classes = []
                for cls in sorted(self.cluster_text_classes.get(cid, [])):
                    cls_tokens = self.cluster_textclass_token_sums[cid].get(cls, 0)
                    pct = (cls_tokens / token_sum * 100) if token_sum > 0 else 0
                    classes.append(f"{cls} ({pct:.1f}%)")

                text_class_str = "\n".join(classes)

                writer.writerow([
                    pad_cluster_id(cid),
                    size,
                    children,
                    summary,
                    label,
                    token_sum,
                    doc_cluster_flag,
                    is_leaf,
                    all_child_leaf,
                    all_child_internal,
                    has_leaf_sibling,
                    has_internal_sibling,
                    pad_cluster_id(parent) if parent != None else "",
                    level,
                    source_docs,
                    text_class_str
                ])


    # ------------------------------------------------------------
    # WRITE PER-CLUSTER CATEGORY JSON FILES
    # ------------------------------------------------------------
    def _write_cluster_category_json_files(self):

        for cluster in self.all_clusters:
            cid = cluster["cluster_id"]
            token_sum = self.cluster_token_sums.get(cid, 0)

            # Prepare directory
            cluster_dir = os.path.join(self.output_dir, str(cid))
            os.makedirs(cluster_dir, exist_ok=True)

            # Build JSON structure
            data = {
                "cluster_id": cid,
                "is_document_cluster": self.document_cluster[cid],
                "label": cluster.get("label", ""),
                "summary": cluster.get("summary", ""),
                "keywords": cluster.get("keywords", []),
                "source_documents": [],
                "text_class": []
            }

            # Source documents
            for doc in sorted(self.cluster_documents.get(cid, [])):
                doc_tokens = self.cluster_doc_token_sums[cid].get(doc, 0)
                pct = (doc_tokens / token_sum * 100) if token_sum > 0 else 0
                data["source_documents"].append({
                    "document_name": doc,
                    "tokens_percentage": round(pct, 2)
                })

            # Text classes
            for cls in sorted(self.cluster_text_classes.get(cid, [])):
                cls_tokens = self.cluster_textclass_token_sums[cid].get(cls, 0)
                pct = (cls_tokens / token_sum * 100) if token_sum > 0 else 0
                data["text_class"].append({
                    "class": cls,
                    "tokens_percentage": round(pct, 2)
                })

            # Write JSON file
            json_path = os.path.join(cluster_dir, f"{cid}_category.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
