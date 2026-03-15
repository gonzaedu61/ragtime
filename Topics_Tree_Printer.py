import json

class Topics_Tree_Printer:
    """
    Loads clusters from VDB and reconstructs the hierarchy from cluster_id strings.
    FULL cluster_id is preserved exactly as stored in VDB.
    No reconstruction. No synthetic cid guessing.
    """

    def __init__(
        self,
        vdb,
        mode="details",
        color=True,
        show_label=False,
        hide_documents=False,
        show_full_cid=False,
        sort_order="cid",
        max_depth=None,
    ):
        self.vdb = vdb
        self.mode = mode
        self.color = color
        self.show_label = show_label
        self.hide_documents = hide_documents
        self.show_full_cid = show_full_cid
        self.sort_order = sort_order
        self.max_depth = max_depth

        # Colors
        if color:
            from colorama import Fore, Style, init
            init(autoreset=True)
            self.cyan = Fore.CYAN
            self.yellow = Fore.YELLOW
            self.green = Fore.GREEN
            self.reset = Style.RESET_ALL
        else:
            self.cyan = self.yellow = self.green = self.reset = ""

        # Load clusters
        self.clusters = self._load_clusters()

        # Build tree
        self.tree = self._build_tree()

    # ------------------------------------------------------------------
    # Load clusters
    # ------------------------------------------------------------------
    def _load_clusters(self):
        _, _, _, metadatas = self.vdb.get_all()
        clusters = {}

        for meta in metadatas:
            rj = meta.get("record_json")
            if not rj:
                continue

            rec = json.loads(rj)
            cid = rec["cluster_id"]

            clusters[cid] = rec

        return clusters

    # ------------------------------------------------------------------
    # Build hierarchy — FULL CID ALWAYS STORED
    # ------------------------------------------------------------------
    def _build_tree(self):
        root = {}

        for cid, data in self.clusters.items():
            parts = cid.split(".")
            node = root
            path = []

            for depth, part in enumerate(parts):
                path.append(part)
                full_path = ".".join(path)

                if part not in node:
                    node[part] = {
                        "cid": full_path,
                        "_data": None,
                        "children": {}
                    }

                if depth == len(parts) - 1:
                    node[part]["_data"] = data

                node = node[part]["children"]

        return root

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------
    def _cid_sort_key(self, cid):
        return tuple(int(p) for p in cid.split("."))

    # ------------------------------------------------------------------
    # Find real root
    # ------------------------------------------------------------------
    def _find_root(self):
        all_ids = set(self.clusters.keys())
        parents = set()

        for cid in all_ids:
            parts = cid.split(".")
            for i in range(1, len(parts)):
                parents.add(".".join(parts[:i]))

        roots = all_ids - parents
        return sorted(roots, key=self._cid_sort_key)[0]

    # ------------------------------------------------------------------
    # Count leaf chunks
    # ------------------------------------------------------------------
    def _count_leaf_chunks(self, node):
        total = 0
        data = node["_data"]
        if data:
            total += len(data.get("leaf_chunks", []))

        for child in node["children"].values():
            total += self._count_leaf_chunks(child)

        return total

    # ------------------------------------------------------------------
    # Print tree
    # ------------------------------------------------------------------
    def print_tree(self):
        root_cid = self._find_root()
        first = root_cid.split(".")[0]

        if first not in self.tree:
            print(f"(Root '{first}' not found)")
            return

        self._print_node(self.tree[first], "", 0, True)

    # ------------------------------------------------------------------
    # Recursive printing — FULL CID ALWAYS USED
    # ------------------------------------------------------------------
    def _print_node(self, node, prefix, depth, is_last):
        data = node["_data"]
        real_cid = node["cid"]

        # Synthetic node → recurse only
        if not data:
            children = sorted(
                node["children"].values(),
                key=lambda n: self._cid_sort_key(n["cid"])
            )
            last_index = len(children) - 1
            for i, child in enumerate(children):
                is_last = (i == last_index)
                self._print_node(child, prefix + "│   ", depth + 1, is_last)
            return


        chunks = self._count_leaf_chunks(node)
        child_count = len(node["children"])

        cid_to_show = f"[{real_cid}]" if self.show_full_cid else ""

        line = (
            f"{self.cyan}{cid_to_show}{self.reset} "
            f"chunks={chunks} | childs={child_count}"
        )

        if self.show_label and data.get("label"):
            line += f": {self.green}{data['label']}{self.reset}"



        if (child_count == 0 and self.hide_documents and is_last):
            print(prefix + "└── " + line)
        else:
            print(prefix + "├── " + line)



        # Print docs
        self._print_summary(data, prefix, True)

        # Recurse
        children = sorted(
            node["children"].values(),
            key=lambda n: self._cid_sort_key(n["cid"])
        )

        last_index = len(children) - 1
        for i, child in enumerate(children):
            is_last = (i == last_index)
            self._print_node(child, prefix + "|   ", depth + 1, is_last)

    # ------------------------------------------------------------------
    # Print Docs summary
    # ------------------------------------------------------------------
    def _print_summary(self, data, doc_prefix, is_last):
        docs = data["metadata"].get("source_documents", [])
        last = len(docs) - 1

        doc_prefix = doc_prefix + "|   "

        for i, doc in enumerate(docs):
            print(
                f"{doc_prefix}{doc['document_name']} | {doc['tokens_percentage']:.1f}%"
            )

