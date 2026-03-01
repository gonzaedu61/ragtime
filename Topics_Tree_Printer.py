from collections import Counter

class Topics_Tree_Printer:
    def __init__(self, mode="details", color=True,
                 show_label=False, hide_documents=False,
                 show_full_cid=False, sort_order="cid",
                 max_depth=None):

        self.mode = mode
        self.color = color
        self.show_label = show_label
        self.hide_documents = hide_documents
        self.show_full_cid = show_full_cid
        self.sort_order = sort_order
        self.max_depth = max_depth

        if color:
            from colorama import Fore, Style, init
            init(autoreset=True)

            self.cyan = Fore.CYAN
            self.yellow = Fore.YELLOW
            self.green = Fore.GREEN
            self.reset = Style.RESET_ALL
        else:
            self.cyan = ""
            self.yellow = ""
            self.green = ""
            self.reset = ""

    def _cid_sort_key(self, cid):
        # If it's already an int, just wrap it
        if isinstance(cid, int):
            return (cid,)

        # Otherwise assume something like "0.0.0.10"
        parts = str(cid).split(".")
        return tuple(int(p) for p in parts if p != "")


    def print_tree(self, node, prefix="", cid_path="", depth=0):
        if node is None:
            print(prefix + "(empty)")
            return

        clusters = node["clusters"]

        # Always sort the clusters we are about to print
        if self.sort_order == "cid":
            clusters = sorted(
                clusters,
                key=lambda c: self._cid_sort_key(c["cluster_id"])
            )
        elif self.sort_order == "size":
            clusters = sorted(clusters, key=lambda c: c["size"], reverse=True)

        last_index = len(clusters) - 1

        for i, cluster in enumerate(clusters):
            is_last = i == last_index

            cid = cluster["cluster_id"]
            full_cid = f"{cid_path}.{cid}" if cid_path else str(cid)
            cid_to_show = full_cid if self.show_full_cid else cid

            branch = "└── " if is_last else "├── "

            chunks = cluster["size"]
            child_count = (
                len(cluster["children"]["clusters"])
                if cluster["children"] else 0
            )

            line = (
                f"{self.cyan}[{cid_to_show}]{self.reset} "
                f"chunks={chunks} | childs={child_count}"
            )

            label = cluster.get("label")
            if self.show_label and label:
                line += f": {self.green}{label}{self.reset}"

            print(prefix + branch + line)

            # Print details or summary for THIS cluster
            if self.mode == "details":
                self._print_details(cluster, prefix, is_last)
            else:
                self._print_summary(cluster, prefix, is_last)

            # If we've reached max depth, do NOT recurse into children
            if self.max_depth is not None and depth >= self.max_depth:
                continue

            # Recurse into children
            if cluster["children"] is not None:
                new_prefix = prefix + ("    " if is_last else "│   ")
                new_cid_path = full_cid
                self.print_tree(
                    cluster["children"],
                    new_prefix,
                    new_cid_path,
                    depth + 1
                )

    def _print_details(self, cluster, prefix, is_last):
        if self.hide_documents:
            return

        metadatas = cluster["metadatas"]
        last_index = len(metadatas) - 1

        for i, meta in enumerate(metadatas):
            doc = meta.get("document_name", "unknown")
            pages = meta.get("pages", [])
            chunk = meta.get("chunk_id", "n/a")

            branch = "└── " if i == last_index else "├── "
            sub_prefix = prefix + ("    " if is_last else "│   ")

            print(
                sub_prefix + branch +
                f"{self.yellow}{doc}{self.reset} | pages={pages} | chunk={chunk}"
            )

    def _print_summary(self, cluster, prefix, is_last):
        metadatas = cluster["metadatas"]
        total_chunks = len(metadatas)

        counts = Counter(meta.get("document_name", "unknown") for meta in metadatas)
        items = list(counts.items())
        last_index = len(items) - 1

        for i, (doc, count) in enumerate(items):
            pct = (count / total_chunks) * 100 if total_chunks else 0

            branch = "└── " if i == last_index else "├── "
            sub_prefix = prefix + ("    " if is_last else "│   ")

            print(
                sub_prefix + branch +
                f"{self.green}{doc}{self.reset} | chunks={count} | {pct:.1f}%"
            )
