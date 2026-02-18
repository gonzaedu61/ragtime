# Topics_Tree_Printer.py
from collections import Counter

class Topics_Tree_Printer:
    def __init__(self, mode="details", color=True):
        self.mode = mode
        self.color = color

        if color:
            # Import colorama ONLY when needed
            from colorama import Fore, Style, init
            init(autoreset=True)

            self.cyan = Fore.CYAN
            self.yellow = Fore.YELLOW
            self.green = Fore.GREEN
            self.reset = Style.RESET_ALL
        else:
            # No ANSI codes at all
            self.cyan = ""
            self.yellow = ""
            self.green = ""
            self.reset = ""

    def print_tree(self, node, prefix=""):
        if node is None:
            print(prefix + "(empty)")
            return

        clusters = node["clusters"]
        last_index = len(clusters) - 1

        for i, cluster in enumerate(clusters):
            is_last = i == last_index

            if self.mode == "details":
                branch = "└── " if is_last else "├── "
                cid = cluster["cluster_id"]
                size = cluster["size"]

                print(prefix + branch +
                      f"{self.cyan}[{cid}]{self.reset} size={size}")

                self._print_details(cluster, prefix, is_last)

            else:
                self._print_summary(cluster, prefix, is_last)

            if cluster["children"] is not None:
                new_prefix = prefix + ("    " if is_last else "│   ")
                self.print_tree(cluster["children"], new_prefix)

    def _print_details(self, cluster, prefix, is_last):
        metadatas = cluster["metadatas"]
        last_index = len(metadatas) - 1

        for i, meta in enumerate(metadatas):
            doc = meta.get("document_name", "unknown")
            pages = meta.get("pages", [])
            chunk = meta.get("chunk_id", "n/a")

            branch = "└── " if i == last_index else "├── "
            sub_prefix = prefix + ("    " if is_last else "│   ")

            print(sub_prefix + branch +
                  f"{self.yellow}{doc}{self.reset} | pages={pages} | chunk={chunk}")

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

            print(sub_prefix + branch +
                  f"{self.green}{doc}{self.reset} | chunks={count} | {pct:.1f}%")
