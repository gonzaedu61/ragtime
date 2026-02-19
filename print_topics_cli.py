#!/usr/bin/env python3
import argparse
import json
import sys
from io import StringIO
from Topics_Tree_Printer import Topics_Tree_Printer

def main():
    parser = argparse.ArgumentParser(description="Pretty-print a cluster tree.")
    parser.add_argument("input_file", help="JSON file containing the tree structure")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("-dd", "--document-details", action="store_true",
                            help="Show full document details (default)")
    mode_group.add_argument("-ds", "--document-summary", action="store_true",
                            help="Show document summary instead of details")

    parser.add_argument("-o", "--output", metavar="FILE",
                        help="Save output to a file instead of printing to console")

    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colors in output")

    # NEW OPTIONS
    parser.add_argument("-cl", "--cluster-label", action="store_true",
                        help="Show cluster label")
    parser.add_argument("-dn", "--documents-no", action="store_true",
                        help="Do not print document info lines")
    parser.add_argument("-cid", "--cluster-id", action="store_true",
                        help="Print full hierarchical cluster ID")

    args = parser.parse_args()

    # Determine mode
    mode = "summary" if args.document_summary else "details"

    # Load tree JSON
    with open(args.input_file, "r") as f:
        tree = json.load(f)

    # Determine color usage
    if args.output:
        use_color = not args.no_color
    else:
        use_color = sys.stdout.isatty() and not args.no_color

    # Capture output in a buffer
    buffer = StringIO()
    original_stdout = sys.stdout
    sys.stdout = buffer

    try:
        printer = Topics_Tree_Printer(
            mode=mode,
            color=use_color,
            show_label=args.cluster_label,
            hide_documents=args.documents_no,
            show_full_cid=args.cluster_id
        )
        printer.print_tree(tree)
    finally:
        sys.stdout = original_stdout

    output_text = buffer.getvalue()

    # Write to file or console
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Output saved to {args.output}")
    else:
        print(output_text)

if __name__ == "__main__":
    main()
