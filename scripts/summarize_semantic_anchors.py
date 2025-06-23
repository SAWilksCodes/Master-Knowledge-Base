#!/usr/bin/env python3
"""
Summarize Semantic Anchors

Reads semantic_anchors.csv and generates a markdown summary with:
- Anchor type distribution
- Most common anchors
- Source distribution
- Cluster statistics
- Recent anchors
"""
import csv
import os
from collections import Counter, defaultdict

INPUT_CSV = 'semantic_intelligence/semantic_anchors.csv'
OUTPUT_MD = 'semantic_intelligence/semantic_anchor_summary.md'

MAX_COMMON = 20
MAX_RECENT = 10

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Input file not found: {INPUT_CSV}")
        return

    anchors = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            anchors.append(row)

    if not anchors:
        print("❌ No anchors found in CSV.")
        return

    # Type distribution
    type_counts = Counter(a['type'] for a in anchors)
    # Source distribution
    source_counts = Counter(a['source'] for a in anchors)
    # Cluster stats
    cluster_counts = Counter(a.get('cluster_id', -1) for a in anchors)
    # Most common anchors
    text_counts = Counter(a['text'].lower() for a in anchors)
    most_common = text_counts.most_common(MAX_COMMON)
    # Recent anchors
    recent_anchors = anchors[-MAX_RECENT:]

    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write("# Semantic Anchor Summary\n\n")
        f.write(f"Total anchors: {len(anchors)}\n\n")

        f.write("## Type Distribution\n\n")
        for t, c in type_counts.most_common():
            pct = (c / len(anchors)) * 100
            f.write(f"- **{t.title()}**: {c} ({pct:.1f}%)\n")
        f.write("\n")

        f.write("## Source Distribution\n\n")
        for s, c in source_counts.most_common():
            pct = (c / len(anchors)) * 100
            f.write(f"- **{s.title()}**: {c} ({pct:.1f}%)\n")
        f.write("\n")

        f.write("## Cluster Statistics\n\n")
        f.write(f"Total clusters: {len(cluster_counts)}\n")
        clustered = sum(count for cid, count in cluster_counts.items() if int(cid) >= 0)
        unclustered = cluster_counts.get('-1', 0)
        f.write(f"Clustered anchors: {clustered}\n")
        f.write(f"Unclustered anchors: {unclustered}\n\n")

        f.write("## Most Common Anchors\n\n")
        for text, count in most_common:
            f.write(f"- **{text}**: {count} occurrences\n")
        f.write("\n")

        f.write("## Recent Anchors\n\n")
        for a in recent_anchors:
            context = a.get('context', '')
            context_preview = context[:100] + ('...' if len(context) > 100 else '')
            f.write(f"- **{a['anchor_id']}**: {a['text']} ({a['type']}) - {context_preview}\n")
    print(f"✅ Summary written to {OUTPUT_MD}")

if __name__ == "__main__":
    main()
