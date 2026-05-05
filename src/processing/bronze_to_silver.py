"""
bronze_to_silver.py — Clean and deduplicate Bronze → Silver Delta table.

Transformations:
  - Drop duplicates on (text, source)
  - Normalize labels to binary: 0 = credible, 1 = misinformation
  - Strip HTML, normalize whitespace
  - Add word_count, char_count feature columns
  - Partition by source

TODO: Implement in Step 3 (Processing Layer)
"""

from __future__ import annotations


def run() -> None:
    raise NotImplementedError("Implement in Step 3")
