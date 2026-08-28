import pandas as pd


# --- Function to avoid overlap ---
def adjust_positions(items, min_gap=0.08):
    items = sorted(items, key=lambda d: d["end_y"])

    for j in range(1, len(items)):
        prev = items[j - 1]
        curr = items[j]

        if curr["end_y"] - prev["end_y"] < min_gap:
            curr["end_y"] = prev["end_y"] + min_gap

    return items

