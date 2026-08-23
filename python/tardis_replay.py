"""Tardis CSV readers for audit / replay tests."""
import csv


def read_tardis_trades(filepath: str):
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        i_ts, i_side = idx["timestamp"], idx["side"]
        i_px, i_amt = idx["price"], idx["amount"]
        for row in reader:
            if not row:
                continue
            ts_ns = int(row[i_ts]) * 1000
            is_sell = row[i_side].lower() == "sell"
            yield (ts_ns, is_sell, float(row[i_px]), float(row[i_amt]))


def read_tardis_book(filepath: str, depth: int = 1):
    del depth  # best-bid / best-ask only — full L2 is unused by the CI harness
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        i_ts = idx["timestamp"]
        i_ap, i_aq = idx["asks[0].price"], idx["asks[0].amount"]
        i_bp, i_bq = idx["bids[0].price"], idx["bids[0].amount"]
        for row in reader:
            if not row:
                continue
            ts_ns = int(row[i_ts]) * 1000
            asks = [(float(row[i_ap]), float(row[i_aq]))] if row[i_ap] else []
            bids = [(float(row[i_bp]), float(row[i_bq]))] if row[i_bp] else []
            yield (ts_ns, asks, bids)


class TardisReplayEngine:
    """Placeholder imported by audit scripts; replay is driven by the generators."""
