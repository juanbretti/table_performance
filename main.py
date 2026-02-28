"""
Performance benchmark: pandas vs DuckDB vs polars
==================================================
Tables
------
  large_table.parquet  – 5 000 000 rows × 20 cols
      num_1 … num_10   float32   (10 numeric)
      bool_1, bool_2   bool      (2 boolean)
      text_1           str       join key → small_table (10 K unique values)
      text_2           str       5 categories (alpha/beta/gamma/delta/epsilon) – used for pivot
      text_3 … text_8  str       random text up to 20 chars with spaces

  small_table.parquet  – 10 000 rows × 10 cols
      text_1           str       unique join key
      num_1 … num_5    float32   (5 numeric)
      bool_1           bool      (1 boolean)
      text_2 … text_4  str       random text up to 20 chars with spaces

Benchmarked operations (time + peak Python RAM via tracemalloc)
---------------------------------------------------------------
  1. Read parquet
  2. Filter on numeric column  (num_1 > 500)
  3. Filter on text column     (text_3 starts with 'a')
  4. Filter on boolean column  (bool_1 == True)
  5. Text transform            (upper-case + keep text before first space)
  6. Merge large × small       (inner join on text_1)
  7. Pivot merged table        (index=bool_2, columns=text_2, values=num_1, agg=mean)
  8. Save result to parquet
"""

import os
import string
import threading
import time
from pathlib import Path

import psutil

import duckdb
import numpy as np
import pandas as pd
import polars as pl

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
LARGE_PARQUET  = BASE_DIR / "large_table.parquet"
SMALL_PARQUET  = BASE_DIR / "small_table.parquet"
# Forward-slash versions for DuckDB SQL strings on Windows
_LARGE_SQL = str(LARGE_PARQUET).replace("\\", "/")
_SMALL_SQL = str(SMALL_PARQUET).replace("\\", "/")

# ── Constants ──────────────────────────────────────────────────────────────────
LARGE_ROWS  = 5_000_000
SMALL_ROWS  = 10_000
SEED        = 42
CATEGORIES  = ["alpha", "beta", "gamma", "delta", "epsilon"]

# ── Measurement helpers ────────────────────────────────────────────────────────

def measure(fn):
    """Run *fn()*, return (result, elapsed_s, peak_delta_mib).

    Memory is measured as the peak RSS delta of the current process during
    the call, sampled every 5 ms from a background thread.  This captures
    native allocations from Rust (Polars) and C++ (DuckDB) in addition to
    Python-heap objects.
    """
    proc      = psutil.Process()
    baseline  = proc.memory_info().rss
    peak_rss  = [baseline]
    stop      = threading.Event()

    def _sampler() -> None:
        while not stop.is_set():
            try:
                rss = proc.memory_info().rss
                if rss > peak_rss[0]:
                    peak_rss[0] = rss
            except Exception:
                pass
            stop.wait(0.005)          # sample every 5 ms

    t = threading.Thread(target=_sampler, daemon=True)
    t.start()
    t0     = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    stop.set()
    t.join()

    peak_mib = max(0.0, (peak_rss[0] - baseline) / 1_048_576)
    return result, elapsed, peak_mib


def _row(label: str, elapsed: float, peak_mib: float) -> None:
    print(f"    {label:<52} {elapsed:>8.3f} s   {peak_mib:>8.1f} MiB")


def _header(title: str) -> None:
    bar = "─" * 76
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
    print(f"    {'Operation':<52} {'Time':>9}   {'Peak RAM':>9}")
    print(f"    {'─'*52} {'─'*9}   {'─'*9}")

# ── Text-pool helpers ──────────────────────────────────────────────────────────

def _build_pool(pool_size: int, seed: int) -> list[str]:
    """Return *pool_size* random strings of 6–20 lowercase letters + at least
    one space so that 'split on space' always returns two parts."""
    rng   = np.random.default_rng(seed)
    chars = list(string.ascii_lowercase)
    pool  = []
    for _ in range(pool_size):
        length = int(rng.integers(6, 21))
        letters = rng.choice(chars, size=length)
        space_pos = int(rng.integers(2, max(3, length - 1)))
        letters[space_pos] = " "
        pool.append("".join(letters))
    return pool


def _sample_array(pool: list[str], n: int, seed: int) -> np.ndarray:
    """Fast vectorised sampling via numpy fancy-indexing."""
    rng = np.random.default_rng(seed)
    arr = np.array(pool)
    return arr[rng.integers(0, len(pool), size=n)]


# ── Data generation ────────────────────────────────────────────────────────────

def generate_data() -> None:
    """Create and save both parquet files if they do not already exist."""
    if LARGE_PARQUET.exists() and SMALL_PARQUET.exists():
        print("Parquet files already exist – skipping generation.")
        return

    print("\n► Generating datasets …\n")

    # ── small_table (10 K × 10) ────────────────────────────────────────────────
    print("  Building small_table.parquet  (10 000 rows × 10 cols) …")
    rng_s = np.random.default_rng(SEED + 10)

    # text_1 acts as the unique primary key (one string per row)
    small_text1_pool = _build_pool(SMALL_ROWS, seed=SEED + 11)   # 10 K unique keys
    small_text_pool  = _build_pool(300, seed=SEED + 12)

    small_df = pd.DataFrame({
        "text_1": small_text1_pool,                                         # unique PK
        "num_1":  rng_s.random(SMALL_ROWS).astype(np.float32) * 500,
        "num_2":  rng_s.random(SMALL_ROWS).astype(np.float32) * 500,
        "num_3":  rng_s.random(SMALL_ROWS).astype(np.float32) * 500,
        "num_4":  rng_s.random(SMALL_ROWS).astype(np.float32) * 500,
        "num_5":  rng_s.random(SMALL_ROWS).astype(np.float32) * 500,
        "bool_1": rng_s.choice([True, False], size=SMALL_ROWS),
        "text_2": _sample_array(small_text_pool, SMALL_ROWS, seed=SEED + 13),
        "text_3": _sample_array(small_text_pool, SMALL_ROWS, seed=SEED + 14),
        "text_4": _sample_array(small_text_pool, SMALL_ROWS, seed=SEED + 15),
    })
    small_df.to_parquet(SMALL_PARQUET, index=False)
    _mb = os.path.getsize(SMALL_PARQUET) / 1_048_576
    print(f"    Saved → {SMALL_PARQUET.name}  ({_mb:.1f} MiB)")

    # ── large_table (5 M × 20) ─────────────────────────────────────────────────
    print("  Building large_table.parquet  (5 000 000 rows × 20 cols) …")
    rng_l = np.random.default_rng(SEED + 20)

    # text_1: FK into small_table (5 M values drawn from the 10 K unique keys)
    large_text1 = np.array(small_text1_pool)[
        rng_l.integers(0, SMALL_ROWS, size=LARGE_ROWS)
    ]
    large_text_pool = _build_pool(500, seed=SEED + 21)

    large_df = pd.DataFrame({
        "num_1":  rng_l.random(LARGE_ROWS).astype(np.float32) * 1000,
        "num_2":  rng_l.random(LARGE_ROWS).astype(np.float32) * 1000,
        "num_3":  rng_l.random(LARGE_ROWS).astype(np.float32) * 1000,
        "num_4":  rng_l.random(LARGE_ROWS).astype(np.float32) * 1000,
        "num_5":  rng_l.random(LARGE_ROWS).astype(np.float32) * 1000,
        "num_6":  rng_l.random(LARGE_ROWS).astype(np.float32) * 1000,
        "num_7":  rng_l.random(LARGE_ROWS).astype(np.float32) * 1000,
        "num_8":  rng_l.random(LARGE_ROWS).astype(np.float32) * 1000,
        "num_9":  rng_l.random(LARGE_ROWS).astype(np.float32) * 1000,
        "num_10": rng_l.random(LARGE_ROWS).astype(np.float32) * 1000,
        "bool_1": rng_l.choice([True, False], size=LARGE_ROWS),
        "bool_2": rng_l.choice([True, False], size=LARGE_ROWS),
        "text_1": large_text1,                                       # FK → small.text_1
        "text_2": rng_l.choice(CATEGORIES, size=LARGE_ROWS),        # 5 cats for pivot
        "text_3": _sample_array(large_text_pool, LARGE_ROWS, seed=SEED + 22),
        "text_4": _sample_array(large_text_pool, LARGE_ROWS, seed=SEED + 23),
        "text_5": _sample_array(large_text_pool, LARGE_ROWS, seed=SEED + 24),
        "text_6": _sample_array(large_text_pool, LARGE_ROWS, seed=SEED + 25),
        "text_7": _sample_array(large_text_pool, LARGE_ROWS, seed=SEED + 26),
        "text_8": _sample_array(large_text_pool, LARGE_ROWS, seed=SEED + 27),
    })
    large_df.to_parquet(LARGE_PARQUET, index=False)
    _mb = os.path.getsize(LARGE_PARQUET) / 1_048_576
    print(f"    Saved → {LARGE_PARQUET.name}  ({_mb:.1f} MiB)\n")


# ── PANDAS benchmark ───────────────────────────────────────────────────────────

def bench_pandas() -> None:
    _header("PANDAS")

    # 1. Read
    large_pdf, t, m = measure(lambda: pd.read_parquet(LARGE_PARQUET))
    _row("Read large parquet", t, m)

    small_pdf = pd.read_parquet(SMALL_PARQUET)     # not timed – setup only

    # 2. Filter numeric
    _, t, m = measure(lambda: large_pdf[large_pdf["num_1"] > 500])
    _row("Filter numeric  (num_1 > 500)", t, m)

    # 3. Filter text
    _, t, m = measure(lambda: large_pdf[large_pdf["text_3"].str.startswith("a")])
    _row("Filter text  (text_3 starts with 'a')", t, m)

    # 4. Filter boolean
    _, t, m = measure(lambda: large_pdf[large_pdf["bool_1"]])
    _row("Filter boolean  (bool_1 == True)", t, m)

    # 5. Text transform: uppercase → split on space → first token
    _, t, m = measure(
        lambda: large_pdf["text_3"].str.upper().str.split(" ").str[0]
    )
    _row("Text transform  (upper + first word)", t, m)

    # 6. Merge
    merged_pdf, t, m = measure(
        lambda: large_pdf.merge(
            small_pdf, on="text_1", how="inner",
            suffixes=("_large", "_small")
        )
    )
    _row("Merge inner join  (on text_1)", t, m)

    # 7. Pivot  – index=bool_2, columns=text_2_large (5 cats), values=num_1_large
    def _pivot_pandas():
        return (
            merged_pdf
            .pivot_table(
                index="bool_2",
                columns="text_2_large",
                values="num_1_large",
                aggfunc="mean",
            )
            .reset_index()
        )

    pivot_pdf, t, m = measure(_pivot_pandas)
    _row("Pivot  (bool_2 × text_2, mean num_1)", t, m)

    # 8. Save
    out = str(BASE_DIR / "output_pandas.parquet")
    _, t, m = measure(lambda: pivot_pdf.to_parquet(out, index=False))
    _row("Save pivot → output_pandas.parquet", t, m)


# ── POLARS benchmark ───────────────────────────────────────────────────────────

def bench_polars() -> None:
    _header("POLARS")

    # 1. Read
    large_pl, t, m = measure(lambda: pl.read_parquet(LARGE_PARQUET))
    _row("Read large parquet", t, m)

    small_pl = pl.read_parquet(SMALL_PARQUET)      # not timed – setup only

    # 2. Filter numeric
    _, t, m = measure(lambda: large_pl.filter(pl.col("num_1") > 500))
    _row("Filter numeric  (num_1 > 500)", t, m)

    # 3. Filter text
    _, t, m = measure(
        lambda: large_pl.filter(pl.col("text_3").str.starts_with("a"))
    )
    _row("Filter text  (text_3 starts with 'a')", t, m)

    # 4. Filter boolean
    _, t, m = measure(lambda: large_pl.filter(pl.col("bool_1")))
    _row("Filter boolean  (bool_1 == True)", t, m)

    # 5. Text transform
    _, t, m = measure(
        lambda: large_pl.with_columns(
            pl.col("text_3")
            .str.to_uppercase()
            .str.split(" ")
            .list.first()
            .alias("text_3_first")
        )
    )
    _row("Text transform  (upper + first word)", t, m)

    # 6. Merge  (conflicting cols from small get _right suffix)
    merged_pl, t, m = measure(
        lambda: large_pl.join(small_pl, on="text_1", how="inner")
    )
    _row("Merge inner join  (on text_1)", t, m)

    # 7. Pivot  – text_2 stays unambiguous (large has it; small's becomes text_2_right)
    def _pivot_polars():
        return merged_pl.pivot(
            on="text_2",
            index="bool_2",
            values="num_1",
            aggregate_function="mean",
        )

    pivot_pl, t, m = measure(_pivot_polars)
    _row("Pivot  (bool_2 × text_2, mean num_1)", t, m)

    # 8. Save
    out = str(BASE_DIR / "output_polars.parquet")
    _, t, m = measure(lambda: pivot_pl.write_parquet(out))
    _row("Save pivot → output_polars.parquet", t, m)


# ── DUCKDB benchmark ───────────────────────────────────────────────────────────

def bench_duckdb() -> None:
    _header("DUCKDB")

    con = duckdb.connect()                         # in-memory database

    # 1. Read – materialize large parquet into a DuckDB in-memory table
    def _read_large():
        con.execute(
            f"CREATE OR REPLACE TABLE large_t AS "
            f"SELECT * FROM read_parquet('{_LARGE_SQL}')"
        )

    _, t, m = measure(_read_large)
    _row("Read large parquet (→ in-memory table)", t, m)

    con.execute(
        f"CREATE OR REPLACE TABLE small_t AS "
        f"SELECT * FROM read_parquet('{_SMALL_SQL}')"
    )                                              # not timed – setup only

    # 2. Filter numeric
    _, t, m = measure(
        lambda: con.execute("SELECT * FROM large_t WHERE num_1 > 500").fetchdf()
    )
    _row("Filter numeric  (num_1 > 500)", t, m)

    # 3. Filter text
    _, t, m = measure(
        lambda: con.execute(
            "SELECT * FROM large_t WHERE text_3 LIKE 'a%'"
        ).fetchdf()
    )
    _row("Filter text  (text_3 starts with 'a')", t, m)

    # 4. Filter boolean
    _, t, m = measure(
        lambda: con.execute(
            "SELECT * FROM large_t WHERE bool_1 = true"
        ).fetchdf()
    )
    _row("Filter boolean  (bool_1 == True)", t, m)

    # 5. Text transform  – upper() + split_part(col, ' ', 1) = text before first space
    _, t, m = measure(
        lambda: con.execute(
            "SELECT *, upper(split_part(text_3, ' ', 1)) AS text_3_first FROM large_t"
        ).fetchdf()
    )
    _row("Text transform  (upper + first word)", t, m)

    # 6. Merge – create a merged table to reuse in pivot
    def _merge_duckdb():
        con.execute(
            """
            CREATE OR REPLACE TABLE merged_t AS
            SELECT
                l.num_1,  l.num_2,  l.num_3,  l.num_4,  l.num_5,
                l.num_6,  l.num_7,  l.num_8,  l.num_9,  l.num_10,
                l.bool_1 AS bool_1_large,
                l.bool_2,
                l.text_1,
                l.text_2,
                l.text_3, l.text_4, l.text_5, l.text_6, l.text_7, l.text_8,
                s.num_1   AS s_num_1,
                s.num_2   AS s_num_2,
                s.num_3   AS s_num_3,
                s.num_4   AS s_num_4,
                s.num_5   AS s_num_5,
                s.bool_1  AS bool_1_small,
                s.text_2  AS s_text_2,
                s.text_3  AS s_text_3,
                s.text_4  AS s_text_4
            FROM large_t l
            INNER JOIN small_t s ON l.text_1 = s.text_1
            """
        )

    _, t, m = measure(_merge_duckdb)
    _row("Merge inner join  (on text_1)", t, m)

    # 7. Pivot  – DuckDB native PIVOT syntax
    def _pivot_duckdb():
        return con.execute(
            "PIVOT merged_t ON text_2 USING avg(num_1) GROUP BY bool_2"
        ).fetchdf()

    pivot_duck, t, m = measure(_pivot_duckdb)
    _row("Pivot  (bool_2 × text_2, mean num_1)", t, m)

    # 8. Save
    out = str(BASE_DIR / "output_duckdb.parquet").replace("\\", "/")

    def _save_duckdb():
        con.execute(
            f"COPY (PIVOT merged_t ON text_2 USING avg(num_1) GROUP BY bool_2) "
            f"TO '{out}' (FORMAT PARQUET)"
        )

    _, t, m = measure(_save_duckdb)
    _row("Save pivot → output_duckdb.parquet", t, m)

    con.close()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 76)
    print("  Performance benchmark: pandas vs polars vs DuckDB")
    print(f"  Large table: {LARGE_ROWS:,} rows × 20 cols")
    print(f"  Small table: {SMALL_ROWS:,} rows × 10 cols")
    print("=" * 76)

    # ── Clean up any previous parquet files ───────────────────────────────────
    deleted = list(BASE_DIR.glob("*.parquet"))
    for f in deleted:
        f.unlink()
    if deleted:
        print(f"\n  Deleted {len(deleted)} existing parquet file(s):")
        for f in deleted:
            print(f"    {f.name}")

    generate_data()

    bench_pandas()
    bench_polars()
    bench_duckdb()

    print("\n" + "─" * 76)
    print("  Done.  Output files:")
    for f in sorted(BASE_DIR.glob("output_*.parquet")):
        print(f"    {f.name}  ({os.path.getsize(f)/1_024:.2f} KiB)")
    print("─" * 76 + "\n")


if __name__ == "__main__":
    main()
