# Test de performance entre Pandas, Polars y DuckDB

## Motivación

Este test, compara las tres librerías analizando su tiempo de ejecución y consumo máximo de memoria RAM.

## Entorno

Librerías de Python

```toml {title="pyproject.toml"}
[project]
name = "performance"
version = "0.1.0"
readme = "README.md"
requires-python = ">=3.14"
dependencies = [
    "duckdb>=1.4.4",
    "pandas>=3.0.1",
    "polars>=1.38.1",
    "pyarrow>=19.0.0",
    "psutil>=6.0.0",
]
```

Ordenador

```
System          Windows 11 64-bit
Processor	    Intel(R) Core(TM) i9-10885H CPU @ 2.40GHz (2.40 GHz)
Installed RAM	32.0 GB
```

## Resultados

```
(performance) PS C:\performance> uv run .\main.py
============================================================================
  Performance benchmark: pandas vs polars vs DuckDB
  Large table: 5,000,000 rows × 20 cols
  Small table: 10,000 rows × 10 cols
============================================================================

  Deleted 5 existing parquet file(s):
    large_table.parquet
    output_duckdb.parquet
    output_pandas.parquet
    output_polars.parquet
    small_table.parquet

► Generating datasets …

  Building small_table.parquet  (10 000 rows × 10 cols) …
    Saved → small_table.parquet  (0.5 MiB)
  Building large_table.parquet  (5 000 000 rows × 20 cols) …
    Saved → large_table.parquet  (262.6 MiB)


────────────────────────────────────────────────────────────────────────────
  PANDAS
────────────────────────────────────────────────────────────────────────────
    Operation                                                 Time    Peak RAM
    ──────────────────────────────────────────────────── ─────────   ─────────
    Read large parquet                                      0.400 s     1458.6 MiB
    Filter numeric  (num_1 > 500)                           0.761 s      575.7 MiB
    Filter text  (text_3 starts with 'a')                   0.421 s      117.4 MiB
    Filter boolean  (bool_1 == True)                        0.772 s      426.5 MiB
    Text transform  (upper + first word)                    5.067 s     1673.8 MiB
    Merge inner join  (on text_1)                           1.610 s      300.8 MiB
    Pivot  (bool_2 × text_2, mean num_1)                    0.357 s      216.0 MiB
    Save pivot → output_pandas.parquet                      0.005 s        0.0 MiB

────────────────────────────────────────────────────────────────────────────
  POLARS
────────────────────────────────────────────────────────────────────────────
    Operation                                                 Time    Peak RAM
    ──────────────────────────────────────────────────── ─────────   ─────────
    Read large parquet                                      0.165 s     1019.6 MiB
    Filter numeric  (num_1 > 500)                           0.092 s      190.0 MiB
    Filter text  (text_3 starts with 'a')                   0.035 s       28.0 MiB
    Filter boolean  (bool_1 == True)                        0.086 s      368.4 MiB
    Text transform  (upper + first word)                    0.204 s      319.4 MiB
    Merge inner join  (on text_1)                           0.385 s     1498.4 MiB
    Pivot  (bool_2 × text_2, mean num_1)                    0.211 s      150.2 MiB
    Save pivot → output_polars.parquet                      0.002 s        0.0 MiB

────────────────────────────────────────────────────────────────────────────
  DUCKDB
────────────────────────────────────────────────────────────────────────────
    Operation                                                 Time    Peak RAM
    ──────────────────────────────────────────────────── ─────────   ─────────
    Read large parquet (→ in-memory table)                  1.084 s     1119.0 MiB
    Filter numeric  (num_1 > 500)                           4.687 s     1669.8 MiB
    Filter text  (text_3 starts with 'a')                   1.039 s      170.6 MiB
    Filter boolean  (bool_1 == True)                        4.267 s     1679.8 MiB
    Text transform  (upper + first word)                    8.585 s     4123.1 MiB
    Merge inner join  (on text_1)                           0.849 s     1510.0 MiB
    Pivot  (bool_2 × text_2, mean num_1)                    0.084 s        5.4 MiB
    Save pivot → output_duckdb.parquet                      0.085 s        2.9 MiB

────────────────────────────────────────────────────────────────────────────
  Done.  Output files:
    output_duckdb.parquet  (0.80 KiB)
    output_pandas.parquet  (3.57 KiB)
    output_polars.parquet  (1.71 KiB)
────────────────────────────────────────────────────────────────────────────
```

## Conclusion

En orden, recomendaría por performance:
1. Polars
2. Pandas
3. DuckDB

En orden, recomendaría por simplicidad de uso:
1. Pandas
2. Polars
3. DuckDB

Es decir, creo que **Pandas 3.0** sigue siendo la librería más fácil de recomendar.
