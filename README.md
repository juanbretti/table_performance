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
    Read large parquet                                      0.491 s     1476.0 MiB
    Filter numeric  (num_1 > 500)                           0.817 s      614.6 MiB
    Filter text  (text_3 starts with 'a')                   0.422 s       17.3 MiB
    Filter boolean  (bool_1 == True)                        0.800 s      433.6 MiB
    Text transform  (upper + first word)                    4.514 s     1674.8 MiB
    Merge inner join  (on text_1)                           1.546 s      294.6 MiB
    Pivot  (bool_2 × text_2, mean num_1)                    0.379 s      218.2 MiB
    Save pivot → output_pandas.parquet                      0.004 s        0.0 MiB
    UDF  (num_1+num_2)*3 if >10                             1.276 s      567.0 MiB

────────────────────────────────────────────────────────────────────────────
  POLARS
────────────────────────────────────────────────────────────────────────────
    Operation                                                 Time    Peak RAM
    ──────────────────────────────────────────────────── ─────────   ─────────
    Read large parquet                                      0.173 s     1082.2 MiB
    Filter numeric  (num_1 > 500)                           0.069 s      210.9 MiB
    Filter text  (text_3 starts with 'a')                   0.035 s        4.4 MiB
    Filter boolean  (bool_1 == True)                        0.066 s      406.4 MiB
    Text transform  (upper + first word)                    0.186 s      319.2 MiB
    Merge inner join  (on text_1)                           0.301 s     1452.7 MiB
    Pivot  (bool_2 × text_2, mean num_1)                    0.171 s      134.4 MiB
    Save pivot → output_polars.parquet                      0.002 s        1.5 MiB
    UDF  (num_1+num_2)*3 if >10                             2.798 s      198.5 MiB

────────────────────────────────────────────────────────────────────────────
  DUCKDB (fetchdf)
────────────────────────────────────────────────────────────────────────────
    Operation                                                 Time    Peak RAM
    ──────────────────────────────────────────────────── ─────────   ─────────
    Read large parquet (→ in-memory table)                  0.952 s     1122.8 MiB
    Filter numeric  (num_1 > 500)                           4.423 s     1668.4 MiB
    Filter text  (text_3 starts with 'a')                   0.914 s      146.0 MiB
    Filter boolean  (bool_1 == True)                        4.054 s     1504.8 MiB
    Text transform  (upper + first word)                    7.823 s     4089.4 MiB
    Merge inner join  (on text_1)                           0.718 s     1511.4 MiB
    Pivot  (bool_2 × text_2, mean num_1)                    0.085 s        5.9 MiB
    Save pivot → output_duckdb.parquet                      0.079 s        4.2 MiB
    UDF  (num_1+num_2)*3 if >10                             8.130 s     3977.8 MiB

────────────────────────────────────────────────────────────────────────────
  DUCKDB (NATIVE & ARROW)
────────────────────────────────────────────────────────────────────────────
    Operation                                                 Time    Peak RAM
    ──────────────────────────────────────────────────── ─────────   ─────────
    Read large parquet (→ virtual view)                     0.003 s        0.4 MiB
    Filter numeric  (num_1 > 500)                           0.032 s       61.5 MiB
    Filter text  (text_3 starts with 'a')                   0.050 s        7.8 MiB
    Filter boolean  (bool_1 == True)                        0.029 s        0.0 MiB
    Text transform  (upper + first word)                    0.032 s        0.4 MiB
    Merge inner join  (on text_1)                           0.009 s        0.0 MiB
    Pivot  (bool_2 × text_2, mean num_1)                    0.248 s       18.8 MiB
    Save pivot → output_duckdb_arrow.parquet                0.250 s       14.2 MiB
    UDF  (num_1+num_2)*3 if >10                             0.032 s       57.0 MiB

────────────────────────────────────────────────────────────────────────────
  Done.  Output files:
    output_duckdb.parquet  (0.80 KiB)
    output_duckdb_arrow.parquet  (0.80 KiB)
    output_pandas.parquet  (3.57 KiB)
    output_polars.parquet  (1.71 KiB)
────────────────────────────────────────────────────────────────────────────


  Chart saved → benchmark_chart.png
```

## Conclusion

![](benchmark_chart.png)

En orden, recomendaría por performance:
1. DuckDB exportando a tablas Arrow
2. Polars
3. Pandas
4. DuckDB exportando a DataFrames

En orden, recomendaría por simplicidad de uso:
1. Pandas
2. Polars
3. DuckDB exportando a tablas Arrow
4. DuckDB exportando a DataFrames

Nota:
* DuckDB -en el caso *Arrow*- acumula el trabajo en los pasos baratos y lo ejecuta todo junto en -por ejemplo-, pivot/save.
* Sobre el consumo de RAM en DuckDB, no estoy seguro que sea completo.