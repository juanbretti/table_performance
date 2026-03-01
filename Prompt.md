Quiero hacer un test de tiempo de ejecución y uso de memoria entre pandas, duckdb y polars.
Para ello hacer dos tablas:
* Crear un archivo parquet con 5_000_000 de filas y 20 columnas. Con 10 columnas numéricas, 2 columnas booleans, 8 columnas de texto de hasta 20 caracteres incluidos espacio.
* Guardar el fichero parquet en la ruta del proyecto

La segunda tabla:
* Crear una segunda tabla con 10 columnnas y 10_000 filas. 5 columnas numéricas, 1 columnas booleans, 4 columnas de texto de hasta 20 caracteres incluidos espacio.
* Guardar el fichero parquet en la ruta del proyecto

Luego, hacer pruebas de velocidad usando las tres librerías. Medir tiempo de ejecución y consumo de memoria RAM.
Hacer las siguientes operaciones:
* Borra todos los ficheros .parquet que hayan en la carpeta del proyecto
* Lectura del fichero parquet
* Aplicar un filtro sobre las columnas numéricas
* Aplicar un filtro sobre las columnas de texto
* Aplicar un filtro sobre las columnas booleans
* Aplicar una transformación de texto, como por ejemplo upper case y split por el espacio, tomando el texto hasta el primer espacio
* Hacer un merge entre las dos tablas
* Hacer un pivot de la tabla merged
* Guardar la nueva tabla en formato parquet

Varios temas:
* No creo que estés capturando todo el consumo de memoria. En particular para el caso DuckDB Arrow. Revisa por favor.
* Agrega un test, de ejecutar una user define function (UDF), que sea la suma de dos columnas numéricas y multiplicado por 3, si es que el producto es superior a 10
