"""Bronze layer, imperative style: land every source as-is, plus lineage columns.

Each source uses the loading pattern that fits how it arrives:

    orders      JSON files    Auto Loader stream, schema evolution (coupon_code appears in batch 3)
    customers   CSV CDC       Auto Loader stream, every change event appended; MERGE happens in Silver
    payments    XML files     Auto Loader stream with rowTag
    clicks      Avro files    Auto Loader stream, the high-volume source
    products    Parquet       batch snapshot, idempotent overwrite of one snapshot_date (replaceWhere)
    rates       REST API      HTTP call (or the file fallback), idempotent MERGE by date

Bronze keeps data raw. Cleaning, typing, and deduplication belong to Silver.
"""

import json
import urllib.request
from typing import NamedTuple

from databricks.sdk.runtime import dbutils, spark
from pyspark.sql import functions as F

from understanding_databricks.common import (
    build_argument_parser,
    ensure_volumes,
    get_logger,
    list_batches,
    volume_path,
)

RATES_API_URL = "https://api.frankfurter.dev/v1/latest?base=EUR&symbols=USD,GBP,PLN,CHF,SEK"
API_TIMEOUT_SECONDS = 15


class FileSource(NamedTuple):
    file_format: str
    bronze_table: str
    reader_options: dict


FILE_SOURCES = {
    "orders": FileSource("json", "bronze_orders", {"cloudFiles.schemaEvolutionMode": "addNewColumns"}),
    "customers": FileSource("csv", "bronze_customers_cdc", {"header": "true"}),
    "payments": FileSource("xml", "bronze_payments", {"rowTag": "payment"}),
    "clicks": FileSource("avro", "bronze_clicks", {}),
}

logger = get_logger("bronze")


def latest_table_version(table: str) -> int:
    """Latest Delta version of a table, or -1 if the table doesn't exist yet."""
    if not spark.catalog.tableExists(table):
        return -1
    return spark.sql(f"DESCRIBE HISTORY {table} LIMIT 1").first()["version"]


def rows_written_since(table: str, version: int) -> int:
    """Rows written by every commit after `version`, read from the table's own Delta history."""
    output_rows = F.coalesce(F.col("operationMetrics")["numOutputRows"].cast("long"), F.lit(0))
    total = (spark.sql(f"DESCRIBE HISTORY {table}")
             .where(F.col("version") > version)
             .select(F.sum(output_rows))
             .first()[0])
    return int(total or 0)


def ingest_files(catalog: str, schema: str, source_name: str) -> None:
    """Incremental file ingestion with Auto Loader: only files not seen before are read."""
    source = FILE_SOURCES[source_name]
    landing = volume_path(catalog, schema, "landing")
    checkpoint = f"{volume_path(catalog, schema, 'checkpoints')}/{source.bronze_table}"
    target_table = f"{catalog}.{schema}.{source.bronze_table}"
    version_before = latest_table_version(target_table)

    new_records = (spark.readStream.format("cloudFiles")
                   .option("cloudFiles.format", source.file_format)
                   .option("cloudFiles.schemaLocation", f"{checkpoint}/_schema")
                   .options(**source.reader_options)
                   .load(f"{landing}/{source_name}/")
                   .withColumn("_source_file", F.col("_metadata.file_path"))
                   .withColumn("_ingested_at", F.current_timestamp()))

    query = (new_records.writeStream
             .option("checkpointLocation", checkpoint)
             .option("mergeSchema", "true")  # columns Auto Loader adds to the source schema reach the table too
             .trigger(availableNow=True)
             .toTable(target_table))
    query.awaitTermination()

    logger.info("%s -> %s: %d new rows", source_name, source.bronze_table,
                rows_written_since(target_table, version_before))


def ingest_products(catalog: str, schema: str) -> None:
    """Full snapshot source: load the latest snapshot and replace that day's rows, so reruns are safe."""
    landing = volume_path(catalog, schema, "landing")
    batches = list_batches(f"{landing}/products")
    if not batches:
        logger.info("products: no snapshots yet")
        return

    latest_batch = batches[-1]
    snapshot = (spark.read.parquet(f"{landing}/products/{latest_batch}")
                .withColumn("_source_file", F.col("_metadata.file_path"))
                .withColumn("_ingested_at", F.current_timestamp()))
    snapshot_date = snapshot.select(F.max("snapshot_date")).first()[0]
    target_table = f"{catalog}.{schema}.bronze_products"

    if spark.catalog.tableExists(target_table):
        writer = snapshot.write.mode("overwrite").option("replaceWhere", f"snapshot_date = '{snapshot_date}'")
    else:
        writer = snapshot.write.mode("append")
    writer.format("delta").saveAsTable(target_table)
    logger.info("products -> bronze_products: snapshot %s from %s, %d rows",
                snapshot_date, latest_batch, snapshot.count())


def fetch_rates_document(catalog: str, schema: str, mode: str) -> tuple[str, str]:
    """The raw JSON rates document and where it came from: the API, or the latest fallback file."""
    if mode == "api":
        with urllib.request.urlopen(RATES_API_URL, timeout=API_TIMEOUT_SECONDS) as response:
            return response.read().decode("utf-8"), "api"

    landing = volume_path(catalog, schema, "landing")
    batches = list_batches(f"{landing}/exchange_rates")
    if not batches:
        raise RuntimeError("No exchange-rate files in the landing volume; run the data generator first")
    latest_batch = batches[-1]
    return dbutils.fs.head(f"{landing}/exchange_rates/{latest_batch}/rates.json"), f"file:{latest_batch}"


def ingest_rates(catalog: str, schema: str, mode: str) -> None:
    """API source: one call per run, upserted by date, so running twice a day doesn't duplicate."""
    raw_document, origin = fetch_rates_document(catalog, schema, mode)
    rates_document = json.loads(raw_document)
    rates = {currency: float(rate) for currency, rate in rates_document["rates"].items()}
    target_table = f"{catalog}.{schema}.bronze_exchange_rates"

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {target_table} (
            rate_date DATE, base STRING, rates MAP<STRING, DOUBLE>,
            raw_payload STRING, source STRING, _ingested_at TIMESTAMP)
    """)
    incoming_rates = (spark.createDataFrame(
        [(rates_document["date"], rates_document["base"], rates, raw_document, origin)],
        "rate_date STRING, base STRING, rates MAP<STRING, DOUBLE>, raw_payload STRING, source STRING")
        .withColumn("rate_date", F.to_date("rate_date"))
        .withColumn("_ingested_at", F.current_timestamp()))
    incoming_rates.createOrReplaceTempView("incoming_rates")

    spark.sql(f"""
        MERGE INTO {target_table} AS existing
        USING incoming_rates AS incoming
        ON existing.rate_date = incoming.rate_date AND existing.base = incoming.base
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)
    logger.info("rates -> bronze_exchange_rates: %s for %s (source: %s)",
                rates_document["base"], rates_document["date"], origin)


def main():
    parser = build_argument_parser("Load one source into the Bronze layer")
    parser.add_argument("--source", required=True, choices=[*FILE_SOURCES, "products", "rates"])
    parser.add_argument("--rates-mode", default="api", choices=["api", "file"])
    args = parser.parse_args()

    ensure_volumes(args.catalog, args.schema)
    if args.source in FILE_SOURCES:
        ingest_files(args.catalog, args.schema, args.source)
    elif args.source == "products":
        ingest_products(args.catalog, args.schema)
    else:
        ingest_rates(args.catalog, args.schema, args.rates_mode)


if __name__ == "__main__":
    main()
