"""Synthetic data generator for the demo online store.

Each run writes one new batch for every source into the `landing` volume, in the format a real
upstream system would use:

    landing/orders/batch=NNNN/          JSON, nested order lines     (new files every run)
    landing/customers/batch=NNNN/       CSV, change data capture      (INSERT / UPDATE / DELETE)
    landing/products/batch=NNNN/        Parquet, full daily snapshot
    landing/payments/batch=NNNN/        XML, one <payment> per row
    landing/clicks/batch=NNNN/          Avro, high-volume clickstream
    landing/exchange_rates/batch=NNNN/  JSON shaped like the currency API response (fallback source)

Every random value is derived from a hash of an id and the batch number, so a batch is
reproducible. Some dirty data is injected on purpose (duplicates, orders without a customer, late
orders, a new column from batch 3) so the Silver layer has real work to do.
"""

import datetime
import json
import zlib

from databricks.sdk.runtime import dbutils, spark
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from understanding_databricks.common import (
    build_argument_parser,
    ensure_volumes,
    get_logger,
    list_batches,
    volume_path,
)

DEFAULT_ORDERS_PER_BATCH = 100_000
CLICKS_PER_ORDER = 5
PRODUCT_COUNT = 2_000
INITIAL_CUSTOMER_COUNT = 50_000
NEW_CUSTOMERS_PER_BATCH = 1_000

MIN_PRICE_EUR = 5.0
PRICE_RANGE_EUR = 495
MAX_DAILY_PRICE_CHANGE = 0.1  # a price moves by up to +/-5% between two snapshots
INACTIVE_PRODUCT_SHARE = 0.03

MOVED_CUSTOMER_SHARE = 0.02
DELETED_CUSTOMER_SHARE = 0.005
MAX_CUSTOMER_AGE_DAYS = 1_000

MAX_LINES_PER_ORDER = 4
MAX_QUANTITY_PER_LINE = 3
ORDER_WITHOUT_CUSTOMER_SHARE = 0.005
LATE_ORDER_SHARE = 0.01
MAX_LATE_DAYS = 5
RESENT_ORDER_SHARE = 0.005
COUPONS_LAUNCH_BATCH = 3
COUPON_SHARE = 0.1
COUPON_VARIANTS = 30

UNPAID_ORDER_SHARE = 0.03
FAILED_PAYMENT_SHARE = 0.05
MINUTES_FROM_ORDER_TO_PAYMENT = 5

ANONYMOUS_CLICK_SHARE = 0.3
EVENTS_PER_SESSION = 8

SECONDS_PER_HOUR = 3_600
SECONDS_PER_DAY = 86_400
HASH_BUCKETS = 1_000_000

# Approximate rates used to price orders in local currency: 1 EUR = rate units of the currency.
EUR_EXCHANGE_RATES = {"EUR": 1.0, "USD": 1.08, "GBP": 0.85, "PLN": 4.30, "CHF": 0.95, "SEK": 11.50}
CATEGORIES = ["Electronics", "Books", "Home", "Garden", "Toys", "Sports", "Beauty", "Grocery"]
BRANDS = ["Acme", "Northwind", "Contoso", "Fabrikam", "Globex", "Initech", "Umbrella", "Stark"]
FIRST_NAMES = ["Olena", "Taras", "Anna", "Max", "Sofia", "Lukas", "Emma", "Noah", "Mia", "Oliver"]
LAST_NAMES = ["Kovalenko", "Schmidt", "Nowak", "Smith", "Rossi", "Larsen", "Dubois", "Garcia"]
LOCATIONS = [
    ("DE", "Berlin"), ("PL", "Warsaw"), ("UA", "Kyiv"), ("GB", "London"),
    ("FR", "Paris"), ("SE", "Stockholm"), ("CH", "Zurich"), ("US", "New York"),
]
PAYMENT_METHODS = ["card", "paypal", "bank_transfer", "apple_pay"]
DEVICES = ["mobile", "desktop", "tablet"]

# Cumulative upper bounds: a random number below 0.05 means "cancelled", below 0.60 "paid", and so on.
ORDER_STATUS_BOUNDS = [("cancelled", 0.05), ("paid", 0.60), ("shipped", 0.90)]
DEFAULT_ORDER_STATUS = "placed"
CLICK_EVENT_BOUNDS = [("page_view", 0.70), ("product_view", 0.88), ("add_to_cart", 0.97)]
DEFAULT_CLICK_EVENT = "checkout"


def uniform_random(*parts) -> Column:
    """A repeatable number in [0, 1), computed from a hash of the given columns and literals."""
    hash_inputs = [part if isinstance(part, Column) else F.lit(part) for part in parts]
    return F.pmod(F.xxhash64(*hash_inputs), F.lit(HASH_BUCKETS)) / F.lit(float(HASH_BUCKETS))


def random_choice(values: list, random_number: Column) -> Column:
    """One of `values`, chosen by a random number in [0, 1)."""
    options = F.array(*[F.lit(value) for value in values])
    return F.element_at(options, (random_number * len(values)).cast("int") + 1)


def weighted_choice(random_number: Column, upper_bounds: list[tuple[str, float]], default: str) -> Column:
    """The first value whose cumulative upper bound is above `random_number`, otherwise `default`."""
    choice = None
    for value, upper_bound in upper_bounds:
        condition = random_number < upper_bound
        choice = F.when(condition, value) if choice is None else choice.when(condition, value)
    return choice.otherwise(default)


def random_location(random_number: Column) -> Column:
    """A (country, city) struct, chosen by a random number in [0, 1)."""
    locations = F.array(*[F.struct(F.lit(country).alias("country"), F.lit(city).alias("city"))
                          for country, city in LOCATIONS])
    return F.element_at(locations, (random_number * len(LOCATIONS)).cast("int") + 1)


def random_product_id(*parts) -> Column:
    return (F.pmod(F.xxhash64(*parts), F.lit(PRODUCT_COUNT)) + 1).cast("bigint")


def catalogue_price_eur(product_id: Column) -> Column:
    """The list price of a product, shared by the products snapshot and the order lines."""
    return F.round(F.lit(MIN_PRICE_EUR) + uniform_random(product_id.cast("bigint"), "price") * PRICE_RANGE_EUR, 2)


def seconds_before(moment: datetime.datetime, seconds: Column) -> Column:
    return F.timestamp_seconds(F.lit(int(moment.timestamp())) - seconds)


def customer_count_before(batch: int) -> int:
    return INITIAL_CUSTOMER_COUNT + (batch - 1) * NEW_CUSTOMERS_PER_BATCH


def with_customer_profile(customer_ids: DataFrame, location_salt: str, as_of: datetime.date) -> DataFrame:
    customer_id = F.col("id")
    first_name = random_choice(FIRST_NAMES, uniform_random(customer_id, "first"))
    last_name = random_choice(LAST_NAMES, uniform_random(customer_id, "last"))
    location = random_location(uniform_random(customer_id, location_salt))
    email = F.concat(F.lower(first_name), F.lit("."), F.lower(last_name), customer_id.cast("string"),
                     F.lit("@example.com"))
    days_since_signup = (uniform_random(customer_id, "signup") * MAX_CUSTOMER_AGE_DAYS).cast("int")
    return (customer_ids
            .withColumn("first_name", first_name)
            .withColumn("last_name", last_name)
            .withColumn("email", email)
            .withColumn("country", location["country"])
            .withColumn("city", location["city"])
            .withColumn("signup_date", F.date_sub(F.lit(as_of), days_since_signup)))


def customer_changes(batch: int, batch_time: datetime.datetime) -> DataFrame:
    """CRM change feed: everyone is new in batch 1; later batches add, move, and delete customers."""
    today = batch_time.date()
    if batch == 1:
        first_customers = spark.range(1, INITIAL_CUSTOMER_COUNT + 1)
        changes = with_customer_profile(first_customers, "loc", today).withColumn("operation", F.lit("INSERT"))
    else:
        last_customer_id = customer_count_before(batch - 1)
        new_ids = spark.range(last_customer_id + 1, last_customer_id + NEW_CUSTOMERS_PER_BATCH + 1)
        existing = (spark.range(1, last_customer_id + 1)
                    .withColumn("change_roll", uniform_random(F.col("id"), f"change-{batch}")))
        change_roll = F.col("change_roll")
        moved_ids = existing.where(change_roll < MOVED_CUSTOMER_SHARE).drop("change_roll")
        deleted_ids = (existing
                       .where((change_roll >= MOVED_CUSTOMER_SHARE)
                              & (change_roll < MOVED_CUSTOMER_SHARE + DELETED_CUSTOMER_SHARE))
                       .drop("change_roll"))
        # A moved customer gets a new random location, so the salt includes the batch number.
        changes = (with_customer_profile(new_ids, "loc", today).withColumn("operation", F.lit("INSERT"))
                   .unionByName(with_customer_profile(moved_ids, f"loc-{batch}", today)
                                .withColumn("operation", F.lit("UPDATE")))
                   .unionByName(deleted_ids.withColumn("operation", F.lit("DELETE")), allowMissingColumns=True))

    seconds_earlier = (uniform_random(F.col("id"), f"opts-{batch}") * SECONDS_PER_HOUR).cast("long")
    return (changes
            .withColumn("op_ts", seconds_before(batch_time, seconds_earlier))
            .withColumnRenamed("id", "customer_id"))


def products_snapshot(batch: int, snapshot_date: datetime.date) -> DataFrame:
    product_id = F.col("id")
    price_change = F.lit(1.0) + (uniform_random(product_id, f"drift-{batch}") - 0.5) * MAX_DAILY_PRICE_CHANGE
    name = F.concat_ws(" ", F.col("brand"), F.col("category"), F.lit("item"), product_id.cast("string"))
    return (spark.range(1, PRODUCT_COUNT + 1)
            .withColumn("category", random_choice(CATEGORIES, uniform_random(product_id, "cat")))
            .withColumn("brand", random_choice(BRANDS, uniform_random(product_id, "brand")))
            .withColumn("name", name)
            .withColumn("price_eur", F.round(catalogue_price_eur(product_id) * price_change, 2))
            .withColumn("is_active", uniform_random(product_id, f"active-{batch}") > INACTIVE_PRODUCT_SHARE)
            .withColumn("snapshot_date", F.lit(snapshot_date))
            .withColumnRenamed("id", "product_id"))


def orders(batch: int, order_count: int, batch_time: datetime.datetime) -> DataFrame:
    order_id = F.col("order_id")
    customer_count = customer_count_before(batch)
    exchange_rate = F.element_at(
        F.create_map(*[F.lit(item) for pair in EUR_EXCHANGE_RATES.items() for item in pair]),
        F.col("currency"),
    )

    def order_line(line_number: Column) -> Column:
        product_id = random_product_id(order_id, line_number, F.lit("p"))
        quantity = (F.pmod(F.xxhash64(order_id, line_number, F.lit("q")), F.lit(MAX_QUANTITY_PER_LINE)) + 1).cast("int")
        unit_price = F.round(catalogue_price_eur(product_id) * exchange_rate, 2)
        return F.struct(product_id.alias("product_id"), quantity.alias("quantity"), unit_price.alias("unit_price"))

    is_late = uniform_random(order_id, "late") < LATE_ORDER_SHARE
    late_seconds = (uniform_random(order_id, "late-days") * MAX_LATE_DAYS + 1).cast("long") * SECONDS_PER_DAY
    recent_seconds = (uniform_random(order_id, "ts") * SECONDS_PER_HOUR).cast("long")
    customer_id = (F.when(uniform_random(order_id, "nocust") < ORDER_WITHOUT_CUSTOMER_SHARE, F.lit(None).cast("bigint"))
                   .otherwise((uniform_random(order_id, "cust") * customer_count).cast("bigint") + 1))
    line_count = (uniform_random(order_id, "n") * MAX_LINES_PER_ORDER).cast("int") + 1
    order_total = F.aggregate("items", F.lit(0.0), lambda total, line: total + line["quantity"] * line["unit_price"])

    order_rows = (spark.range(order_count)
                  .withColumn("order_id", F.format_string("ORD-%04d-%08d", F.lit(batch), F.col("id")))
                  .withColumn("customer_id", customer_id)
                  .withColumn("currency", random_choice(list(EUR_EXCHANGE_RATES), uniform_random(order_id, "cur")))
                  .withColumn("order_ts", seconds_before(batch_time, F.when(is_late, late_seconds).otherwise(recent_seconds)))
                  .withColumn("status", weighted_choice(uniform_random(order_id, "status"), ORDER_STATUS_BOUNDS,
                                                        DEFAULT_ORDER_STATUS))
                  .withColumn("items", F.transform(F.sequence(F.lit(1), line_count), order_line))
                  .withColumn("total_amount", F.round(order_total, 2))
                  .withColumn("shipping", random_location(uniform_random(order_id, "ship")))
                  .drop("id"))

    if batch >= COUPONS_LAUNCH_BATCH:  # the shop launched coupons; the upstream schema changes
        coupon = F.concat(F.lit("SAVE"), (uniform_random(order_id, "cv") * COUPON_VARIANTS).cast("int").cast("string"))
        order_rows = order_rows.withColumn(
            "coupon_code", F.when(uniform_random(order_id, "coupon") < COUPON_SHARE, coupon))

    resent_orders = order_rows.where(uniform_random(order_id, "dup") < RESENT_ORDER_SHARE)
    return order_rows.unionByName(resent_orders)


def payments(order_rows: DataFrame) -> DataFrame:
    order_id = F.col("order_id")
    paid_orders = (order_rows
                   .dropDuplicates(["order_id"])
                   .where(F.col("status") != "cancelled")
                   .where(uniform_random(order_id, "paysample") < 1 - UNPAID_ORDER_SHARE))
    status = F.when(uniform_random(order_id, "fail") < FAILED_PAYMENT_SHARE, "failed").otherwise("succeeded")
    return paid_orders.select(
        F.concat(F.lit("PAY-"), order_id).alias("payment_id"),
        order_id,
        F.col("total_amount").alias("amount"),
        "currency",
        random_choice(PAYMENT_METHODS, uniform_random(order_id, "method")).alias("method"),
        status.alias("status"),
        (F.col("order_ts") + F.expr(f"INTERVAL {MINUTES_FROM_ORDER_TO_PAYMENT} MINUTES")).alias("paid_at"),
    )


def clicks(batch: int, event_count: int, batch_time: datetime.datetime) -> DataFrame:
    event_number = F.col("id")
    customer_count = customer_count_before(batch)
    customer_id = (F.when(uniform_random(event_number, f"anon-{batch}") < ANONYMOUS_CLICK_SHARE,
                          F.lit(None).cast("bigint"))
                   .otherwise((uniform_random(event_number, f"c-{batch}") * customer_count).cast("bigint") + 1))
    event_type = weighted_choice(uniform_random(event_number, f"type-{batch}"), CLICK_EVENT_BOUNDS, DEFAULT_CLICK_EVENT)
    seconds_earlier = (uniform_random(event_number, f"ts-{batch}") * SECONDS_PER_HOUR).cast("long")
    return (spark.range(event_count)
            .withColumn("event_id", F.format_string("EV-%04d-%09d", F.lit(batch), event_number))
            .withColumn("session_id", F.format_string("S-%04d-%08d", F.lit(batch),
                                                      (event_number / EVENTS_PER_SESSION).cast("long")))
            .withColumn("customer_id", customer_id)
            .withColumn("event_type", event_type)
            .withColumn("product_id", random_product_id(event_number, F.lit(f"p-{batch}")))
            .withColumn("device", random_choice(DEVICES, uniform_random(event_number, f"dev-{batch}")))
            .withColumn("event_ts", seconds_before(batch_time, seconds_earlier))
            .drop("id"))


def exchange_rates_document(batch: int, rate_date: datetime.date) -> str:
    """A JSON document shaped like the Frankfurter API response, used when the API isn't reachable."""
    def daily_rate(currency: str, base_rate: float) -> float:
        change = ((zlib.crc32(f"{currency}-{batch}".encode()) % 100) - 50) / 5000  # within +/-1%
        return round(base_rate * (1 + change), 4)

    rates = {currency: daily_rate(currency, rate) for currency, rate in EUR_EXCHANGE_RATES.items() if currency != "EUR"}
    return json.dumps({"amount": 1.0, "base": "EUR", "date": rate_date.isoformat(), "rates": rates})


def landing_datasets(batch: int, order_count: int, batch_time: datetime.datetime) -> dict:
    """source -> (DataFrame with its output file count applied, file format, writer options)"""
    order_rows = orders(batch, order_count, batch_time)
    return {
        "orders": (order_rows.repartition(20), "json", {}),
        "customers": (customer_changes(batch, batch_time).coalesce(1), "csv", {"header": "true"}),
        "products": (products_snapshot(batch, batch_time.date()).coalesce(1), "parquet", {}),
        "payments": (payments(order_rows).coalesce(4), "xml", {"rootTag": "payments", "rowTag": "payment"}),
        "clicks": (clicks(batch, order_count * CLICKS_PER_ORDER, batch_time).repartition(10), "avro", {}),
    }


def write_landing_files(landing: str, folder: str, datasets: dict, logger) -> None:
    for source, (dataset, file_format, options) in datasets.items():
        path = f"{landing}/{source}/{folder}"
        # A failed earlier attempt may have left files here. mode("overwrite") only marks old files
        # as stale, but Auto Loader lists every file in the folder and would ingest them as well.
        dbutils.fs.rm(path, recurse=True)
        dataset.write.mode("overwrite").format(file_format).options(**options).save(path)
        logger.info("Wrote %s/%s", source, folder)


def main():
    parser = build_argument_parser("Write one batch of synthetic online-store data into the landing volume")
    parser.add_argument("--orders", type=int, default=DEFAULT_ORDERS_PER_BATCH,
                        help=f"orders per batch; clicks are {CLICKS_PER_ORDER}x this")
    args = parser.parse_args()
    logger = get_logger("generator")

    ensure_volumes(args.catalog, args.schema)
    landing = volume_path(args.catalog, args.schema, "landing")
    # A batch counts only after its completion marker is written, so a failed run is simply redone
    # with the same batch number on the next attempt.
    completion_markers = f"{landing}/_generator"
    batch = len(list_batches(completion_markers)) + 1
    folder = f"batch={batch:04d}"
    batch_time = datetime.datetime.now(datetime.UTC).replace(microsecond=0)
    logger.info("Generating %s with %d orders", folder, args.orders)

    # No .cache(): serverless compute doesn't support it, and the data is deterministic anyway.
    write_landing_files(landing, folder, landing_datasets(batch, args.orders, batch_time), logger)
    dbutils.fs.put(f"{landing}/exchange_rates/{folder}/rates.json",
                   exchange_rates_document(batch, batch_time.date()), overwrite=True)
    logger.info("Wrote exchange_rates/%s", folder)

    dbutils.fs.put(f"{completion_markers}/{folder}", batch_time.isoformat(), overwrite=True)
    logger.info("Batch %s done", folder)


if __name__ == "__main__":
    main()
