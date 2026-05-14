"""
spark_session.py — Centralized SparkSession factory.

Import get_spark() in any PySpark job instead of building
a session inline. This ensures consistent Delta Lake config
across ingestion, processing, and training jobs.
"""

from __future__ import annotations

from delta import configure_spark_with_delta_pip
from loguru import logger
from pyspark.sql import SparkSession

from src.config import settings


def get_spark(app_name: str = "MisinformationLakehouse") -> SparkSession:
    """
    Return a SparkSession configured for Delta Lake.

    Automatically switches between local filesystem and S3
    based on STORAGE_MODE environment variable.
    """
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # Performance tuning for local dev
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")  # low for local dev
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    )

    # Wire Delta JARs (and S3 JARs when needed) from Maven into Spark's classpath
    if settings.storage_mode == "s3":
        # hadoop-aws must match the Hadoop version bundled in PySpark 4.1.x (3.4.2)
        s3_packages = [
            "org.apache.hadoop:hadoop-aws:3.4.2",
            "com.amazonaws:aws-java-sdk-bundle:1.12.367",
        ]
        builder = configure_spark_with_delta_pip(builder, extra_packages=s3_packages)
        logger.info("Configuring Spark for S3 storage")
        builder = (
            builder.config(
                "spark.hadoop.fs.s3a.impl",
                "org.apache.hadoop.fs.s3a.S3AFileSystem",
            )
            .config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
            )
            .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
            .config("spark.hadoop.fs.s3a.path.style.access", "false")
            .config("spark.hadoop.fs.s3a.access.key", settings.aws_access_key_id)
            .config("spark.hadoop.fs.s3a.secret.key", settings.aws_secret_access_key)
        )
    else:
        builder = configure_spark_with_delta_pip(builder)
        logger.info("Configuring Spark for local storage")

    spark = builder.getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"SparkSession initialized | app={app_name} | mode={settings.storage_mode}")
    return spark
