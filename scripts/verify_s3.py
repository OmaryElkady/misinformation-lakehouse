"""
verify_s3.py — Verify AWS S3 connectivity before switching to production storage.

Checks:
  1. Bucket exists and is accessible
  2. Can write, read, and delete a test object

Exit code 0 on success, 1 on failure.
"""

import sys
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import settings

console = Console()

TEST_KEY = "health-check/test.txt"
TEST_CONTENT = b"misinformation-lakehouse s3 connectivity check"


def _check_bucket(s3: Any) -> tuple[bool, str]:
    try:
        s3.head_bucket(Bucket=settings.s3_bucket_name)
        return True, f"Bucket '{settings.s3_bucket_name}' exists and is accessible"
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "404":
            return False, f"Bucket '{settings.s3_bucket_name}' not found"
        if code == "403":
            return False, f"Access denied to bucket '{settings.s3_bucket_name}'"
        return False, f"ClientError {code}: {e}"
    except BotoCoreError as e:
        return False, f"BotoCoreError: {e}"


def _write_object(s3: Any) -> tuple[bool, str]:
    try:
        s3.put_object(Bucket=settings.s3_bucket_name, Key=TEST_KEY, Body=TEST_CONTENT)
        return True, f"Wrote s3://{settings.s3_bucket_name}/{TEST_KEY}"
    except (ClientError, BotoCoreError) as e:
        return False, f"Write failed: {e}"


def _read_object(s3: Any) -> tuple[bool, str]:
    try:
        response = s3.get_object(Bucket=settings.s3_bucket_name, Key=TEST_KEY)
        body = response["Body"].read()
        if body != TEST_CONTENT:
            return False, f"Content mismatch: expected {TEST_CONTENT!r}, got {body!r}"
        return True, "Read-back content verified"
    except (ClientError, BotoCoreError) as e:
        return False, f"Read failed: {e}"


def _delete_object(s3: Any) -> tuple[bool, str]:
    try:
        s3.delete_object(Bucket=settings.s3_bucket_name, Key=TEST_KEY)
        return True, f"Deleted s3://{settings.s3_bucket_name}/{TEST_KEY}"
    except (ClientError, BotoCoreError) as e:
        return False, f"Delete failed: {e}"


def main() -> int:
    console.print(Panel("[bold cyan]S3 Connectivity Verification[/bold cyan]", expand=False))
    console.print(f"  Bucket : [yellow]{settings.s3_bucket_name}[/yellow]")
    console.print(f"  Region : [yellow]{settings.aws_default_region}[/yellow]")
    console.print(f"  Mode   : [yellow]{settings.storage_mode}[/yellow]\n")

    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
            region_name=settings.aws_default_region,
        )
    except Exception as e:
        console.print(f"[bold red]Failed to create S3 client:[/bold red] {e}")
        return 1

    checks = [
        ("Bucket access", lambda: _check_bucket(s3)),
        ("Write object", lambda: _write_object(s3)),
        ("Read-back verify", lambda: _read_object(s3)),
        ("Delete object", lambda: _delete_object(s3)),
    ]

    table = Table(title="Check Results", show_header=True, header_style="bold magenta")
    table.add_column("Check", style="dim", min_width=18)
    table.add_column("Status", justify="center", min_width=6)
    table.add_column("Message")

    all_passed = True
    for name, fn in checks:
        passed, msg = fn()
        status = "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]"
        table.add_row(name, status, msg)
        if not passed:
            all_passed = False

    console.print(table)

    if all_passed:
        console.print(
            Panel("[bold green]All checks passed — S3 is ready for production.[/bold green]", expand=False)
        )
        return 0

    console.print(
        Panel("[bold red]One or more checks failed — fix before running ingestion.[/bold red]", expand=False)
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
