"""
Database Manager Module

This module provides utilities for loading and managing data in SQLite databases,
including Excel file import and data enrichment functions.
"""

import pandas as pd
import sqlite3
from tabulate import tabulate


def load_excel_to_sqlite(
    xlsx_path: str, db_path: str = ":memory:"
) -> sqlite3.Connection:
    """
    Load data from Excel file into SQLite database.

    Args:
        xlsx_path: Path to Excel file
        db_path: SQLite database path (default is in-memory)

    Returns:
        SQLite database connection
    """
    # Load Excel file
    sheet_names = pd.ExcelFile(xlsx_path).sheet_names
    print(f"Found sheets: {sheet_names}")

    # --- Step 1: Load and Enrich Accounts Sheet ---
    accounts_df = pd.read_excel(xlsx_path, sheet_name="Accounts")
    accounts_df = accounts_df.rename(
        columns={
            "Customer Number": "customer_number",
            "Customer Name": "customer_name",
            "Account Number": "account_number",
            "Account Name": "account_name",
            "Account Open Date": "open_date",
            "Account Closed Date": "closed_date",
            "Account Status": "account_status",
            "Dormant": "is_dormant",
            "DACA": "is_daca",
            "DACA Expiry": "daca_expiry",
            "Payroll": "is_payroll",
            "Billing Direct Debit Account": "is_direct_debit",
            "Available Balance": "available_balance",
            "Ledger Balance": "ledger_balance",
            "Deposit Rate": "deposit_rate",
            "Overdraft Rate": "overdraft_rate",
            "Date": "snapshot_date",
        }
    )

    # Clean & enrich
    accounts_df["open_date"] = pd.to_datetime(accounts_df["open_date"], errors="coerce")
    accounts_df["closed_date"] = pd.to_datetime(
        accounts_df["closed_date"], errors="coerce"
    )
    accounts_df["snapshot_date"] = pd.to_datetime(
        accounts_df["snapshot_date"], errors="coerce"
    )
    accounts_df["daca_expiry"] = pd.to_datetime(
        accounts_df["daca_expiry"], origin="1899-12-30", unit="D", errors="coerce"
    )

    for col in ["is_dormant", "is_daca", "is_payroll", "is_direct_debit"]:
        accounts_df[col] = (
            accounts_df[col].astype(str).str.lower().map({"yes": True, "no": False})
        )

    # Calculate account age in days
    today = pd.Timestamp.today()
    accounts_df["account_age_days"] = accounts_df["open_date"].apply(
        lambda x: (today - x).days if pd.notnull(x) else None
    )
    accounts_df["is_active"] = accounts_df["closed_date"].isna()

    # --- Step 2: Load and Enrich Daily Metrics from Account Sheets ---
    daily_metrics_list = []

    for sheet in sheet_names:
        if sheet == "Accounts":
            continue

        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        df = df[
            [
                "Date",
                "Available Balance",
                "Ledger Balance",
                "Deposit Rate",
                "Overdraft Rate",
                "Accrual",
            ]
        ].rename(
            columns={
                "Date": "date",
                "Available Balance": "available_balance",
                "Ledger Balance": "ledger_balance",
                "Deposit Rate": "deposit_rate",
                "Overdraft Rate": "overdraft_rate",
                "Accrual": "accrual",
            }
        )
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in [
            "available_balance",
            "ledger_balance",
            "deposit_rate",
            "overdraft_rate",
            "accrual",
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Add account reference and calculations
        df["account_number"] = sheet
        df["estimated_revenue"] = (
            df["available_balance"] * df["deposit_rate"] / 365
        ).fillna(0)
        df["net_change"] = df["available_balance"].diff()
        daily_metrics_list.append(df)

    daily_metrics_df = pd.concat(daily_metrics_list, ignore_index=True)

    # --- Step 3: Write to SQLite ---
    conn = sqlite3.connect(db_path)
    accounts_df.to_sql("accounts", conn, index=False, if_exists="replace")
    daily_metrics_df.to_sql("daily_metrics", conn, index=False, if_exists="replace")

    print("Data loaded into SQLite successfully.")
    # --- Output tables using tabulate ---
    print("\n📄 ACCOUNTS TABLE SAMPLE")
    print(
        tabulate(accounts_df.head(5).to_dict("list"), headers="keys", tablefmt="grid")
    )

    print("\n📈 DAILY METRICS TABLE SAMPLE")
    print(
        tabulate(
            daily_metrics_df.head(5).to_dict("list"), headers="keys", tablefmt="pipe"
        )
    )

    return conn  # return connection for querying
