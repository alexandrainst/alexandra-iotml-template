"""Data dump script.

Populate a postgres/TimescaleDB table with the contents of a CSV file, using
iotml-core's :class:`~iotml_core.utils.sql.SQLSession`. If the CSV carries a
numeric ``Time`` column (hours since the start of the run), it is expanded into
the canonical ``time_since_start`` / ``timestamp`` columns before being written.

Example:
    uv run python src/scripts/dump_csv_to_psql.py -i data/raw/run.csv -t observations
"""

import argparse
import logging

import pandas as pd
from iotml_core.utils.data_tools import compute_time_columns
from iotml_core.utils.sql import SQLSession

logger = logging.getLogger("dump_csv_to_psql")
logging.basicConfig(level=logging.INFO)


def dump_csv(input_file: str, table: str, if_exists: str = "append") -> None:
    """Read a CSV file and dump its rows into a postgres ``table``."""
    df = pd.read_csv(input_file)

    if "Time" in df.columns:
        # Canonical iotml time columns — see iotml_core.utils.data_tools.
        time_since_start, timestamps = compute_time_columns(df["Time"].tolist())
        df["time_since_start"] = time_since_start
        df["timestamp"] = timestamps
        df = df.drop(columns="Time")

    # SQLSession.dump_data_to_sql expects a {column: [values, ...]} mapping.
    data = df.to_dict(orient="list")

    session = SQLSession()
    session.dump_data_to_sql(table=table, data=data, if_exists=if_exists)
    session.close()
    logger.info(f"Dumped {len(df)} rows from {input_file} into table '{table}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dump CSV data into postgres")
    parser.add_argument(
        "-i", "--input-file", required=True, help="Path to the CSV file"
    )
    parser.add_argument("-t", "--table", required=True, help="Target SQL table name")
    parser.add_argument(
        "--if-exists",
        default="append",
        choices=["fail", "replace", "append"],
        help="Behavior when the target table already exists",
    )
    args = parser.parse_args()

    dump_csv(input_file=args.input_file, table=args.table, if_exists=args.if_exists)
