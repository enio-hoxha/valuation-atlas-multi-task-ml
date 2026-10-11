import pandas as pd
import sys
import os
import argparse

def infer_pg_type(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "DOUBLE PRECISION"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    else:
        return "TEXT"

def generate_psql_script(csv_path, table_name=None):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    if table_name is None:
        table_name = os.path.splitext(os.path.basename(csv_path))[0]
        # Ensure valid postgres table name
        table_name = table_name.replace("-", "_").replace(" ", "_").lower()

    # Read a sample to infer types
    df = pd.read_csv(csv_path, nrows=100)
    
    columns = []
    for col, dtype in df.dtypes.items():
        pg_type = infer_pg_type(dtype)
        # Quote column names to handle special characters/reserved words
        columns.append(f'    "{col}" {pg_type}')

    create_table = f"CREATE TABLE IF NOT EXISTS {table_name} (\n" + ",\n".join(columns) + "\n);"
    
    # \copy is a psql meta-command that runs from the client side, 
    # making it more portable than the SQL COPY command.
    abs_path = os.path.abspath(csv_path)
    copy_cmd = f"\\copy {table_name} FROM '{abs_path}' WITH (FORMAT csv, HEADER true);"

    print(f"-- Script to import {csv_path} into table {table_name}")
    print(create_table)
    print(copy_cmd)
    print(f"\n-- To run this, use:")
    print(f"-- psql -h your_host -U your_user -d your_db -f generated_script.sql")
    print(f"-- OR pipe it directly:")
    print(f"-- python csv_to_psql.py {csv_path} | psql -h your_host -U your_user -d your_db")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PostgreSQL import script from CSV.")
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument("--table", help="Target table name (defaults to filename)")
    
    args = parser.parse_args()
    generate_psql_script(args.csv_path, args.table)
