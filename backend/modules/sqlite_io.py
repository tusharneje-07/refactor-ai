"""
sqlite_io.py
============
Class-based module for easy, safe read/write access to SQLite
(.db / .sqlite) files.

Usage
-----
    from sqlite_io import SQLiteDB

    db = SQLiteDB("app.db")

    error, data = db.read("users", "SELECT * FROM {table} WHERE age > ?",
                           json=True, params=(18,))

    error, data = db.write("users", "INSERT INTO {table} (name, age) VALUES (?, ?)",
                            params=("Alice", 30))

    error, data = db.runQuery(
        "UPDATE users SET age = ? WHERE name = ? RETURNING id, age",
        params=(31, "Alice")
    )

One `SQLiteDB` object per database file. A new short-lived sqlite3
connection is opened and closed on every call -- this keeps the class
simple, thread-safe, and avoids stale/leaked connections. SQLite's
connection overhead is negligible for this usage pattern.

Public API
----------
db.read(table_name, sql_query, json=False, params=())
    Read-only. Accepts SELECT, CTEs, joins, PRAGMA reads, EXPLAIN, etc.
    The connection is opened strictly read-only, so any statement that
    tries to mutate data/schema is rejected by SQLite itself.

db.write(table_name, sql_query, params=())
    Any DDL/DML statement -- INSERT, UPDATE, DELETE, CREATE/ALTER/DROP,
    CREATE INDEX/VIEW, PRAGMA, etc. Auto-commits. If the statement
    returns rows (e.g. a RETURNING clause), they're included in the result.

db.runQuery(sql_query, table_name=None, params=(), json=False)
    Handles ANY single SQL statement, read or write alike -- SELECT,
    INSERT, UPDATE, DELETE, CREATE/ALTER/DROP, PRAGMA, joins, CTEs,
    RETURNING clauses, and so on. Auto-commits when the statement
    mutates data and captures any rows it returns. This is the
    general-purpose entry point when you don't want to think about
    whether a query is a "read" or a "write".

Every method returns (error: bool, result_data).

SQL-injection protection
-------------------------
1. `table_name` is optional. When given, it is validated as a strict SQL
   identifier (letters, digits, underscore, must not start with a digit).
   If `sql_query` contains the literal token "{table}", it is substituted
   with a properly quoted version of `table_name` -- never with raw string
   concatenation. Pass table_name=None for joins/multi-table/PRAGMA
   queries not tied to one table.
2. Every call executes exactly ONE statement via sqlite3's parameterized
   `execute()`. This blocks stacked-query injection (e.g. "...; DROP TABLE").
   Any values (user input, filters, etc.) must be passed through `params`
   using "?" placeholders in `sql_query` -- never embedded directly into
   the query string.
3. `read()` opens the database in true read-only mode (SQLite URI
   `mode=ro` + `PRAGMA query_only`), so a mutating statement passed to
   `read()` is rejected by SQLite itself, regardless of intent.
"""

import os
import re
import sqlite3
import json as _json
from urllib.parse import quote as _urlquote

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name):
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid table name: {name!r}")


def _quote_identifier(name):
    return '"' + name.replace('"', '""') + '"'


class SQLiteDB:
    """Class-based read/write/runQuery access to a single SQLite database file."""

    def __init__(self, filepath):
        self.filepath = filepath

    # ---------- internal helpers ----------

    @staticmethod
    def _prepare_query(table_name, sql_query):
        if not isinstance(sql_query, str) or not sql_query.strip():
            raise ValueError("sql_query must be a non-empty string")
        if table_name:
            _validate_identifier(table_name)
            if "{table}" in sql_query:
                sql_query = sql_query.replace("{table}", _quote_identifier(table_name))
        return sql_query

    def _connect_readonly(self):
        if not os.path.isfile(self.filepath):
            raise FileNotFoundError(f"Database file not found: {self.filepath}")
        uri = f"file:{_urlquote(os.path.abspath(self.filepath))}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = 1;")
        return conn

    def _connect_readwrite(self):
        conn = sqlite3.connect(self.filepath)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _execute(self, table_name, sql_query, params, readonly):
        """Shared execution path for write() and runQuery()/read()."""
        query = self._prepare_query(table_name, sql_query)
        conn = self._connect_readonly() if readonly else self._connect_readwrite()
        try:
            cursor = conn.execute(query, params)
            data = [dict(row) for row in cursor.fetchall()] if cursor.description is not None else None
            if not readonly:
                conn.commit()
            return {
                "rows_affected": cursor.rowcount,
                "last_row_id": cursor.lastrowid,
                "data": data,
            }
        except (sqlite3.Error, ValueError):
            if not readonly:
                conn.rollback()
            raise
        finally:
            conn.close()

    # ---------- public API ----------

    def read(self, table_name, sql_query, json=False, params=()):
        """
        Execute a single read-only statement.

        Returns:
            (error, result_data)
            error=False -> result_data is a list[dict] (or a JSON string if json=True)
            error=True  -> result_data is a human-readable error message (str)
        """
        try:
            result = self._execute(table_name, sql_query, params, readonly=True)
        except (sqlite3.Error, ValueError, FileNotFoundError) as e:
            return True, str(e)

        data = result["data"] if result["data"] is not None else []
        if json:
            try:
                return False, _json.dumps(data, default=str)
            except (TypeError, ValueError) as e:
                return True, f"JSON serialization failed: {e}"
        return False, data

    def write(self, table_name, sql_query, params=()):
        """
        Execute a single write/DDL/DML statement. Auto-commits.

        Returns:
            (error, result_data)
            error=False -> result_data = {
                "rows_affected": int,       # cursor.rowcount (-1 for DDL/PRAGMA)
                "last_row_id": int or None, # cursor.lastrowid, when applicable
                "data": list[dict] or None  # populated only if the statement
                                             # returned rows (e.g. RETURNING)
            }
            error=True  -> result_data is a human-readable error message (str)
        """
        try:
            return False, self._execute(table_name, sql_query, params, readonly=False)
        except (sqlite3.Error, ValueError) as e:
            return True, str(e)

    def runQuery(self, sql_query, table_name=None, params=(), json=False):
        """
        Execute ANY single SQL statement -- SELECT, INSERT, UPDATE, DELETE,
        CREATE/ALTER/DROP, PRAGMA, joins, CTEs, RETURNING clauses, etc.
        Auto-commits if the statement mutates data; captures any rows the
        statement produced (works for SELECT and for RETURNING alike).

        Args:
            sql_query: a single SQL statement. Use "?" for parameters.
            table_name: optional; fills the "{table}" placeholder if present.
            params: tuple/dict of values for "?" / ":name" placeholders.
            json: if True, result_data is returned as a JSON string.

        Returns:
            (error, result_data)
            error=False -> result_data = {
                "rows_affected": int,
                "last_row_id": int or None,
                "data": list[dict] or None
            } (serialized to a JSON string instead, if json=True)
            error=True  -> result_data is a human-readable error message (str)
        """
        try:
            result = self._execute(table_name, sql_query, params, readonly=False)
        except (sqlite3.Error, ValueError) as e:
            return True, str(e)

        if json:
            try:
                return False, _json.dumps(result, default=str)
            except (TypeError, ValueError) as e:
                return True, f"JSON serialization failed: {e}"
        return False, result


if __name__ == "__main__":
    # Self-test / usage demo covering full CRUD + extras
    demo_db = "demo.db"
    db = SQLiteDB(demo_db)

    print("CREATE  ->", db.runQuery(
        "CREATE TABLE IF NOT EXISTS {table} "
        "(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age INTEGER)",
        table_name="users"
    ))

    print("CREATE  ->", db.runQuery(
        "CREATE TABLE IF NOT EXISTS {table} "
        "(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, total REAL)",
        table_name="orders"
    ))

    err, res = db.write("users", "INSERT INTO {table} (name, age) VALUES (?, ?)",
                         params=("Alice", 30))
    print("INSERT  ->", err, res)
    alice_id = res["last_row_id"]

    print("INSERT  ->", db.runQuery(
        "INSERT INTO {table} (user_id, total) VALUES (?, ?)",
        table_name="orders", params=(alice_id, 99.5)
    ))

    print("SELECT (json) ->", db.read(
        "users", "SELECT * FROM {table} WHERE age > ?", json=True, params=(18,)
    ))

    print("UPDATE+RETURNING (runQuery) ->", db.runQuery(
        "UPDATE {table} SET age = ? WHERE name = ? RETURNING id, age",
        table_name="users", params=(31, "Alice")
    ))

    print("JOIN (runQuery, no table_name) ->", db.runQuery(
        "SELECT u.name, o.total FROM users u JOIN orders o ON o.user_id = u.id"
    ))

    print("DELETE (runQuery) ->", db.runQuery(
        "DELETE FROM {table} WHERE user_id = ?", table_name="orders", params=(alice_id,)
    ))

    print("DROP (runQuery) ->", db.runQuery("DROP TABLE {table}", table_name="orders"))

    # Sanity check: read() must reject write attempts
    print("READ blocks write ->", db.read("users", "DELETE FROM {table}"))

    os.remove(demo_db)