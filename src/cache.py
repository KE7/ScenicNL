import json
import sqlite3
import threading
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

# Much of this class and code is inspired and borrowed from TensorTrust AI

@dataclass(frozen=True)
class APIError:
    """
    We don't want to blindly store APIErrors as values so we create a class to capture this
    """
    message: str

class Cache:
    '''
    Caching layer backed by SQLite3 database.
    '''

    def __init__(self, path: str | Path) -> None:
        self.path = path
        # Connection and lock will be populated by a context manager.
        self._conn = None
        self._lock = None

    def create_table(self) -> None:
        '''
        Create tables if they don't exist.
        '''
        with self.lock:
            with self.conn:
                self.conn.execute(
                    '''
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                    '''
                )

    def set(self, key: str, value: list[Union[str, APIError]]) -> None:
        # Convert APIErrors into dicts so they're json serializable
        def make_serializable(item):
            if isinstance(item, APIError):
                return {
                    "__type__": "APIError",
                    **asdict(item),
                }
            return item

        value_json = json.dumps([make_serializable(v) for v in value])
        with self.lock:
            with self.conn:
                self.conn.execute(
                    """
                INSERT OR REPLACE INTO kv_store (key, value)
                VALUES (?, ?);
                """,
                    (key, value_json),
                )

    def get(self, key: str) -> list[Union[str, APIError]]:
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT value FROM kv_store WHERE key = ?;
                """,
                (key,),
            )
            row = cursor.fetchone()
        if row is None:
            return []
        results = json.loads(row[0])

        # Convert APIError dicts back into APIErrors
        def deserialize_into_api_error(item):
            if isinstance(item, dict):
                if item.get("__type__") != "APIError":
                    warnings.warn(f"Expected an APIError when pulling from the cache." +
                                  "Found Unknown type {item.get('__type__')}")
                item = dict(item)
                item.pop("__type__")
                return APIError(**item)
            return item
        
        return [deserialize_into_api_error(v) for v in results]
        

    def delete(self, key: str) -> None:
        with self.lock:
            with self.conn:
                self.conn.execute(
                    """
                    DELETE FROM kv_store WHERE key = ?;
                    """,
                    (key,),
                )
        

    def __enter__(self) -> "Cache":
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        self.create_table()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        with self.lock:
            self.conn.close()
        self._conn = None
        self._lock = None


    @property
    def lock(self) -> threading.Lock:
        if self._lock is None:
            raise ValueError("Cache not initialized. Please use a context manager.")
        return self._lock
    
    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise ValueError("Cache not initialized. Please use a context manager.")
        return self._conn