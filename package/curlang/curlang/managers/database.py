import json
import logging
import os
import sqlite3

from contextlib import contextmanager
from datetime import datetime
from sqlite3 import Error
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class DatabaseManager:
    """A modern SQLite database manager with advanced ORM-like features and connection pooling."""

    def __init__(self, curlang_directory: str, db_path: str) -> None:
        self.curlang_directory = curlang_directory
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize_schema()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections with automatic cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _execute(
            self,
            query: str,
            params: Union[Tuple, List, Dict] = (),
            *,
            commit: bool = True
    ) -> Tuple[Optional[int], int]:
        """
        Execute a write operation with advanced error handling and debugging.

        Returns:
            Tuple (lastrowid, rowcount) from the executed query
        """
        logger.debug("Executing query: %s\nParams: %s", query, params)

        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                lastrowid = cursor.lastrowid
                rowcount = cursor.rowcount
                if commit:
                    conn.commit()
                logger.debug("Operation successful. Rows affected: %d",
                             rowcount)
                return lastrowid, rowcount
        except Error as e:
            logger.error("Database error: %s\nQuery: %s\nParams: %s", e, query,
                         params)
            raise

    def _fetch(
            self,
            query: str,
            params: Union[Tuple, List, Dict] = (),
            fetch_all: bool = False
    ) -> Union[List[Dict], Optional[Dict]]:
        """Generic fetch method with row factory and error handling."""
        logger.debug("Fetching query: %s\nParams: %s", query, params)

        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                if fetch_all:
                    return [dict(row) for row in cursor.fetchall()]
                result = cursor.fetchone()
                return dict(result) if result else None
        except Error as e:
            logger.error("Database error: %s\nQuery: %s\nParams: %s", e, query,
                         params)
            raise

    def _initialize_schema(self) -> None:
        """Initialize database schema with atomic transaction."""
        schema = [
            """CREATE TABLE IF NOT EXISTS curlang_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_id TEXT NOT NULL,
                selected_text TEXT NOT NULL,
                comment TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS curlang_hooks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hook_name TEXT NOT NULL,
                hook_placement TEXT NOT NULL,
                hook_script TEXT NOT NULL,
                hook_type TEXT NOT NULL,
                expose_to_public_api BOOLEAN DEFAULT 0,
                show_on_frontpage BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS curlang_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL UNIQUE,
                value TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS curlang_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                pattern TEXT,
                datetimes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_run TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS curlang_source_hook_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                target_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_id, target_id)
            )""",
            """CREATE TABLE IF NOT EXISTS curlang_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_name, source_type)
            )"""
        ]

        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                for table in schema:
                    cursor.execute(table)
                conn.commit()
        except Error as e:
            logger.critical("Failed to initialize database schema: %s", e)
            raise

    def initialize_database(self) -> None:
        """Public method to initialize database schema."""
        self._initialize_schema()
        logger.info("Database initialization complete")

    # Comment operations
    def add_comment(self, block_id: str, selected_text: str,
                    comment: str) -> int:
        """Add a new comment and return its ID."""
        lastrowid, _ = self._execute(
            "INSERT INTO curlang_comments (block_id, selected_text, comment) VALUES (?, ?, ?)",
            (block_id, selected_text, comment)
        )
        return lastrowid or -1

    def delete_comment(self, comment_id: int) -> bool:
        """Delete a comment by ID. Returns True if operation succeeded."""
        _, rowcount = self._execute(
            "DELETE FROM curlang_comments WHERE id = ?",
            (comment_id,)
        )
        return rowcount > 0

    def get_all_comments(self) -> List[Dict[str, Any]]:
        """Retrieve all comments with metadata."""
        return self._fetch(
            """SELECT id, block_id, selected_text, comment, created_at
               FROM curlang_comments ORDER BY created_at DESC""",
            fetch_all=True
        )

    # Hook operations
    def add_hook(
            self,
            hook_name: str,
            hook_placement: str,
            hook_script: str,
            hook_type: str,
            expose_to_public_api: bool = False,
            show_on_frontpage: bool = False
    ) -> int:
        """Add a new hook and return its ID."""
        hook_name = hook_name.lower()
        lastrowid, _ = self._execute(
            """INSERT INTO curlang_hooks 
               (hook_name, hook_placement, hook_script, hook_type, expose_to_public_api, show_on_frontpage)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (hook_name, hook_placement, hook_script, hook_type,
             int(expose_to_public_api), int(show_on_frontpage))
        )
        self._sync_hooks_file()
        return lastrowid or -1

    def delete_hook(self, hook_id: int) -> bool:
        """Delete a hook and its related connections with atomic transaction."""
        try:
            hook = self.get_hook(hook_id)
            if not hook:
                return False

            with self._connection() as conn:
                cursor = conn.cursor()
                base_name = hook['hook_name'].split('-')[0]
                cursor.execute(
                    "DELETE FROM curlang_source_hook_mappings WHERE target_id LIKE ?",
                    (f"{base_name}-%",)
                )

                cursor.execute("DELETE FROM curlang_hooks WHERE id = ?",
                               (hook_id,))
                conn.commit()

            self._sync_hooks_file()
            self._sync_connections_file()
            return True
        except Error as e:
            logger.error("Failed to delete hook %d: %s", hook_id, e)
            return False

    def get_all_hooks(self) -> List[Dict[str, Any]]:
        """Retrieve all registered hooks with proper boolean conversion"""
        hooks = self._fetch(
            """SELECT id, hook_name, hook_placement, hook_script, hook_type,
                      expose_to_public_api, show_on_frontpage, created_at
               FROM curlang_hooks ORDER BY created_at DESC""",
            fetch_all=True
        )

        if hooks:
            for hook in hooks:
                hook['expose_to_public_api'] = bool(
                    hook['expose_to_public_api'])
                hook['show_on_frontpage'] = bool(hook['show_on_frontpage'])
        return hooks

    def get_hook(self, hook_id: int) -> Optional[Dict[str, Any]]:
        """Get a single hook by ID with proper boolean conversion"""
        hook = self._fetch(
            """SELECT id, hook_name, hook_placement, hook_script, hook_type,
                      expose_to_public_api, show_on_frontpage, created_at
               FROM curlang_hooks WHERE id = ?""",
            (hook_id,)
        )
        if hook:
            hook['expose_to_public_api'] = bool(hook['expose_to_public_api'])
            hook['show_on_frontpage'] = bool(hook['show_on_frontpage'])
        return hook

    def get_hook_by_name(self, hook_name: str) -> Optional[Dict[str, Any]]:
        """Get a hook by name with proper boolean conversion"""
        hook = self._fetch(
            """SELECT id, hook_name, hook_placement, hook_script, hook_type,
                      expose_to_public_api, show_on_frontpage, created_at
               FROM curlang_hooks WHERE hook_name = ?""",
            (hook_name,)
        )
        if hook:
            hook['expose_to_public_api'] = bool(hook['expose_to_public_api'])
            hook['show_on_frontpage'] = bool(hook['show_on_frontpage'])
        return hook

    def get_exposed_hooks(self) -> List[Dict[str, Any]]:
        """Retrieve all hooks marked for public API exposure with proper boolean conversion"""
        hooks = self._fetch(
            """SELECT id, hook_name, hook_placement, hook_script, hook_type,
                      expose_to_public_api, show_on_frontpage, created_at
               FROM curlang_hooks 
               WHERE expose_to_public_api = 1
               ORDER BY created_at DESC""",
            fetch_all=True
        )

        if hooks:
            for hook in hooks:
                hook['expose_to_public_api'] = bool(
                    hook['expose_to_public_api'])
                hook['show_on_frontpage'] = bool(hook['show_on_frontpage'])

        return hooks

    def hook_exists(self, hook_name: str) -> bool:
        """Check if a hook exists by name using modern query patterns."""
        result = self._fetch(
            "SELECT COUNT(*) AS count FROM curlang_hooks WHERE hook_name = ?",
            (hook_name,)
        )
        return bool(result['count']) if result else False

    def update_hook(
            self,
            hook_id: int,
            hook_name: str,
            hook_placement: str,
            hook_script: str,
            hook_type: str,
            expose_to_public_api: bool = False,
            show_on_frontpage: bool = False
    ) -> bool:
        """Update an existing hook's details."""
        _, rowcount = self._execute(
            """UPDATE curlang_hooks
               SET hook_name=?, hook_placement=?, hook_script=?, hook_type=?, expose_to_public_api=?, show_on_frontpage=?
               WHERE id=?""",
            (hook_name, hook_placement, hook_script, hook_type,
             int(expose_to_public_api), int(show_on_frontpage), hook_id)
        )

        if rowcount > 0:
            self._sync_hooks_file()
        return rowcount > 0

    # Metadata operations
    def set_metadata(self, key: str, value: str) -> None:
        """Set metadata key-value pair with automatic upsert."""
        self._execute(
            """INSERT INTO curlang_metadata (key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            (key, value)
        )

    def get_metadata(self, key: str) -> Optional[str]:
        """Retrieve metadata value by key."""
        result = self._fetch(
            "SELECT value FROM curlang_metadata WHERE key = ?",
            (key,)
        )
        return result['value'] if result else None

    def delete_metadata(self, key: str) -> bool:
        """Delete metadata entry by key."""
        _, rowcount = self._execute(
            "DELETE FROM curlang_metadata WHERE key = ?",
            (key,)
        )
        return rowcount > 0

    # Schedule operations
    def add_schedule(
            self,
            schedule_type: str,
            pattern: Optional[str] = None,
            datetimes: Optional[List[str]] = None
    ) -> int:
        """Add a new schedule and return its ID."""
        lastrowid, _ = self._execute(
            "INSERT INTO curlang_schedule (type, pattern, datetimes) VALUES (?, ?, ?)",
            (schedule_type, pattern,
             json.dumps(datetimes) if datetimes else None)
        )
        return lastrowid or -1

    def delete_schedule(self, schedule_id: int) -> bool:
        """Delete a schedule entry by ID."""
        _, rowcount = self._execute(
            "DELETE FROM curlang_schedule WHERE id = ?",
            (schedule_id,)
        )
        return rowcount > 0

    def get_all_schedules(self) -> List[Dict[str, Any]]:
        """Retrieve all schedules with parsed datetime information."""
        schedules = self._fetch(
            """SELECT id, type, pattern, datetimes, created_at, last_run
               FROM curlang_schedule ORDER BY created_at DESC""",
            fetch_all=True
        )
        for s in schedules:
            if s['datetimes']:
                s['datetimes'] = json.loads(s['datetimes'])
        return schedules

    def update_schedule_last_run(self, schedule_id: int,
                                 last_run: datetime) -> bool:
        """Update a schedule's last run timestamp."""
        _, rowcount = self._execute(
            "UPDATE curlang_schedule SET last_run = ? WHERE id = ?",
            (last_run.isoformat(), schedule_id)
        )
        return rowcount > 0

    # Source-hook mappings
    def add_source_hook_mapping(
            self,
            source_id: str,
            target_id: str,
            source_type: str,
            target_type: str
    ) -> int:
        """Create or update a source-hook mapping."""
        lastrowid, _ = self._execute(
            """INSERT INTO curlang_source_hook_mappings
               (source_id, target_id, source_type, target_type)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(source_id, target_id) DO UPDATE SET
               source_type=excluded.source_type, target_type=excluded.target_type""",
            (source_id, target_id, source_type, target_type)
        )
        self._sync_connections_file()
        return lastrowid or -1

    def delete_all_source_hook_mappings(self) -> bool:
        """Delete all source-hook mappings with atomic transaction."""
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM curlang_source_hook_mappings")
                conn.commit()

            self._sync_connections_file()
            return True
        except Error as e:
            logger.error("Error deleting all mappings: %s", e)
            return False

    def delete_source_hook_mapping(self, mapping_id: int) -> bool:
        """Delete a specific source-hook mapping by ID with transaction safety."""
        try:
            _, rowcount = self._execute(
                "DELETE FROM curlang_source_hook_mappings WHERE id = ?",
                (mapping_id,)
            )

            if rowcount > 0:
                self._sync_connections_file()

            return rowcount > 0

        except Error as e:
            logger.error("Error deleting mapping %d: %s", mapping_id, e)
            return False

    def get_all_source_hook_mappings(self) -> List[Dict[str, Any]]:
        """Retrieve all source-hook mappings."""
        return self._fetch(
            """SELECT id, source_id, target_id, source_type, target_type, created_at
               FROM curlang_source_hook_mappings ORDER BY created_at DESC""",
            fetch_all=True
        )

    def get_source_hook_mapping(self, mapping_id: int) -> Optional[
        Dict[str, Any]]:
        """Get a specific source-hook mapping by ID with proper error handling."""
        try:
            return self._fetch(
                """SELECT id, source_id, target_id, source_type, target_type, created_at
                   FROM curlang_source_hook_mappings
                   WHERE id = ?""",
                (mapping_id,)
            )
        except Error as e:
            logger.error("Error retrieving mapping %d: %s", mapping_id, e)
            return None

    def source_hook_mapping_exists(
            self, source_id: str,
            target_id: str
    ) -> bool:
        """Check if a source-hook mapping with the given source_id and target_id exists."""
        query = "SELECT 1 FROM curlang_source_hook_mappings WHERE source_id = ? AND target_id = ?"
        result = self._fetch(query, (source_id, target_id), fetch_all=False)
        return bool(result)

    # Source management
    def add_source(
            self,
            source_name: str,
            source_type: str,
            source_details: Optional[Dict[str, Any]] = None
    ) -> int:
        """Add a new data source and return its ID.
        Prevent adding sources with the same name and type."""
        try:
            lastrowid, _ = self._execute(
                """INSERT INTO curlang_sources (source_name, source_type, source_details) 
                   VALUES (?, ?, ?)""",
                (source_name, source_type,
                 json.dumps(source_details) if source_details else None)
            )
            self._sync_sources_file()
            return lastrowid or -1
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                logger.warning(
                    f"Source with name '{source_name}' and type '{source_type}' already exists."
                )
                return -1
            else:
                raise

    def delete_source(self, source_id: int) -> bool:
        """Delete a source and its mappings with atomic transaction."""
        try:
            source = self._fetch(
                "SELECT id, source_name FROM curlang_sources WHERE id = ?",
                (source_id,)
            )
            if not source:
                return False

            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM curlang_source_hook_mappings WHERE LOWER(source_id) = ?",
                    (source['source_name'].lower(),)
                )
                cursor.execute(
                    "DELETE FROM curlang_sources WHERE id = ?",
                    (source_id,)
                )
                conn.commit()

            self._sync_connections_file()
            self._sync_sources_file()
            return True

        except Error as e:
            logger.error("Error deleting source %d: %s", source_id, e)
            return False

    def get_all_sources(self) -> List[Dict[str, Any]]:
        """Retrieve all registered sources with parsed details."""
        sources = self._fetch(
            """SELECT id, source_name, source_type, source_details, created_at
               FROM curlang_sources ORDER BY created_at DESC""",
            fetch_all=True
        )
        for s in sources:
            if s['source_details']:
                s['source_details'] = json.loads(s['source_details'])
        return sources

    def get_source_by_id(self, source_id: int) -> Optional[Dict[str, Any]]:
        """Get source details by ID with proper deserialization."""
        source = self._fetch(
            """SELECT id, source_name, source_type, source_details, created_at
               FROM curlang_sources WHERE id = ?""",
            (source_id,)
        )
        if source and source.get('source_details'):
            try:
                source['source_details'] = json.loads(source['source_details'])
            except json.JSONDecodeError:
                source['source_details'] = None
        return source

    def source_exists(self, source_name: str) -> bool:
        """Check if a source with the given name exists."""
        query = "SELECT 1 FROM curlang_sources WHERE source_name = ?"
        result = self._fetch(query, (source_name,), fetch_all=False)
        return bool(result)

    def update_source(
            self,
            source_id: int,
            source_name: str,
            source_type: str,
            source_details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing source and sync files."""
        _, rowcount = self._execute(
            """UPDATE curlang_sources
               SET source_name=?, source_type=?, source_details=?
               WHERE id=?""",
            (source_name, source_type,
             json.dumps(source_details) if source_details else None,
             source_id)
        )

        if rowcount > 0:
            self._sync_sources_file()
            self._sync_connections_file()
        return rowcount > 0

    # File synchronization methods
    def _sync_hooks_file(self) -> None:
        """Maintain hooks.json file synchronization."""
        hooks = self.get_all_hooks()
        hooks_file = os.path.join(
            self.curlang_directory,
            "hooks.json"
        )
        data = {"hooks": [{
            "hook_id": h["id"],
            "hook_name": h["hook_name"],
            "hook_placement": h["hook_placement"],
            "hook_script": h["hook_script"],
            "hook_type": h["hook_type"],
            "expose_to_public_api": bool(h["expose_to_public_api"]),
            "show_on_frontpage": bool(h["show_on_frontpage"])
        } for h in hooks]}

        with open(hooks_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _sync_connections_file(self) -> None:
        """Maintain connections.json file synchronization."""
        mappings = self.get_all_source_hook_mappings()
        connections_file = os.path.join(
            self.curlang_directory,
            "connections.json"
        )
        data = {"connections": [{
            "source_id": m["source_id"],
            "target_id": m["target_id"],
            "source_type": m["source_type"],
            "target_type": m["target_type"]
        } for m in mappings]}

        with open(connections_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _sync_sources_file(self) -> None:
        """Sync sources with sources.json, adding new sources from the file to the database and vice versa."""
        sources_file = os.path.join(
            self.curlang_directory,
            "sources.json"
        )

        logger.debug(f"Syncing sources from: {sources_file}")

        try:
            with open(sources_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Sources file not found: {sources_file}")
            return
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {sources_file}")
            return

        logger.debug(f"Data read from sources.json: {data}")

        sources_from_db = {
            (s["source_name"], s["source_type"])
            for s in self.get_all_sources()
        }

        if "sources" not in data:
            logger.warning("No 'sources' key found in sources.json")
            return

        new_sources = []

        for source in data["sources"]:
            source_name = source["source_name"]
            source_type = source["source_type"]
            source_details = source.get("source_details")

            if (source_name, source_type) not in sources_from_db:
                logger.info(f"Adding source: {source_name}, {source_type}")
                self.add_source(source_name, source_type, source_details)
            else:
                logger.debug(
                    f"Source already exists: {source_name}, {source_type}")

        updated_sources = [
            {
                "source_name": s["source_name"],
                "source_type": s["source_type"],
                "source_details": s.get("source_details")
            }
            for s in self.get_all_sources()
        ]

        try:
            with open(sources_file, 'w') as f:
                json.dump({"sources": updated_sources}, f, indent=4)
            logger.debug(f"Updated sources.json written successfully")
        except IOError:
            logger.error(f"Failed to write updated sources to {sources_file}")
