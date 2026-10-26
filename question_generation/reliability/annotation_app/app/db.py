"""SQLite helpers for ESG QG annotation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "annotation.db"
QUESTIONS_PATH = DATA_DIR / "questions_2015.json"
SECRETS_PATH = ROOT / "secrets.json"
ANNOTATIONS_JSON = DATA_DIR / "annotations.json"
SCORES_DIR = DATA_DIR / "scores"

DIMS = [
    "clarity",
    "esg_relevance",
    "answerability",
    "answer_correctness",
    "evidence_support",
    "question_type_fit",
    "type_specific_validity",
]

TYPE_HINTS = {
    "Fact Retrieval": "L1: single verifiable fact; minimal reasoning.",
    "Complex Reasoning": "L2: multi-hop / multi-condition reasoning over evidence.",
    "Contextual Summarize": "L3: multi-point coverage summary grounded in evidence.",
    "Creative Generation": "L4: fluent rewrite allowed, but must stay faithful to evidence.",
}


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('annotator', 'admin')),
            display_name TEXT NOT NULL,
            last_question_index INTEGER NOT NULL DEFAULT 0,
            last_active_at TEXT
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS questions (
            seq INTEGER PRIMARY KEY,
            question_id TEXT NOT NULL UNIQUE,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            question_type TEXT NOT NULL,
            evidence TEXT NOT NULL,
            evidence_triple TEXT,
            document_id TEXT,
            company TEXT,
            pack_id TEXT,
            payload_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            clarity INTEGER CHECK(clarity BETWEEN 1 AND 3),
            esg_relevance INTEGER CHECK(esg_relevance BETWEEN 1 AND 3),
            answerability INTEGER CHECK(answerability BETWEEN 1 AND 3),
            answer_correctness INTEGER CHECK(answer_correctness BETWEEN 1 AND 3),
            evidence_support INTEGER CHECK(evidence_support BETWEEN 1 AND 3),
            question_type_fit INTEGER CHECK(question_type_fit BETWEEN 1 AND 3),
            type_specific_validity INTEGER CHECK(type_specific_validity BETWEEN 1 AND 3),
            notes TEXT,
            updated_at TEXT NOT NULL,
            UNIQUE(question_id, user_id),
            FOREIGN KEY(question_id) REFERENCES questions(question_id),
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()


DEFAULT_USERS = [
    {"username": "annotator1", "password": "annotator1", "role": "annotator", "display_name": "Annotator 1"},
    {"username": "annotator2", "password": "annotator2", "role": "annotator", "display_name": "Annotator 2"},
    {"username": "annotator3", "password": "annotator3", "role": "annotator", "display_name": "Annotator 3"},
    {"username": "admin", "password": "admin123", "role": "admin", "display_name": "Admin"},
]


def load_secrets() -> dict[str, Any]:
    """Load secrets.json, merge default users, override passwords from env (Render)."""
    import os
    import re

    if SECRETS_PATH.is_file():
        data = json.loads(SECRETS_PATH.read_text(encoding="utf-8"))
    else:
        data = {"session_secret": "change-me", "users": []}

    by_name = {u["username"]: dict(u) for u in DEFAULT_USERS}
    for u in data.get("users") or []:
        name = str(u.get("username") or "").strip()
        if not name:
            continue
        base = by_name.get(
            name,
            {
                "username": name,
                "password": name,
                "role": "annotator",
                "display_name": name,
            },
        )
        base.update({k: v for k, v in u.items() if v is not None})
        if base.get("role") not in ("annotator", "admin"):
            base["role"] = "annotator"
        by_name[name] = base

    env_secret = os.getenv("ANNOTATION_SESSION_SECRET")
    if env_secret:
        data["session_secret"] = env_secret

    # Prefer explicit env for known accounts; also ANNOTATION_PASSWORD_<USERNAME>
    pw_map = {
        "annotator1": os.getenv("ANNOTATION_PASSWORD_ANNOTATOR1"),
        "annotator2": os.getenv("ANNOTATION_PASSWORD_ANNOTATOR2"),
        "annotator3": os.getenv("ANNOTATION_PASSWORD_ANNOTATOR3"),
        "admin": os.getenv("ANNOTATION_PASSWORD_ADMIN"),
    }
    users = []
    # defaults first for stable order, then any extra usernames
    ordered = ["annotator1", "annotator2", "annotator3", "admin"] + [
        n for n in by_name.keys() if n not in ("annotator1", "annotator2", "annotator3", "admin")
    ]
    for name in ordered:
        u = dict(by_name[name])
        override = pw_map.get(name) or os.getenv(
            "ANNOTATION_PASSWORD_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper()
        )
        if override:
            u["password"] = override
        users.append(u)
    data["users"] = users
    return data


def hash_password(password: str, salt: str = "esg-qg-annotation") -> str:
    import hashlib

    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120000).hex()


def ensure_users(conn: sqlite3.Connection) -> None:
    secrets = load_secrets()
    for u in secrets.get("users", []):
        ph = hash_password(str(u["password"]))
        conn.execute(
            """
            INSERT INTO users (username, password_hash, role, display_name)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
              password_hash=excluded.password_hash,
              role=excluded.role,
              display_name=excluded.display_name
            """,
            (u["username"], ph, u["role"], u.get("display_name") or u["username"]),
        )
    conn.commit()


def import_questions(conn: sqlite3.Connection, force: bool = False) -> int:
    n = conn.execute("SELECT COUNT(*) AS c FROM questions").fetchone()["c"]
    if n > 0 and not force:
        return int(n)
    if force:
        conn.execute("DELETE FROM annotations")
        conn.execute("DELETE FROM questions")
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    rows = data.get("questions") or []
    for i, q in enumerate(rows):
        conn.execute(
            """
            INSERT INTO questions (
              seq, question_id, question, answer, question_type, evidence,
              evidence_triple, document_id, company, pack_id, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                i,
                q["id"],
                q.get("question") or "",
                q.get("answer") or "",
                q.get("question_type") or "",
                q.get("evidence") or "",
                q.get("evidence_triple") or "",
                q.get("document_id") or "",
                q.get("company") or "",
                q.get("pack_id") or "",
                json.dumps(q, ensure_ascii=False),
            ),
        )
    conn.commit()
    return len(rows)


def init_db(force_reimport: bool = False) -> None:
    conn = connect()
    try:
        init_schema(conn)
        ensure_users(conn)
        import_questions(conn, force=force_reimport)
        export_annotations_json(conn)
    finally:
        conn.close()


def row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    return {k: row[k] for k in row.keys()}


def annotation_complete(ann: Optional[dict[str, Any]]) -> bool:
    if not ann:
        return False
    return all(ann.get(d) in (1, 2, 3) for d in DIMS)


def export_annotations_json(conn: sqlite3.Connection) -> Path:
    """Mirror all scores + progress into readable local JSON files."""
    from datetime import datetime, timezone

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    users = conn.execute(
        """
        SELECT id, username, display_name, role, last_question_index, last_active_at
        FROM users ORDER BY id
        """
    ).fetchall()
    by_user: dict[str, Any] = {}
    for u in users:
        if u["role"] != "annotator":
            continue
        rows = conn.execute(
            f"""
            SELECT a.question_id, q.seq, q.question_type, q.company,
                   {', '.join('a.' + d for d in DIMS)},
                   a.notes, a.updated_at
            FROM annotations a
            JOIN questions q ON q.question_id = a.question_id
            WHERE a.user_id = ?
            ORDER BY q.seq
            """,
            (u["id"],),
        ).fetchall()
        scores: dict[str, Any] = {}
        completed = 0
        for r in rows:
            item = {d: r[d] for d in DIMS}
            item.update(
                {
                    "seq": r["seq"],
                    "question_type": r["question_type"],
                    "company": r["company"],
                    "notes": r["notes"],
                    "updated_at": r["updated_at"],
                }
            )
            scores[r["question_id"]] = item
            if all(r[d] in (1, 2, 3) for d in DIMS):
                completed += 1
        payload = {
            "username": u["username"],
            "display_name": u["display_name"],
            "last_question_index": u["last_question_index"],
            "last_active_at": u["last_active_at"],
            "completed": completed,
            "total_scored_rows": len(scores),
            "scores": scores,
        }
        by_user[u["username"]] = payload
        (SCORES_DIR / f"{u['username']}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    total_q = conn.execute("SELECT COUNT(*) AS c FROM questions").fetchone()["c"]
    combined = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_questions": total_q,
        "annotators": by_user,
    }
    ANNOTATIONS_JSON.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ANNOTATIONS_JSON
