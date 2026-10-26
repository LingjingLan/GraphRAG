"""ESG QG annotation API + static UI."""

from __future__ import annotations

import csv
import io
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .db import (
    DIMS,
    ROOT,
    TYPE_HINTS,
    annotation_complete,
    connect,
    export_annotations_json,
    hash_password,
    init_db,
    row_to_dict,
)

STATIC = ROOT / "static"
COOKIE_NAME = "esg_qg_session"
SESSION_DAYS = 14

app = FastAPI(title="ESG QG Annotation", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


class LoginBody(BaseModel):
    username: str
    password: str


class CreateUserBody(BaseModel):
    username: str
    password: str
    display_name: Optional[str] = None
    role: str = "annotator"


class RegisterBody(BaseModel):
    username: str
    password: str
    display_name: Optional[str] = None
    invite_code: Optional[str] = None


class ScoreBody(BaseModel):
    clarity: int = Field(ge=1, le=3)
    esg_relevance: int = Field(ge=1, le=3)
    answerability: int = Field(ge=1, le=3)
    answer_correctness: int = Field(ge=1, le=3)
    evidence_support: int = Field(ge=1, le=3)
    question_type_fit: int = Field(ge=1, le=3)
    type_specific_validity: int = Field(ge=1, le=3)
    notes: Optional[str] = None
    seq: Optional[int] = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


@app.on_event("startup")
def _startup() -> None:
    init_db(force_reimport=False)


def get_session_user(request: Request) -> Optional[dict[str, Any]]:
    sid = request.cookies.get(COOKIE_NAME)
    if not sid:
        return None
    conn = connect()
    try:
        row = conn.execute(
            """
            SELECT u.*, s.expires_at
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.session_id = ?
            """,
            (sid,),
        ).fetchone()
        if not row:
            return None
        if datetime.fromisoformat(row["expires_at"]) < utc_now():
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
            conn.commit()
            return None
        return row_to_dict(row)
    finally:
        conn.close()


def require_user(request: Request) -> dict[str, Any]:
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in")
    return user


def require_admin(request: Request) -> dict[str, Any]:
    user = require_user(request)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return user


def touch_user(conn, user_id: int, last_index: Optional[int] = None) -> None:
    if last_index is None:
        conn.execute(
            "UPDATE users SET last_active_at = ? WHERE id = ?",
            (iso(utc_now()), user_id),
        )
    else:
        conn.execute(
            "UPDATE users SET last_active_at = ?, last_question_index = ? WHERE id = ?",
            (iso(utc_now()), int(last_index), user_id),
        )


def count_completed(conn, user_id: int) -> int:
    return conn.execute(
        f"""
        SELECT COUNT(*) AS c FROM annotations
        WHERE user_id = ?
          AND {' AND '.join(f'{d} IS NOT NULL' for d in DIMS)}
        """,
        (user_id,),
    ).fetchone()["c"]


def total_questions(conn) -> int:
    return conn.execute("SELECT COUNT(*) AS c FROM questions").fetchone()["c"]


def get_annotation(conn, question_id: str, user_id: int) -> Optional[dict[str, Any]]:
    return row_to_dict(
        conn.execute(
            "SELECT * FROM annotations WHERE question_id = ? AND user_id = ?",
            (question_id, user_id),
        ).fetchone()
    )


def question_payload(conn, seq: int, user_id: int) -> dict[str, Any]:
    total = total_questions(conn)
    if total == 0:
        raise HTTPException(status_code=404, detail="No questions imported")
    if seq < 0 or seq >= total:
        raise HTTPException(status_code=404, detail="Question out of range")
    q = row_to_dict(conn.execute("SELECT * FROM questions WHERE seq = ?", (seq,)).fetchone())
    assert q is not None
    ann = get_annotation(conn, q["question_id"], user_id)
    scores = {d: (ann or {}).get(d) for d in DIMS} if ann else {d: None for d in DIMS}
    return {
        "seq": seq,
        "total": total,
        "completed": count_completed(conn, user_id),
        "question": {
            "id": q["question_id"],
            "question": q["question"],
            "answer": q["answer"],
            "question_type": q["question_type"],
            "evidence": q["evidence"],
            "evidence_triple": q["evidence_triple"],
            "document_id": q["document_id"],
            "company": q["company"],
            "pack_id": q["pack_id"],
            "type_hint": TYPE_HINTS.get(q["question_type"], ""),
        },
        "scores": scores,
        "notes": (ann or {}).get("notes"),
        "complete": annotation_complete(ann),
        "dims": DIMS,
    }


@app.post("/api/auth/login")
def login(body: LoginBody, response: Response):
    conn = connect()
    try:
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?", (body.username.strip(),)
        ).fetchone()
        if not user or user["password_hash"] != hash_password(body.password):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        sid = secrets.token_urlsafe(32)
        exp = utc_now() + timedelta(days=SESSION_DAYS)
        conn.execute(
            "INSERT INTO sessions (session_id, user_id, expires_at) VALUES (?, ?, ?)",
            (sid, user["id"], iso(exp)),
        )
        touch_user(conn, user["id"])
        conn.commit()
        response.set_cookie(
            key=COOKIE_NAME,
            value=sid,
            httponly=True,
            samesite="lax",
            max_age=SESSION_DAYS * 86400,
            path="/",
        )
        return {
            "ok": True,
            "user": {
                "username": user["username"],
                "display_name": user["display_name"],
                "role": user["role"],
                "last_question_index": user["last_question_index"],
            },
        }
    finally:
        conn.close()


@app.get("/api/auth/register_config")
def register_config():
    """Frontend uses this to know if invite code is required."""
    enabled = os.getenv("ALLOW_PUBLIC_REGISTER", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    need_invite = bool(os.getenv("REGISTER_INVITE_CODE", "").strip())
    return {"enabled": enabled, "invite_required": need_invite}


@app.post("/api/auth/register")
def register(body: RegisterBody, response: Response):
    """Public self-registration as annotator only."""
    enabled = os.getenv("ALLOW_PUBLIC_REGISTER", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if not enabled:
        raise HTTPException(status_code=403, detail="Public registration is disabled")

    expected = os.getenv("REGISTER_INVITE_CODE", "").strip()
    if expected and (body.invite_code or "").strip() != expected:
        raise HTTPException(status_code=403, detail="Invalid invite code")

    username = body.username.strip()
    password = body.password
    if not re.fullmatch(r"[A-Za-z0-9_\-.]{2,32}", username):
        raise HTTPException(
            status_code=400,
            detail="Username: 2–32 chars, letters/numbers/_-.",
        )
    if len(password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters")
    if username.lower() == "admin":
        raise HTTPException(status_code=400, detail="Username reserved")

    display = (body.display_name or username).strip()
    conn = connect()
    try:
        exists = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if exists:
            raise HTTPException(status_code=409, detail="Username already taken")
        conn.execute(
            """
            INSERT INTO users (username, password_hash, role, display_name)
            VALUES (?, ?, 'annotator', ?)
            """,
            (username, hash_password(password), display),
        )
        conn.commit()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        sid = secrets.token_urlsafe(32)
        exp = utc_now() + timedelta(days=SESSION_DAYS)
        conn.execute(
            "INSERT INTO sessions (session_id, user_id, expires_at) VALUES (?, ?, ?)",
            (sid, user["id"], iso(exp)),
        )
        touch_user(conn, user["id"])
        conn.commit()
        export_annotations_json(conn)
        response.set_cookie(
            key=COOKIE_NAME,
            value=sid,
            httponly=True,
            samesite="lax",
            max_age=SESSION_DAYS * 86400,
            path="/",
        )
        return {
            "ok": True,
            "user": {
                "username": user["username"],
                "display_name": user["display_name"],
                "role": "annotator",
                "last_question_index": 0,
            },
        }
    finally:
        conn.close()


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    sid = request.cookies.get(COOKIE_NAME)
    if sid:
        conn = connect()
        try:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
            conn.commit()
        finally:
            conn.close()
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/api/me")
def me(user: dict[str, Any] = Depends(require_user)):
    conn = connect()
    try:
        return {
            "username": user["username"],
            "display_name": user["display_name"],
            "role": user["role"],
            "last_question_index": user["last_question_index"],
            "completed": count_completed(conn, user["id"]),
            "total": total_questions(conn),
        }
    finally:
        conn.close()


@app.get("/api/me/state")
def my_state(user: dict[str, Any] = Depends(require_user)):
    if user["role"] == "admin":
        return {"role": "admin", "redirect": "/admin.html"}
    conn = connect()
    try:
        idx = int(user.get("last_question_index") or 0)
        total = total_questions(conn)
        if total and idx >= total:
            idx = total - 1
        payload = question_payload(conn, idx, user["id"])
        touch_user(conn, user["id"], last_index=idx)
        conn.commit()
        return payload
    finally:
        conn.close()


@app.get("/api/questions/{seq}")
def get_question(seq: int, user: dict[str, Any] = Depends(require_user)):
    if user["role"] == "admin":
        raise HTTPException(status_code=403, detail="Annotator only")
    conn = connect()
    try:
        payload = question_payload(conn, seq, user["id"])
        touch_user(conn, user["id"], last_index=seq)
        conn.commit()
        return payload
    finally:
        conn.close()


@app.put("/api/questions/{question_id}/scores")
def save_scores(
    question_id: str,
    body: ScoreBody,
    user: dict[str, Any] = Depends(require_user),
):
    if user["role"] == "admin":
        raise HTTPException(status_code=403, detail="Annotator only")
    conn = connect()
    try:
        q = conn.execute(
            "SELECT seq FROM questions WHERE question_id = ?", (question_id,)
        ).fetchone()
        if not q:
            raise HTTPException(status_code=404, detail="Question not found")
        now = iso(utc_now())
        conn.execute(
            f"""
            INSERT INTO annotations (
              question_id, user_id, {', '.join(DIMS)}, notes, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(question_id, user_id) DO UPDATE SET
              clarity=excluded.clarity,
              esg_relevance=excluded.esg_relevance,
              answerability=excluded.answerability,
              answer_correctness=excluded.answer_correctness,
              evidence_support=excluded.evidence_support,
              question_type_fit=excluded.question_type_fit,
              type_specific_validity=excluded.type_specific_validity,
              notes=excluded.notes,
              updated_at=excluded.updated_at
            """,
            (
                question_id,
                user["id"],
                body.clarity,
                body.esg_relevance,
                body.answerability,
                body.answer_correctness,
                body.evidence_support,
                body.question_type_fit,
                body.type_specific_validity,
                body.notes,
                now,
            ),
        )
        next_index = body.seq if body.seq is not None else int(q["seq"])
        touch_user(conn, user["id"], last_index=next_index)
        conn.commit()
        export_annotations_json(conn)
        return {
            "ok": True,
            "completed": count_completed(conn, user["id"]),
            "total": total_questions(conn),
            "complete": True,
        }
    finally:
        conn.close()


@app.get("/api/progress/next_incomplete")
def next_incomplete(user: dict[str, Any] = Depends(require_user)):
    if user["role"] == "admin":
        raise HTTPException(status_code=403, detail="Annotator only")
    conn = connect()
    try:
        row = conn.execute(
            f"""
            SELECT q.seq FROM questions q
            LEFT JOIN annotations a
              ON a.question_id = q.question_id AND a.user_id = ?
            WHERE a.id IS NULL
               OR NOT ({' AND '.join(f'a.{d} IS NOT NULL' for d in DIMS)})
            ORDER BY q.seq ASC
            LIMIT 1
            """,
            (user["id"],),
        ).fetchone()
        if not row:
            return {"seq": None, "done": True}
        return {"seq": row["seq"], "done": False}
    finally:
        conn.close()


def _annotations_csv_bytes(conn, user_id: Optional[int] = None) -> bytes:
    if user_id is None:
        rows = conn.execute(
            """
            SELECT q.seq, q.question_id, q.question_type, q.company,
                   u.username,
                   a.clarity, a.esg_relevance, a.answerability, a.answer_correctness,
                   a.evidence_support, a.question_type_fit, a.type_specific_validity,
                   a.notes, a.updated_at
            FROM annotations a
            JOIN questions q ON q.question_id = a.question_id
            JOIN users u ON u.id = a.user_id
            ORDER BY q.seq, u.username
            """
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT q.seq, q.question_id, q.question_type, q.company,
                   u.username,
                   a.clarity, a.esg_relevance, a.answerability, a.answer_correctness,
                   a.evidence_support, a.question_type_fit, a.type_specific_validity,
                   a.notes, a.updated_at
            FROM annotations a
            JOIN questions q ON q.question_id = a.question_id
            JOIN users u ON u.id = a.user_id
            WHERE a.user_id = ?
            ORDER BY q.seq
            """,
            (user_id,),
        ).fetchall()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "seq",
            "question_id",
            "question_type",
            "company",
            "annotator",
            *DIMS,
            "notes",
            "updated_at",
        ]
    )
    for r in rows:
        writer.writerow([r[k] for k in r.keys()])
    return buf.getvalue().encode("utf-8-sig")


@app.get("/api/me/export.csv")
def my_export_csv(user: dict[str, Any] = Depends(require_user)):
    """Annotator exports only their own scores."""
    if user["role"] != "annotator":
        raise HTTPException(status_code=403, detail="Annotator only")
    conn = connect()
    try:
        data = _annotations_csv_bytes(conn, user_id=user["id"])
        fname = f"qg_scores_{user['username']}.csv"
        return StreamingResponse(
            io.BytesIO(data),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )
    finally:
        conn.close()


@app.get("/api/admin/progress")
def admin_progress(_: dict[str, Any] = Depends(require_admin)):
    conn = connect()
    try:
        total = total_questions(conn)
        users = conn.execute(
            """
            SELECT id, username, display_name, role, last_question_index, last_active_at
            FROM users ORDER BY id
            """
        ).fetchall()
        out = []
        for u in users:
            if u["role"] != "annotator":
                continue
            completed = count_completed(conn, u["id"])
            out.append(
                {
                    "username": u["username"],
                    "display_name": u["display_name"],
                    "completed": completed,
                    "total": total,
                    "pct": round(100.0 * completed / total, 2) if total else 0.0,
                    "last_question_index": u["last_question_index"],
                    "last_active_at": u["last_active_at"],
                }
            )
        return {"total_questions": total, "annotators": out, "dims": DIMS}
    finally:
        conn.close()


@app.post("/api/admin/users")
def admin_create_user(body: CreateUserBody, _: dict[str, Any] = Depends(require_admin)):
    """Admin creates an annotator account (no public self-registration)."""
    username = body.username.strip()
    password = body.password
    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")
    if len(username) < 2 or len(password) < 4:
        raise HTTPException(status_code=400, detail="username min 2 chars, password min 4")
    role = body.role if body.role in ("annotator", "admin") else "annotator"
    display = (body.display_name or username).strip()
    conn = connect()
    try:
        exists = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if exists:
            raise HTTPException(status_code=409, detail="username already exists")
        conn.execute(
            """
            INSERT INTO users (username, password_hash, role, display_name)
            VALUES (?, ?, ?, ?)
            """,
            (username, hash_password(password), role, display),
        )
        conn.commit()
        export_annotations_json(conn)
        return {"ok": True, "username": username, "role": role, "display_name": display}
    finally:
        conn.close()


@app.get("/api/admin/export.csv")
def admin_export_csv(_: dict[str, Any] = Depends(require_admin)):
    conn = connect()
    try:
        data = _annotations_csv_bytes(conn, user_id=None)
        return StreamingResponse(
            io.BytesIO(data),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=qg_annotations_all.csv"},
        )
    finally:
        conn.close()


@app.get("/api/health")
def health():
    """Public check: confirms service + which usernames exist (no passwords)."""
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT username, role FROM users ORDER BY id"
        ).fetchall()
        return {
            "ok": True,
            "users": [{"username": r["username"], "role": r["role"]} for r in rows],
            "questions": total_questions(conn),
        }
    finally:
        conn.close()


@app.get("/")
def root():
    return HTMLResponse('<meta http-equiv="refresh" content="0; url=/login.html">')


@app.get("/login.html")
def login_page():
    return FileResponse(STATIC / "login.html")


@app.get("/register.html")
def register_page():
    return FileResponse(STATIC / "register.html")


@app.get("/annotate.html")
def annotate_page():
    return FileResponse(STATIC / "annotate.html")


@app.get("/admin.html")
def admin_page():
    return FileResponse(STATIC / "admin.html")
