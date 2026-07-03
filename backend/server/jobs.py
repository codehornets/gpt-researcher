"""Async jobs API for server-to-server callers (outreach360 Base44 functions).

The upstream POST /report/ endpoint either blocks for the full research run
(minutes) or backgrounds it with no status/markdown retrieval and no failure
tracking. Callers with short request budgets (Base44 Deno functions cap out
around 120s) need: POST /jobs -> {job_id} immediately, then poll
GET /jobs/{job_id} -> {status, report}.

Jobs live in an in-memory store — runs are minutes-long and a restart simply
fails the caller's row over to a retry, so no database is warranted.

Auth: if the GPT_RESEARCHER_TOKEN env var is set, both endpoints require
`Authorization: Bearer <token>`.
"""

import asyncio
import logging
import os
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from server.websocket_manager import run_agent
from gpt_researcher.utils.enum import Tone

logger = logging.getLogger(__name__)

router = APIRouter()

_jobs: Dict[str, Dict[str, Any]] = {}


class JobRequest(BaseModel):
    query: str
    report_type: str = "research_report"


def _check_auth(authorization: Optional[str]) -> None:
    token = os.getenv("GPT_RESEARCHER_TOKEN")
    if not token:
        return
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Unauthorized")


async def _run_job(job_id: str, query: str, report_type: str) -> None:
    try:
        report_information = await run_agent(
            task=query,
            report_type=report_type,
            report_source="web",
            source_urls=[],
            document_urls=[],
            tone=Tone.Objective,
            websocket=None,
            stream_output=None,
            headers=None,
            query_domains=[],
            config_path="",
            return_researcher=True,
        )
        report = (
            report_information[0]
            if isinstance(report_information, (list, tuple))
            else report_information
        )
        _jobs[job_id] = {"status": "completed", "report": report, "error": None}
        logger.info("jobs: research %s completed (%d chars)", job_id, len(report or ""))
    except Exception as e:  # noqa: BLE001 — job must record any failure
        logger.exception("jobs: research %s failed", job_id)
        _jobs[job_id] = {"status": "failed", "report": None, "error": str(e)}


@router.post("/jobs", status_code=202)
async def create_job(req: JobRequest, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Missing query")
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "running", "report": None, "error": None}
    asyncio.create_task(_run_job(job_id, req.query.strip(), req.report_type))
    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return {"status": job["status"], "report": job.get("report"), "error": job.get("error")}
