"""Supabase-backed persistence for resume history, parsed resumes, and jobs.

Usage is optional: if ``SUPABASE_URL`` / ``SUPABASE_KEY`` are not set, the
backend simply skips all persistence (stateless mode). This module's
``DatabaseService`` is only instantiated by the caller when a Supabase
client is available.
"""

import hashlib

from supabase import Client

from .logger import logger
from .models import Resume


class DatabaseService:
    def __init__(self, supabase: Client):
        self.supabase = supabase

    @staticmethod
    def _hash_description(description: str) -> str:
        return hashlib.md5(description.strip().encode()).hexdigest()

    @staticmethod
    def _hash_file_content(content: bytes) -> str:
        return hashlib.md5(content).hexdigest()

    async def get_or_create_job(self, user_id: str, description: str, title: str = None, company: str = None) -> str:
        """Get existing job or create new one. Updates title/company if changed. Returns job_id."""
        desc_hash = self._hash_description(description)

        result = self.supabase.table("jobs").select("id, title, company").eq("description_hash", desc_hash).execute()

        if result.data:
            job = result.data[0]
            job_id = job["id"]
            updates = {}
            if title and title != job.get("title"):
                updates["title"] = title
            if company and company != job.get("company"):
                updates["company"] = company
            if updates:
                self.supabase.table("jobs").update(updates).eq("id", job_id).execute()
            return job_id

        job_data = {
            "user_id": user_id,
            "title": title,
            "company": company,
            "description": description,
            "description_hash": desc_hash,
        }
        result = self.supabase.table("jobs").insert(job_data).execute()
        return result.data[0]["id"]

    async def save_resume_generation(
        self,
        user_id: str,
        job_id: str,
        filename: str,
        resume: Resume,
        ats_score: float,
        resume_template: str = "resume_template_7.html",
        parsed_resume_id: str = None,
    ) -> str:
        """Save resume generation. Returns generation_id."""
        data = {
            "user_id": user_id,
            "job_id": job_id,
            "original_filename": filename,
            "resume_json": resume.model_dump(),
            "ats_score": ats_score,
            "resume_template": resume_template,
            "parsed_resume_id": parsed_resume_id,
        }
        result = self.supabase.table("resume_generations").insert(data).execute()
        return result.data[0]["id"]

    async def get_resume_generation(self, generation_id: str) -> dict | None:
        result = self.supabase.table("resume_generations").select("*").eq("id", generation_id).execute()
        return result.data[0] if result.data else None

    async def get_user_resume_history_filtered(
        self,
        user_id: str,
        page: int = 1,
        limit: int = 10,
        company: str = None,
        min_score: float = None,
        max_score: float = None,
        start_date: str = None,
        end_date: str = None,
    ) -> dict:
        """Get filtered and paginated resume history."""
        offset = (page - 1) * limit

        query = (
            self.supabase.table("resume_generations")
            .select("*, jobs!inner(title, company, description)", count="exact")
            .eq("user_id", user_id)
        )

        if company:
            query = query.eq("jobs.company", company)
        if min_score is not None:
            query = query.gte("ats_score", min_score)
        if max_score is not None:
            query = query.lte("ats_score", max_score)
        if start_date:
            query = query.gte("created_at", start_date)
        if end_date:
            query = query.lte("created_at", end_date)

        query = query.order("created_at", desc=True).range(offset, offset + limit - 1)
        result = query.execute()

        return {
            "history": result.data,
            "total": result.count,
            "page": page,
            "limit": limit,
            "total_pages": (result.count + limit - 1) // limit if result.count else 0,
        }

    async def get_parsed_resume(self, user_id: str, file_hash: str) -> dict | None:
        result = (
            self.supabase.table("parsed_resumes")
            .select("*")
            .eq("user_id", user_id)
            .eq("file_hash", file_hash)
            .execute()
        )
        return result.data[0] if result.data else None

    async def save_parsed_resume(self, user_id: str, filename: str, file_content: bytes, parsed_resume: dict) -> str:
        """Save parsed resume. Returns parsed_resume_id."""
        file_hash = self._hash_file_content(file_content)

        existing = await self.get_parsed_resume(user_id, file_hash)
        if existing:
            data = {
                "filename": filename,
                "parsed_resume": parsed_resume,
                "updated_at": "now()",
            }
            self.supabase.table("parsed_resumes").update(data).eq("id", existing["id"]).execute()
            return existing["id"]

        data = {
            "user_id": user_id,
            "filename": filename,
            "file_hash": file_hash,
            "parsed_resume": parsed_resume,
        }
        result = self.supabase.table("parsed_resumes").insert(data).execute()
        return result.data[0]["id"]

    async def get_user_parsed_resumes(self, user_id: str) -> list[dict]:
        result = (
            self.supabase.table("parsed_resumes")
            .select("id, filename, parsed_resume, created_at, updated_at")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .execute()
        )
        return result.data

    async def get_user_parsed_resume_by_id(self, resume_id: str) -> list[dict] | None:
        result = (
            self.supabase.table("parsed_resumes").select("id, filename, parsed_resume").eq("id", resume_id).execute()
        )
        return result.data

    async def delete_parsed_resume(self, user_id: str, resume_id: str) -> bool:
        """Delete a parsed resume and its associated generations."""
        result = self.supabase.table("parsed_resumes").select("id").eq("id", resume_id).eq("user_id", user_id).execute()
        if not result.data:
            return False

        self.supabase.table("resume_generations").delete().eq("parsed_resume_id", resume_id).execute()
        self.supabase.table("parsed_resumes").delete().eq("id", resume_id).execute()
        return True

    async def log_request(
        self,
        user_id: str | None,
        user_email: str | None,
        user_name: str | None,
        endpoint: str,
        status: int,
        safe_metadata: dict,
    ) -> None:
        """Fire-and-forget request audit log. ``safe_metadata`` must already be
        redacted of sensitive headers — this layer does not inspect it."""
        try:
            self.supabase.table("requests").insert(
                {
                    "user_id": user_id,
                    "user_email": user_email,
                    "user_name": user_name,
                    "endpoint": endpoint,
                    "status": status,
                    "metadata": safe_metadata,
                }
            ).execute()
        except Exception as e:
            logger.error("Failed to log request", extra={"error_message": str(e)})

    async def save_feedback(
        self,
        user_id: str | None,
        category: str,
        message: str,
        user_email: str | None,
        user_name: str | None,
    ) -> None:
        try:
            self.supabase.table("feedback").insert(
                {
                    "user_id": user_id,
                    "category": category,
                    "message": message,
                    "user_email": user_email,
                    "user_name": user_name,
                }
            ).execute()
        except Exception as e:
            logger.error("Failed to store feedback", extra={"error": str(e)})
