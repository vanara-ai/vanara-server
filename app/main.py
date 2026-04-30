"""FastAPI entrypoint for Vanara.ai.

BYOK (Bring Your Own Key) architecture:
- Users send their Groq API key via the ``X-Groq-Key`` request header.
- The key is used only for that request's LLM calls — never logged, never
  persisted, never written to disk.
- Zero-knowledge: the server stores no credentials.

Persistence is optional. If ``SUPABASE_URL`` and ``SUPABASE_KEY`` env vars
are set, the backend adds resume history, parsed-resume reuse, and
feedback collection. Without them, the backend runs stateless — the core
optimize flow still works, but history endpoints return 501.
"""

import contextlib
import os
import tempfile
from datetime import datetime

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .constants import PDF_SUFFIX
from .deps import get_groq_key
from .email_service import EmailService
from .env import setup_environment
from .errors import classify_error
from .logger import logger
from .logging_utils import safe_headers
from .models import FeedbackRequest, GeneratePDFRequest, Resume
from .optimized_pipeline import process_resume_and_jobdesc
from .pdf_utils import generate_pdf_from_resume
from .resume_optimizer import ResumeOptimizer
from .resume_parser import ResumeParser

setup_environment()

app = FastAPI(
    title="Vanara.ai — ResumeAI",
    description=(
        "Open-source, BYOK resume optimization. Your Groq API key is sent "
        "from the client on every request and is never persisted server-side."
    ),
)


def _parse_origins(raw: str | None) -> list[str]:
    if not raw:
        return ["http://localhost:3000"]
    return [o.strip() for o in raw.split(",") if o.strip()]


CORS_ORIGINS = _parse_origins(os.getenv("CORS_ORIGINS"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Optional Supabase persistence --------------------------------------------
# If SUPABASE_URL + SUPABASE_KEY are both set, enable DB-backed features
# (history, parsed-resume reuse, feedback persistence, request audit log).
# Otherwise run stateless.

db_service = None
_supabase_url = os.getenv("SUPABASE_URL")
_supabase_key = os.getenv("SUPABASE_KEY")

if _supabase_url and _supabase_key:
    try:
        from supabase import create_client

        from .database import DatabaseService

        _supabase = create_client(_supabase_url, _supabase_key)
        db_service = DatabaseService(_supabase)
        logger.info("Supabase persistence enabled")
    except Exception as e:
        logger.error(
            "Failed to initialize Supabase; running stateless",
            extra={"error": str(e)},
        )
        db_service = None
else:
    logger.info("Supabase env vars not set — running stateless (no history)")


def require_db(reason: str = "This endpoint requires Supabase to be configured."):
    if db_service is None:
        raise HTTPException(status_code=501, detail=reason)
    return db_service


email_service = EmailService()


# --- Request audit logging middleware -----------------------------------------
# Only runs if Supabase is configured. Headers are redacted via allow-list
# (see logging_utils.safe_headers) so API keys are never stored.


@app.middleware("http")
async def log_request(request: Request, call_next):
    response = await call_next(request)

    if request.method == "OPTIONS" or request.url.path in ("/analytics/requests", "/health"):
        return response

    if db_service is None:
        return response

    try:
        await db_service.log_request(
            user_id=request.headers.get("X-User-ID"),
            user_email=request.headers.get("X-User-Email"),
            user_name=request.headers.get("X-User-Name"),
            endpoint=request.url.path,
            status=response.status_code,
            safe_metadata={
                "method": request.method,
                "headers": safe_headers(request.headers),
                "query_params": dict(request.query_params),
            },
        )
    except Exception as e:
        logger.error("Request log failed", extra={"error": str(e)})

    return response


# --- Routes -------------------------------------------------------------------


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "persistence": "enabled" if db_service else "stateless",
    }


@app.post("/optimize_resume/")
async def optimize_resume(
    resume_pdf: UploadFile = File(...),
    jobdesc_text: str = Form(...),
    job_title: str = Form(None),
    company: str = Form(None),
    resume_template: str = Form("resume_template_7.html"),
    request: Request = None,
    groq_key: str = Depends(get_groq_key),
):
    try:
        if not resume_pdf.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")
        if not resume_pdf.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        if not jobdesc_text or not jobdesc_text.strip():
            raise HTTPException(status_code=400, detail="Job description is required")

        file_content = await resume_pdf.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        logger.info(
            "Starting resume optimization",
            extra={
                "resume_name": resume_pdf.filename,
                "content_type": resume_pdf.content_type,
                "file_size": len(file_content),
                "job_description_length": len(jobdesc_text),
            },
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=PDF_SUFFIX) as resume_tmp:
            resume_tmp.write(file_content)
            resume_path = resume_tmp.name

        optimizer = ResumeOptimizer(groq_api_key=groq_key)
        result = await process_resume_and_jobdesc(optimizer, resume_path, jobdesc_text, resume_template)

        if not result["pdf_download_url"]:
            logger.error(
                "Failed to process resume",
                extra={"error_message": result["result_text"]},
            )
            user_err = classify_error(result["result_text"])
            raise HTTPException(status_code=user_err.status_code, detail=user_err.detail)

        logger.info(
            "Resume optimization completed",
            extra={
                "output_path": result["pdf_download_url"],
                "result_length": len(result["result_text"]),
            },
        )

        # Optional DB persistence (history + auto-save parsed resume)
        user_id = request.headers.get("X-User-ID") if request else None
        if db_service and user_id and "diff" in result and "optimized" in result["diff"]:
            try:
                optimized_resume = Resume(**result["diff"]["optimized"])
                ats_score = result.get("final_score", 0.0)
                job_id = await db_service.get_or_create_job(user_id, jobdesc_text, job_title, company)

                # Auto-save parsed resume to Smart Library
                parsed_resume_id = None
                if "original" in result["diff"]:
                    try:
                        parsed_resume_id = await db_service.save_parsed_resume(
                            user_id, resume_pdf.filename, file_content, result["diff"]["original"]
                        )
                    except Exception as e:
                        logger.error("Failed to auto-save parsed resume", extra={"error": str(e)})

                generation_id = await db_service.save_resume_generation(
                    user_id, job_id, resume_pdf.filename, optimized_resume, ats_score, resume_template, parsed_resume_id
                )
                result["generation_id"] = generation_id
            except Exception as e:
                logger.error("Failed to save to database", extra={"error": str(e)})

        return result
    except HTTPException:
        raise
    except Exception as e:
        user_err = classify_error(e)
        logger.exception(
            "Unexpected error during resume optimization",
            extra={
                "error_message": str(e),
                "resume_name": resume_pdf.filename if resume_pdf else None,
                "error_ref": user_err.request_id,
            },
        )
        raise HTTPException(status_code=user_err.status_code, detail=user_err.detail) from None


@app.get("/download/{filename}")
async def download_pdf(filename: str):
    try:
        # Block path traversal — only basenames are valid outputs.
        if os.path.basename(filename) != filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        file_path = os.path.join(tempfile.gettempdir(), filename)
        if not os.path.exists(file_path):
            logger.error("File not found", extra={"requested_file": filename})
            raise HTTPException(status_code=404, detail="File not found")

        logger.info("File download requested", extra={"requested_file": filename})
        return FileResponse(file_path, media_type="application/pdf", filename=filename)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "Error during file download",
            extra={"error_message": str(e), "requested_file": filename},
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@app.get("/resume-history")
async def get_resume_history(
    request: Request,
    page: int = 1,
    limit: int = 10,
    company: str = None,
    min_score: float = None,
    max_score: float = None,
    start_date: str = None,
    end_date: str = None,
):
    db = require_db("Resume history requires Supabase persistence.")
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        return await db.get_user_resume_history_filtered(
            user_id, page, limit, company, min_score, max_score, start_date, end_date
        )
    except Exception as e:
        user_err = classify_error(e)
        logger.exception("Failed to fetch resume history", extra={"error_ref": user_err.request_id})
        raise HTTPException(status_code=user_err.status_code, detail=user_err.detail) from None


@app.post("/generate-pdf")
async def generate_pdf_from_history(pdf_request: GeneratePDFRequest, request: Request):
    db = require_db("PDF regeneration from history requires Supabase persistence.")
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        generation = await db.get_resume_generation(pdf_request.resume_generation_id)
        if not generation or generation["user_id"] != user_id:
            raise HTTPException(status_code=404, detail="Resume generation not found")

        resume = Resume(**generation["resume_json"])
        template_to_use = pdf_request.resume_template or generation.get("resume_template", "resume_template_7.html")
        pdf_path = generate_pdf_from_resume(resume, generation["original_filename"], template_to_use)
        filename = os.path.basename(pdf_path)
        return {"pdf_download_url": f"/download/{filename}"}
    except HTTPException:
        raise
    except Exception as e:
        user_err = classify_error(e)
        logger.exception("Failed to generate PDF", extra={"error_ref": user_err.request_id})
        raise HTTPException(status_code=user_err.status_code, detail=user_err.detail) from None


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest, request: Request):
    """Submit user feedback, bug reports, or feature requests."""
    try:
        user_id = request.headers.get("X-User-ID")
        user_email = feedback.user_email or request.headers.get("X-User-Email")
        user_name = feedback.user_name or request.headers.get("X-User-Name")

        logger.info(
            "Feedback received",
            extra={"category": feedback.category, "user_id": user_id, "user_email": user_email},
        )

        if db_service:
            await db_service.save_feedback(
                user_id=user_id,
                category=feedback.category,
                message=feedback.message,
                user_email=user_email,
                user_name=user_name,
            )

        result = await email_service.send_feedback_email(
            category=feedback.category,
            message=feedback.message,
            user_email=user_email,
            user_name=user_name,
        )

        if not result["success"]:
            logger.warning(
                "Email notification failed but feedback was recorded",
                extra={"error": result["message"]},
            )

        return {"success": True, "message": "Thank you for your feedback!"}
    except Exception as e:
        user_err = classify_error(e)
        logger.exception("Failed to process feedback", extra={"error_ref": user_err.request_id})
        raise HTTPException(status_code=user_err.status_code, detail=user_err.detail) from None


@app.post("/parse-resume/")
async def parse_resume_endpoint(
    resume_pdf: UploadFile = File(...),
    request: Request = None,
    groq_key: str = Depends(get_groq_key),
):
    """Parse a resume and (if persistence enabled) cache it for reuse."""
    try:
        if not resume_pdf.filename or not resume_pdf.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        file_content = await resume_pdf.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        user_id = request.headers.get("X-User-ID") if request else None

        # Fast path: if we have persistence and auth, reuse cached parse
        if db_service and user_id:
            file_hash = db_service._hash_file_content(file_content)
            existing = await db_service.get_parsed_resume(user_id, file_hash)
            if existing:
                return {
                    "parsed_resume_id": existing["id"],
                    "filename": existing["filename"],
                    "status": "already_parsed",
                    "created_at": existing["created_at"],
                    "updated_at": existing["updated_at"],
                }

        parser = ResumeParser(groq_api_key=groq_key)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name

        try:
            parsed_resume = await parser.parse_resume(tmp_path)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

        if db_service and user_id:
            parsed_resume_id = await db_service.save_parsed_resume(
                user_id, resume_pdf.filename, file_content, parsed_resume
            )
            return {
                "parsed_resume_id": parsed_resume_id,
                "filename": resume_pdf.filename,
                "status": "parsed",
                "resume_data": parsed_resume,
            }

        # Stateless response
        return {
            "parsed_resume_id": None,
            "filename": resume_pdf.filename,
            "status": "parsed",
            "resume_data": parsed_resume,
        }

    except HTTPException:
        raise
    except Exception as e:
        user_err = classify_error(e)
        logger.exception("Error parsing resume", extra={"error_ref": user_err.request_id})
        raise HTTPException(status_code=user_err.status_code, detail=user_err.detail) from None


@app.get("/parsed-resumes/")
async def get_parsed_resumes(request: Request):
    db = require_db("Parsed-resume history requires Supabase persistence.")
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        resumes = await db.get_user_parsed_resumes(user_id)
        return {"parsed_resumes": resumes}
    except Exception as e:
        user_err = classify_error(e)
        logger.exception("Failed to fetch parsed resumes", extra={"error_ref": user_err.request_id})
        raise HTTPException(status_code=user_err.status_code, detail=user_err.detail) from None


@app.delete("/parsed-resumes/{resume_id}")
async def delete_parsed_resume(resume_id: str, request: Request):
    db = require_db("Parsed-resume deletion requires Supabase persistence.")
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        success = await db.delete_parsed_resume(user_id, resume_id)
        if not success:
            raise HTTPException(status_code=404, detail="Parsed resume not found")
        return {"message": "Parsed resume deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        user_err = classify_error(e)
        logger.exception("Failed to delete parsed resume", extra={"error_ref": user_err.request_id})
        raise HTTPException(status_code=user_err.status_code, detail=user_err.detail) from None


@app.post("/optimize-from-parsed/")
async def optimize_from_parsed(
    parsed_resume_id: str = Form(...),
    jobdesc_text: str = Form(...),
    job_title: str = Form(None),
    company: str = Form(None),
    resume_template: str = Form("resume_template_7.html"),
    request: Request = None,
    groq_key: str = Depends(get_groq_key),
):
    """Optimize a resume using a previously-parsed record."""
    db = require_db("Optimize-from-parsed requires Supabase persistence.")
    user_id = request.headers.get("X-User-ID") if request else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        parsed_resume_data = await db.get_user_parsed_resume_by_id(parsed_resume_id)
        if not parsed_resume_data:
            raise HTTPException(status_code=404, detail="Parsed resume not found")

        parsed_data = parsed_resume_data[0]["parsed_resume"]
        filename = parsed_resume_data[0]["filename"]

        optimizer = ResumeOptimizer(groq_api_key=groq_key)
        result = await optimizer.optimize_from_parsed(parsed_data, jobdesc_text)

        optimized_resume = Resume(**result["optimized_resume"])
        pdf_path = generate_pdf_from_resume(optimized_resume, filename, resume_template)

        job_id = await db.get_or_create_job(user_id, jobdesc_text, job_title, company)
        generation_id = await db.save_resume_generation(
            user_id, job_id, filename, optimized_resume, result["final_score"], resume_template, parsed_resume_id
        )

        return {
            "generation_id": generation_id,
            "initial_score": result["initial_score"],
            "final_score": result["final_score"],
            "pdf_download_url": f"/download/{os.path.basename(pdf_path)}",
            "iterations": result["iterations"],
            "status": "success",
            "summary": result["ats_feedback"].get("summary", ""),
            "strengths": result["ats_feedback"].get("strengths", []),
            "weaknesses": result["ats_feedback"].get("weaknesses", []),
            "diff": {
                "original": result["parsed_resume"],
                "optimized": result["optimized_resume"],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        user_err = classify_error(e)
        logger.exception(
            "Error optimizing from parsed resume",
            extra={"error_ref": user_err.request_id},
        )
        raise HTTPException(status_code=user_err.status_code, detail=user_err.detail) from None


@app.get("/analytics/requests")
async def get_request_analytics(
    user_email: str = None,
    user_name: str = None,
    endpoint: str = None,
    start_timestamp: str = None,
    end_timestamp: str = None,
):
    db = require_db("Analytics requires Supabase persistence.")
    try:
        query = db.supabase.table("requests").select("*", count="exact")
        if user_email:
            query = query.eq("user_email", user_email)
        if user_name:
            query = query.eq("user_name", user_name)
        if endpoint:
            query = query.eq("endpoint", endpoint)
        if start_timestamp:
            query = query.gte("timestamp", start_timestamp)
        if end_timestamp:
            query = query.lte("timestamp", end_timestamp)

        response = query.execute()
        data = response.data if hasattr(response, "data") else []
        count = response.count if hasattr(response, "count") else None
        return {"data": data, "count": count}
    except Exception as e:
        user_err = classify_error(e)
        logger.exception("Failed to fetch analytics", extra={"error_ref": user_err.request_id})
        raise HTTPException(status_code=user_err.status_code, detail=user_err.detail) from None


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
