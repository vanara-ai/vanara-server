import json
import os
import tempfile

from deepdiff import DeepDiff

from .constants import TEMPLATE_DIR
from .env import setup_environment
from .logger import logger
from .pdf_utils import (
    _sanitize_pdf_payload,
    extract_pdf_text,
    html_to_pdf,
    render_resume_to_html,
)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, type):
            return obj.__name__
        return super().default(obj)


async def process_resume_and_jobdesc(pipeline, resume_path, jobdesc_text, resume_template):
    """Run the full parse → optimize → render pipeline.

    ``pipeline`` is a ResumeOptimizer instance, already constructed with the
    caller's Groq API key. This function does not read any LLM credentials
    from the environment.
    """
    try:
        setup_environment()

        logger.info(
            "Processing request",
            extra={
                "resume_path": resume_path,
                "job_description_length": len(jobdesc_text),
            },
        )

        logger.info("Extracting text from PDF", extra={"resume_path": resume_path})
        resume_text = extract_pdf_text(resume_path)

        result = await pipeline.optimize(resume_text, jobdesc_text)
        original_resume = result["parsed_resume"]
        improved_resume = result["optimized_resume"]
        initial_score = result["initial_score"]
        ats_score = result["final_score"]
        iterations = result["iterations"]
        ats_feedback = result["ats_feedback"]
        summary = ats_feedback.get("summary", "No summary provided")
        strengths = ats_feedback.get("strengths", ["No strengths identified"])
        weaknesses = ats_feedback.get("weaknesses", ["No weaknesses identified"])

        logger.info("Comparing resumes to track changes")

        diff = DeepDiff(
            original_resume,
            improved_resume,
            ignore_order=True,
            report_repetition=True,
            verbose_level=2,
            exclude_paths=[
                "root['contact_info']['linkedin']",
                "root['certifications'][\\d+]['date']",
            ],
        )

        diff_dict = json.loads(json.dumps(diff.to_dict(), cls=CustomJSONEncoder))
        original_dict = json.loads(json.dumps(original_resume, cls=CustomJSONEncoder))
        improved_dict = json.loads(json.dumps(improved_resume, cls=CustomJSONEncoder))

        resume_diff = {
            "changes": diff_dict,
            "original": original_dict,
            "optimized": improved_dict,
        }

        logger.info(
            "Resume rewrite completed",
            extra={
                "ats_score": ats_score,
                "iterations": iterations,
                "summary": summary,
                "strengths": strengths,
                "weaknesses": weaknesses,
            },
        )

        with (
            tempfile.NamedTemporaryFile(delete=False, suffix=".html") as html_tmp,
            tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as pdf_tmp,
        ):
            html_path = html_tmp.name
            pdf_path = pdf_tmp.name
            # Sanitize Unicode so xhtml2pdf does not emit black boxes for
            # characters missing from the bundled Vera font (e.g. U+2011
            # non-breaking hyphen). base_path lets xhtml2pdf find
            # templates/fonts/*.ttf registered via @font-face.
            sanitized = _sanitize_pdf_payload(improved_resume)
            render_resume_to_html(sanitized, str(TEMPLATE_DIR), resume_template, html_path)
            html_to_pdf(html_path, pdf_path, base_path=str(TEMPLATE_DIR))

        result_text = (
            f"ATS Score: {ats_score}\n\nSummary: {summary}\n\n"
            f"Strengths: {strengths}\n\nWeaknesses: {weaknesses}\n\nIterations: {iterations}"
        )

        return {
            "result_text": result_text,
            "pdf_download_url": f"/download/{os.path.basename(pdf_path)}",
            "diff": resume_diff,
            "initial_score": initial_score,
            "final_score": ats_score,
            "iterations": iterations,
            "summary": summary,
            "strengths": strengths,
            "weaknesses": weaknesses,
        }

    except Exception as e:
        logger.exception(
            "Error in resume processing pipeline",
            extra={
                "error": str(e),
                "resume_path": resume_path,
            },
        )
        return {
            "result_text": f"Error during resume optimization: {str(e)}",
            "pdf_download_url": "",
            "diff": {"changes": {}, "original": {}, "optimized": {}},
            "initial_score": 0,
            "final_score": 0,
            "iterations": 0,
            "summary": "Error occurred",
            "strengths": [],
            "weaknesses": [],
        }
