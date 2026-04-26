from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from .logger import logger
from .models import Resume
from .pdf_utils import extract_pdf_text_with_pypdf2

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
PARSE_MODEL = "openai/gpt-oss-20b"


class ResumeParser:
    """Parses PDF resumes into structured data using a Groq-hosted LLM.

    BYOK: the ``groq_api_key`` is required and is used only for the lifetime
    of this instance. It is never logged, cached to disk, or persisted.
    """

    def __init__(self, groq_api_key: str):
        if not groq_api_key:
            raise ValueError("groq_api_key is required")
        self.parse_llm = ChatOpenAI(
            base_url=GROQ_BASE_URL,
            model=PARSE_MODEL,
            api_key=groq_api_key,
            max_completion_tokens=32768,
        )

    async def parse_resume(self, resume_path: str) -> Resume:
        try:
            resume_text = extract_pdf_text_with_pypdf2(resume_path)

            logger.info(
                "PDF text extracted",
                extra={
                    "char_count": len(resume_text),
                    "preview": resume_text[:200].replace("\n", " ") if resume_text else "<empty>",
                },
            )
            if not resume_text or len(resume_text.strip()) < 50:
                raise ValueError(
                    f"Could not extract text from PDF (got {len(resume_text)} chars). "
                    "The PDF may be scanned/image-only, password-protected, or corrupted. "
                    "Try exporting your resume from Word/Google Docs as a text-based PDF."
                )

            parser = JsonOutputParser(pydantic_object=Resume)
            prompt = PromptTemplate(
                template="""You are an expert-level resume parser with a meticulous eye for detail. Your task is to convert the following resume text into structured data using the ResumeData format defined in the Pydantic models.

                **MANDATORY GUIDELINES:**
                - You MUST extract **EVERY SINGLE PIECE OF INFORMATION** from the original resume text.
                - **DO NOT OMIT, SUMMARIZE, CONDENSE, OR REPHRASE** any content — especially in the **Professional Experience** and **Summary** sections.
                - The extracted data must match the original wording, structure, and detail level exactly as present in the resume.
                - Ensure all bullet points under job roles are preserved **as-is**, including all quantifiable metrics, technologies, tools, responsibilities, and achievements.
                - Include a space before any number. Example: "over 3 years of experience" ✅, not "over3 years of experience" ❌.

                **LOCATION PARSING RULES:**
                - For professional experience entries, carefully separate company names from locations
                - When you see text like "Wright State University, Dayton, Ohio", parse as:
                  * Company: "Wright State University"
                  * Location: "Dayton, Ohio"
                - When you see text like "ABC Company, New York, NY", parse as:
                  * Company: "ABC Company"
                  * Location: "New York, NY"
                - The location typically appears after the company name, separated by a comma
                - Common location formats: "City, State", "City, State Abbreviation", "City, Country"

                If any field is genuinely missing from the resume, set it to `None`, but **never skip or infer** details not explicitly written in the text.

                Resume Text: {resume_text}

                {format_instructions}

                Return ONLY the JSON, no explanations.""",
                input_variables=["resume_text"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )

            chain = prompt | self.parse_llm | parser
            parsed_data = chain.invoke({"resume_text": resume_text})
            logger.info(
                "Resume parsed successfully",
                extra={"resume_sections": list(parsed_data.keys())},
            )

            return parsed_data

        except Exception as e:
            logger.error("Resume parsing failed", extra={"error": str(e)})
            raise
