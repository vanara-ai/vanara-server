"""
Smoke test: ATSScoreOutput extraction with provider strict enforcement.

After refactoring models.py (Dict -> List for job_specific_instructions and
project_specific_instructions), ATSScoreOutput should be strict-mode compatible.

Run:
    cd evidence/resumeai-main
    export GROQ_API_KEY=<key>
    .venv/bin/python -m scripts.smoke_test_ats_strict
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from app.models import ATSScoreOutput


FIXTURE_JD = Path(__file__).parent / "fixtures" / "sample_jd.txt"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
SCORE_MODEL = "openai/gpt-oss-120b"


SAMPLE_RESUME_JSON = """{
  "summary": "Data engineer with 4 years building ETL pipelines.",
  "contact_info": {
    "full_name": "Jane Doe",
    "email": "jane@example.com",
    "phone": "555-0100"
  },
  "skills": [
    {"category": "Languages", "skills": ["Python", "SQL"]},
    {"category": "Cloud", "skills": ["AWS"]}
  ],
  "certifications": [],
  "professional_experience": [
    {
      "title": "Data Engineer",
      "company": "Acme Corp",
      "start_date": "2021-06",
      "end_date": "Present",
      "responsibilities": [
        "Built ETL pipelines in Python and Airflow",
        "Designed dimensional models in Redshift"
      ]
    }
  ],
  "education": [
    {"degree": "Bachelor of Science in Computer Science"}
  ],
  "projects": []
}"""


PROMPT_TEMPLATE = """You are an expert ATS analyzer. Score the resume against the job description.

Return:
- score (0-100)
- summary (brief explanation)
- score_breakdown (keywords 0-40, skills 0-25, experience 0-25, education 0-10 — with reasoning for each)
- strengths (list)
- weaknesses (list)
- section_instructions: per-section improvement guidance
  - summary: current_issues, missing_keywords, missing_value_propositions, actionable_steps, target_length, tone_guidance
  - skills: missing_technical_skills, missing_soft_skills, categorization_improvements, actionable_steps
  - professional_experience:
      - job_specific_instructions: LIST of items, one per role in the resume. Each item has:
        job_title, missing_keywords, responsibilities_to_add, responsibilities_to_modify,
        quantification_needed, achievements_to_highlight, action_verbs_to_use
      - overall_improvements, experience_ordering, actionable_steps
  - education: formatting_requirements, actionable_steps
  - projects:
      - project_specific_instructions: LIST of items, one per project. Each item has:
        project_name, alignment_issues, missing_technologies, missing_outcomes, description_improvements
      - projects_to_add, projects_to_emphasize, overall_improvements, actionable_steps

Job Description:
{jd}

Resume:
{resume_json}
"""


def main():
    if not FIXTURE_JD.exists():
        print(f"Missing fixture: {FIXTURE_JD}")
        sys.exit(1)
    jd_text = FIXTURE_JD.read_text()

    groq_api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not groq_api_key:
        print("Missing GROQ_API_KEY env var")
        sys.exit(1)

    model = os.environ.get("SMOKE_TEST_MODEL", SCORE_MODEL)
    bar = "=" * 70
    print(f"base_url = {GROQ_BASE_URL}")
    print(f"model    = {model}")
    print(f"JD: {len(jd_text)} chars, resume: {len(SAMPLE_RESUME_JSON)} chars\n")
    print(bar)
    print("MODE: strict json_schema on ATSScoreOutput")
    print(bar)

    llm = ChatOpenAI(
        base_url=GROQ_BASE_URL,
        model=model,
        api_key=groq_api_key,
        temperature=0,
    )
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["jd", "resume_json"],
    )
    structured_llm = llm.with_structured_output(ATSScoreOutput, method="json_schema")
    chain = prompt | structured_llm

    t0 = time.time()
    try:
        result = chain.invoke({"jd": jd_text, "resume_json": SAMPLE_RESUME_JSON})
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAIL after {elapsed:.1f}s  {type(e).__name__}: {e}")
        sys.exit(1)

    elapsed = time.time() - t0
    is_ats = isinstance(result, ATSScoreOutput)
    sb = result.score_breakdown
    se = result.section_instructions
    pe_jobs = se.professional_experience.job_specific_instructions
    proj_items = se.projects.project_specific_instructions

    print(f"  elapsed            {elapsed:.1f}s")
    print(f"  isinstance(ATS)    {is_ats}")
    print(f"  score              {result.score}")
    print(f"  summary (first80)  {result.summary[:80]!r}")
    print(f"  breakdown          kw={sb.keywords_score} sk={sb.skills_score} exp={sb.experience_score} edu={sb.education_score}")
    print(f"  strengths          {len(result.strengths)} items")
    print(f"  weaknesses         {len(result.weaknesses)} items")
    print(f"  pe.job_specific    {len(pe_jobs)} items (was dict, now list)")
    if pe_jobs:
        print(f"                     first: job_title={pe_jobs[0].job_title!r}")
    print(f"  projects.specific  {len(proj_items)} items (was dict, now list)")
    print(f"  summary.issues     {len(se.summary.current_issues)} items")
    print(f"  skills.missing_tech {len(se.skills.missing_technical_skills)} items")

    all_ok = (
        is_ats
        and result.score > 0
        and len(pe_jobs) >= 1
        and len(result.strengths) >= 1
    )

    print(f"\n{bar}")
    print(f"  [{'PASS' if all_ok else 'FAIL'}]  strict json_schema on ATSScoreOutput  ({elapsed:.1f}s)")
    print(bar)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
