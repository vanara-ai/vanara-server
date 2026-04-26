"""
Smoke test: Resume extraction with provider strict enforcement.

Resume schema has no Dict fields, but has EmailStr which emits
{format: "email"} in the JSON schema. This tests whether Groq strict
mode tolerates it.

Run:
    cd evidence/resumeai-main
    export GROQ_API_KEY=<key>
    .venv/bin/python -m scripts.smoke_test_resume_strict
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from app.models import Resume


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
PARSE_MODEL = "openai/gpt-oss-120b"


SAMPLE_RESUME_TEXT = """Jane Doe
jane.doe@example.com | 555-0100 | San Francisco, CA
LinkedIn: linkedin.com/in/janedoe

PROFESSIONAL SUMMARY
Data engineer with 4 years of experience building production ETL pipelines
and cloud data warehouses. Expertise in Python, SQL, Airflow, and AWS.

SKILLS
Languages: Python, SQL, Bash
Cloud: AWS (S3, Redshift, Lambda, Glue)
Tools: Apache Airflow, dbt, Docker, Git
Databases: PostgreSQL, Redshift, Snowflake

EXPERIENCE
Senior Data Engineer | Acme Corp | San Francisco, CA | 2023-01 to Present
- Led migration of legacy ETL jobs from cron to Airflow, reducing failures by 60%
- Built dimensional models in Snowflake using dbt, serving 40+ analysts
- Mentored 2 junior engineers; established code review and testing standards

Data Engineer | StartupXYZ | Remote | 2021-06 to 2022-12
- Built ETL pipelines in Python processing 500M rows/day
- Designed Redshift schemas for BI dashboards
- Collaborated with ML team to deploy feature pipelines

EDUCATION
Bachelor of Science in Computer Science
University of California, Berkeley | 2017-09 to 2021-05

CERTIFICATIONS
AWS Certified Data Analytics - Specialty (2022)

PROJECTS
Real-time Fraud Detection Pipeline
Built a Kafka + Spark Streaming pipeline detecting fraudulent transactions
in under 100ms. Deployed on AWS EKS with auto-scaling.

Personal Finance Tracker (Open Source)
Full-stack Flask + React app with 2k+ GitHub stars.
"""


PROMPT_TEMPLATE = """You are a resume parser. Extract structured information
from the resume text below.

- summary: professional summary / objective
- contact_info: full_name, email, phone, location, linkedin
- skills: list of categories, each with category name and skills list
- certifications: name + optional date
- professional_experience: list of roles with title, company, location, dates, responsibilities
- education: list with institution, degree, location, dates
- projects: list with title and description

For responsibilities, extract only true accomplishments and action items.
Skip tech stack / environment / tools lines.

Resume:
{resume_text}
"""


def main():
    groq_api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not groq_api_key:
        print("Missing GROQ_API_KEY env var")
        sys.exit(1)

    model = os.environ.get("SMOKE_TEST_MODEL", PARSE_MODEL)
    bar = "=" * 70
    print(f"base_url = {GROQ_BASE_URL}")
    print(f"model    = {model}")
    print(f"Resume: {len(SAMPLE_RESUME_TEXT)} chars\n")
    print(bar)
    print("MODE: strict json_schema on Resume")
    print(bar)

    llm = ChatOpenAI(
        base_url=GROQ_BASE_URL,
        model=model,
        api_key=groq_api_key,
        temperature=0,
    )
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["resume_text"],
    )
    structured_llm = llm.with_structured_output(Resume, method="json_schema")
    chain = prompt | structured_llm

    t0 = time.time()
    try:
        result = chain.invoke({"resume_text": SAMPLE_RESUME_TEXT})
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAIL after {elapsed:.1f}s  {type(e).__name__}: {e}")
        sys.exit(1)

    elapsed = time.time() - t0
    is_resume = isinstance(result, Resume)

    print(f"  elapsed            {elapsed:.1f}s")
    print(f"  isinstance(Resume) {is_resume}")
    print(f"  summary (first80)  {result.summary[:80]!r}")
    print(f"  contact.name       {result.contact_info.full_name!r}")
    print(f"  contact.email      {result.contact_info.email!r}")
    print(f"  contact.phone      {result.contact_info.phone!r}")
    print(f"  skills             {len(result.skills)} categories")
    for cat in result.skills[:3]:
        print(f"                     {cat.category}: {cat.skills[:5]}")
    print(f"  experience         {len(result.professional_experience)} roles")
    for role in result.professional_experience:
        print(f"                     {role.title} @ {role.company} ({role.start_date} - {role.end_date})")
        print(f"                       {len(role.responsibilities)} responsibilities")
    print(f"  education          {len(result.education)} items")
    for edu in result.education:
        print(f"                     {edu.degree} @ {edu.institution}")
    print(f"  certifications     {len(result.certifications)} items")
    print(f"  projects           {len(result.projects)} items")

    # Assertions
    all_ok = (
        is_resume
        and result.contact_info.full_name
        and "@" in result.contact_info.email
        and len(result.skills) >= 2
        and len(result.professional_experience) >= 2
        and len(result.education) >= 1
        and len(result.projects) >= 1
    )

    print(f"\n{bar}")
    print(f"  [{'PASS' if all_ok else 'FAIL'}]  strict json_schema on Resume  ({elapsed:.1f}s)")
    print(bar)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
