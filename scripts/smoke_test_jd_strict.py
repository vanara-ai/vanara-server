"""
Smoke test: JD extraction with provider strict enforcement.

Tests `llm.with_structured_output(JD, method="json_schema")` — the pattern that
uses the provider's native structured-output / JSON schema enforcement.

JD schema is safe for strict mode: no Dict[str, X] fields, so Groq's
`additionalProperties: false` requirement is satisfied automatically.

Run:
    cd evidence/resumeai-main
    export GROQ_API_KEY=<your-key>
    python -m scripts.smoke_test_jd_strict
"""
import os
import sys
import time
import json
from pathlib import Path

# Make `app.*` importable when invoked as `python -m scripts.smoke_test_jd_strict`
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from app.models import JD


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_jd.txt"

# Mirror production configuration from app/resume_optimizer.py
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
PARSE_MODEL = "openai/gpt-oss-120b"


PROMPT_TEMPLATE = """You are an expert job description analyzer.
Extract ALL explicit and implicit requirements from the job description below.

- target_job_title: main job title / position
- required_skills: skills marked required, must-have, essential, or appearing in job title / core responsibilities
- preferred_skills: skills marked preferred, nice-to-have, bonus, or plus
- experience_requirements: years, domains, conditional paths, alternative paths
- education_requirements: degrees, alternative experience, certifications
- key_responsibilities: preserve original wording for ATS relevance
- company_info: industry inferred from context

Return empty lists/strings for missing information.

Job Description:
{jd}
"""


def run_mode(label: str, model: str, groq_api_key: str, method_kwargs: dict, jd_text: str):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"MODE: {label}")
    print(f"  model={model}  method_kwargs={method_kwargs}")
    print(bar)

    # Construct exactly as production does in resume_optimizer.py
    llm = ChatOpenAI(
        base_url=GROQ_BASE_URL,
        model=model,
        api_key=groq_api_key,
        temperature=0,
    )

    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["jd"])
    structured_llm = llm.with_structured_output(JD, **method_kwargs)
    chain = prompt | structured_llm

    t0 = time.time()
    try:
        result = chain.invoke({"jd": jd_text})
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAIL after {elapsed:.1f}s  {type(e).__name__}: {e}")
        return {"label": label, "ok": False, "elapsed": elapsed, "error": str(e)}

    elapsed = time.time() - t0

    # Assertions
    is_jd = isinstance(result, JD)
    title_ok = bool(result.target_job_title and result.target_job_title.strip())
    tech_skills = result.required_skills.technical_skills
    responsibilities = result.key_responsibilities
    tech_ok = len(tech_skills) >= 3
    resp_ok = len(responsibilities) >= 3

    print(f"  elapsed            {elapsed:.1f}s")
    print(f"  isinstance(JD)     {is_jd}")
    print(f"  target_job_title   {result.target_job_title!r}  [{'OK' if title_ok else 'EMPTY'}]")
    print(f"  required tech      {len(tech_skills)} items  [{'OK' if tech_ok else 'TOO FEW'}]")
    print(f"                     {tech_skills[:5]}")
    print(f"  responsibilities   {len(responsibilities)} items  [{'OK' if resp_ok else 'TOO FEW'}]")
    print(f"  preferred tech     {len(result.preferred_skills.technical_skills)} items")
    print(f"  exp minimum_years  {result.experience_requirements.minimum_years!r}")
    print(f"  degrees            {result.education_requirements.degree_requirements}")
    print(f"  industry           {result.company_info.industry!r}")

    all_ok = is_jd and title_ok and tech_ok and resp_ok
    return {
        "label": label,
        "ok": all_ok,
        "elapsed": elapsed,
        "result": result.model_dump(),
    }


def main():
    if not FIXTURE_PATH.exists():
        print(f"Missing fixture: {FIXTURE_PATH}")
        sys.exit(1)
    jd_text = FIXTURE_PATH.read_text()

    groq_api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not groq_api_key:
        print("Missing GROQ_API_KEY (or OPENAI_API_KEY) env var")
        sys.exit(1)

    model = os.environ.get("SMOKE_TEST_MODEL", PARSE_MODEL)
    print(f"base_url  = {GROQ_BASE_URL}")
    print(f"model     = {model}")
    print(f"JD fixture: {len(jd_text)} chars")

    # The mode we care about: provider-native strict JSON schema
    summary = []
    summary.append(run_mode(
        label="strict json_schema (provider-native)",
        model=model,
        groq_api_key=groq_api_key,
        method_kwargs={"method": "json_schema"},
        jd_text=jd_text,
    ))

    bar = "=" * 70
    print(f"\n{bar}")
    print("SUMMARY")
    print(bar)
    for s in summary:
        status = "PASS" if s["ok"] else "FAIL"
        print(f"  [{status}]  {s['label']}  ({s['elapsed']:.1f}s)")

    sys.exit(0 if all(s["ok"] for s in summary) else 1)


if __name__ == "__main__":
    main()
