import concurrent.futures
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from dateutil.parser import parse
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.cache.memory import InMemoryCache
from langgraph.graph import END, StateGraph
from langgraph.pregel import RetryPolicy
from langgraph.types import CachePolicy

from .logger import logger
from .models import JD, ATSScoreOutput, Resume

# Configuration — all models hosted on Groq (single-provider BYOK).
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
PARSE_MODEL = "openai/gpt-oss-120b"
SCORE_MODEL = "openai/gpt-oss-120b"
REWRITE_MODEL = "openai/gpt-oss-120b"
MAX_REWRITE_ATTEMPTS = 5
TARGET_SCORE = 90

cache = InMemoryCache()


class AgentState(TypedDict):
    # Inputs
    raw_resume_text: str
    job_description: str
    jd_structured: dict

    # Processing
    parsed_resume: dict
    current_resume: Annotated[str, lambda old, new: new or old]
    ats_feedback: ATSScoreOutput
    ats_feedback_history: list[ATSScoreOutput]  # Store history of ATS feedback
    iterations: int

    # Tracking
    initial_ats_score: float
    final_ats_score: float
    status: Literal["parsing", "scoring", "rewriting", "complete", "error"]


class ResumeOptimizer:
    """Multi-agent resume optimizer backed by Groq-hosted LLMs.

    BYOK: ``groq_api_key`` is required and is used only for the lifetime
    of this instance. Construct a new optimizer per request; do not share
    instances across users.
    """

    def __init__(self, groq_api_key: str):
        if not groq_api_key:
            raise ValueError("groq_api_key is required")

        # 120b needs generous completion budget; keep default reasoning (medium)
        # which the smoke tests validated as reliable for strict JSON schema.
        _llm_kwargs = {
            "base_url": GROQ_BASE_URL,
            "api_key": groq_api_key,
            "temperature": 0,
            "max_completion_tokens": 32768,
        }
        self.parse_llm = ChatOpenAI(model=PARSE_MODEL, **_llm_kwargs)
        self.score_llm = ChatOpenAI(model=SCORE_MODEL, **_llm_kwargs)
        self.rewrite_llm = ChatOpenAI(model=REWRITE_MODEL, **_llm_kwargs)

        # Efficient: Build agent prompts once
        self.agent_prompts = {
            "summary": self._get_summary_agent_prompt(),
            "contact_info": self._get_contact_info_agent_prompt(),
            "skills": self._get_skills_agent_prompt(),
            "certifications": self._get_certifications_agent_prompt(),
            "professional_experience": self._get_experience_agent_prompt(),
            "education": self._get_education_agent_prompt(),
            "projects": self._get_projects_agent_prompt(),
        }
        self._build_graph()
        self._build_graph_parsed()

    def _embed_cloud_context_in_roles(self, roles: list[dict]) -> list[dict]:
        """
        Embed cloud context directly into each role object.
        Detects which cloud(s) each role uses based on responsibilities.
        """
        from .cloud_taxonomy import detect_cloud_providers, generate_role_id

        enriched_roles = []

        for role in roles:
            enriched_role = role.copy()

            # Generate stable ID for tracking
            role_id = generate_role_id(role)

            # Combine all responsibilities to detect cloud
            responsibilities_text = " ".join(role.get("responsibilities", []))

            # Detect cloud providers for this role
            cloud_providers = detect_cloud_providers(responsibilities_text)

            # Embed context (internal field, will be stripped before final output)
            enriched_role["__cloud_context__"] = {
                "role_id": role_id,
                "cloud_providers": cloud_providers,
                "is_multi_cloud": len([c for c in cloud_providers if c != "Cloud-Agnostic"]) > 1,
                "original_bullet_count": len(role.get("responsibilities", [])),
                "original_responsibilities": role.get("responsibilities", []).copy(),
            }

            enriched_roles.append(enriched_role)

        return enriched_roles

    def _strip_cloud_context_from_roles(self, roles: list[dict]) -> list[dict]:
        """
        Remove internal cloud context fields before final output.
        """
        cleaned_roles = []
        for role in roles:
            cleaned_role = {k: v for k, v in role.items() if not k.startswith("__")}
            cleaned_roles.append(cleaned_role)
        return cleaned_roles

    def _validate_and_fix_contamination(self, updated_roles: list[dict]) -> tuple[list[dict], list[str]]:
        """
        Validate roles for cross-cloud contamination and fix violations.
        Returns a tuple: (validated_roles, contamination_notes).
        """
        from .cloud_taxonomy import detect_role_contamination

        validated_roles = []
        contamination_notes: list[str] = []

        for role in updated_roles:
            cloud_context = role.get("__cloud_context__", {})
            original_responsibilities = cloud_context.get("original_responsibilities", [])

            # Build original role for comparison
            original_role = {"responsibilities": original_responsibilities}

            # Build updated role for comparison
            updated_role_for_check = {"responsibilities": role.get("responsibilities", [])}

            # Detect contamination
            contaminations, orig_clouds, upd_clouds = detect_role_contamination(original_role, updated_role_for_check)

            if contaminations:
                allowed_clouds = cloud_context.get("cloud_providers", ["Cloud-Agnostic"])
                note = f"{role.get('title')} at {role.get('company')}: added clouds {[c['cloud'] for c in contaminations]} (allowed: {allowed_clouds})"
                contamination_notes.append(note)
                logger.warning(
                    f"Contamination detected in '{role.get('title')}' at '{role.get('company')}': "
                    f"Added clouds {[c['cloud'] for c in contaminations]}"
                )

                # Revert to original responsibilities (safe fallback)
                role["responsibilities"] = original_responsibilities
                logger.info(f"Reverted to original responsibilities for {role.get('title')}")

            validated_roles.append(role)

        return validated_roles, contamination_notes

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Add nodes with task-specific models
        workflow.add_node(
            "parse_resume",
            self._parse_resume_node,
            cache_policy=CachePolicy(),
            retry=RetryPolicy(max_attempts=5, retry_on=lambda exc: isinstance(exc, OutputParserException)),
        )
        workflow.add_node(
            "parse_jd",
            self._parse_jd_node,
            cache_policy=CachePolicy(),
            retry=RetryPolicy(max_attempts=5, retry_on=lambda exc: isinstance(exc, OutputParserException)),
        )
        workflow.add_node(
            "score_resume",
            self._score_resume_node,
            retry=RetryPolicy(max_attempts=5, retry_on=lambda exc: isinstance(exc, OutputParserException)),
        )
        workflow.add_node(
            "rewrite_resume",
            self._orchestrated_rewrite_node,
            retry=RetryPolicy(max_attempts=5, retry_on=lambda exc: isinstance(exc, OutputParserException)),
        )

        # Configure edges
        workflow.add_edge("parse_resume", "parse_jd")
        workflow.add_edge("parse_jd", "score_resume")
        workflow.add_conditional_edges(
            "score_resume", self._should_rewrite, {"rewrite": "rewrite_resume", "complete": END}
        )
        workflow.add_edge("rewrite_resume", "score_resume")
        workflow.set_entry_point("parse_resume")

        # Error handling
        workflow.add_node("handle_error", self._error_handler)
        workflow.add_edge("handle_error", END)

        self.graph = workflow.compile(cache=cache)

    def _build_graph_parsed(self):
        workflow = StateGraph(AgentState)

        # Add nodes with task-specific models
        workflow.add_node(
            "parse_jd",
            self._parse_jd_node,
            cache_policy=CachePolicy(),
            retry=RetryPolicy(max_attempts=5, retry_on=lambda exc: isinstance(exc, OutputParserException)),
        )
        workflow.add_node(
            "score_resume",
            self._score_resume_node,
            retry=RetryPolicy(max_attempts=5, retry_on=lambda exc: isinstance(exc, OutputParserException)),
        )
        workflow.add_node(
            "rewrite_resume",
            self._orchestrated_rewrite_node,
            retry=RetryPolicy(max_attempts=5, retry_on=lambda exc: isinstance(exc, OutputParserException)),
        )

        # Configure edges
        workflow.add_edge("parse_jd", "score_resume")
        workflow.add_conditional_edges(
            "score_resume", self._should_rewrite, {"rewrite": "rewrite_resume", "complete": END}
        )
        workflow.add_edge("rewrite_resume", "score_resume")
        workflow.set_entry_point("parse_jd")

        # Error handling
        workflow.add_node("handle_error", self._error_handler)
        workflow.add_edge("handle_error", END)

        self.graph_parsed = workflow.compile(cache=cache)

    def _parse_resume_node(self, state: AgentState) -> dict:
        logger.debug("Parsing resume...")
        try:
            prompt = PromptTemplate(
                template="""You are an expert-level resume parser with a meticulous eye for detail. Your task is to convert the following resume text into structured data using the ResumeData format defined in the Pydantic models.

                **MANDATORY GUIDELINES:**
                - You MUST extract **EVERY SINGLE PIECE OF INFORMATION** from the original resume text.
                - **DO NOT OMIT, SUMMARIZE, CONDENSE, OR REPHRASE** any content — especially in the **Professional Experience** and **Summary** sections.
                - The extracted data must match the original wording, structure, and detail level exactly as present in the resume.
                - Ensure all bullet points under job roles are preserved **as-is**, including all quantifiable metrics, technologies, tools, responsibilities, and achievements.
                - Include a space before any number. Example: "over 3 years of experience" ✅, not "over3 years of experience" ❌.

                If any field is genuinely missing from the resume, set it to `None`, but **never skip or infer** details not explicitly written in the text.

                Resume Text: {resume_text}


                Return ONLY the JSON, no explanations.""",
                input_variables=["resume_text"],
            )

            chain = prompt | self.parse_llm.with_structured_output(Resume, method="json_schema")
            parsed_obj = chain.invoke({"resume_text": state["raw_resume_text"]})
            parsed = parsed_obj.model_dump()

            return {"parsed_resume": parsed, "current_resume": json.dumps(parsed), "status": "scoring"}
        except OutputParserException:
            # Do NOT catch or handle here, just let it propagate
            raise
        except Exception as e:
            logger.error(f"Parsing failed: {str(e)}")
            raise

    def _parse_jd_node(self, state: AgentState) -> dict:
        logger.debug("Parsing job description...")
        prompt = PromptTemplate(
            template="""
            You are an expert job description analyzer specializing in comprehensive requirement extraction.
            Use advanced techniques to extract ALL explicit and implicit requirements.

            **EXTRACTION METHODOLOGY:**

            ** target_job_title: The main job title or position being recruited for (e.g., "Data Engineer", "Senior Software Developer"). If multiple titles are mentioned, pick the most prominent or first listed.

            **REQUIRED SKILLS** - Extract using these detection patterns:
            - Explicit indicators: "required", "must have", "essential", "mandatory", "need", "should have"
            - Implicit indicators: Skills in job title, core responsibilities, "you will", primary qualifications section
            - Technical Skills: Programming languages, databases, tools, platforms, methodologies
            - Soft Skills: Communication, leadership, problem-solving, stakeholders, analytical thinking, teamwork
            - Domain Skills: Industry-specific knowledge, business acumen, regulatory understanding
            - Do NOT omit any technology, tool, or platform mentioned, even if it appears only once (e.g., "Snowflake", "Google BigQuery", "SSMS", "Fivetran", "Tableau", etc.).

            **PREFERRED SKILLS** - Extract using these indicators:
            - Explicit: "preferred", "nice to have", "bonus", "plus", "desired", "advantageous", "would be beneficial"
            - Context clues: Secondary sections, "additional qualifications", items after "ideal candidate"

            **EXPERIENCE REQUIREMENTS** - Extract ALL threshold patterns:
            - Direct patterns: "X+ years", "X-Y years", "minimum X years"
            - Conditional patterns: "X years with Bachelor's", "Y years with Master's", "Z years or equivalent degree"
            - Domain-specific: "X years in [industry]", "Y years in [specific technology]"
            - Alternative paths: "Degree OR X years experience", "Certification OR Y years"

            **EDUCATION REQUIREMENTS** - Consolidate ALL educational pathways:
            - Degree requirements: Level (Bachelor's/Master's/PhD), preferred fields
            - Alternative experience: "OR equivalent experience" clauses
            - Professional certifications: Required vs preferred
            - Continuing education: Training, professional development expectations

            **KEY RESPONSIBILITIES** - Extract using these patterns:
            - For key_responsibilities, preserve original wording for maximum ATS relevance.
            - Bullet points and numbered lists
            - Verb-led sentences: "Develop", "Manage", "Collaborate", "Design", "Implement"
            - Responsibility indicators: "will be responsible for", "duties include", "key activities"
            - Preserve original phrasing for keyword optimization

            **COMPANY INFO** - Extract contextual information:
            - Industry: Explicitly stated or inferred from business description
            - Company size: Revenue, employee count, market presence indicators
            - Work environment: Remote/hybrid/onsite, team structure, reporting relationships
            - Company culture: Values, mission, work style indicators

            **CRITICAL PARSING RULES:**
            1. **Synonym Recognition**: Treat variations as same skill (SQL Server = MSSQL = Microsoft SQL Server)
            2. **Context Sensitivity**: If a skill appears in both required and preferred contexts, list it under required_skills.
            3. **Implicit Requirements**: Skills mentioned in responsibilities are typically required
            4. **Hierarchy Respect**: More specific trumps general (Python > Programming Languages)
            5. **Completeness Priority**: Better to over-extract than miss requirements

            Return STRICT JSON output. Include EVERY requirement mentioned - don't omit equivalent experience paths or soft skills.
            If information is not available, return an empty list or object as appropriate for the key.

            Job Description:
            {jd}
            """,
            input_variables=["jd"],
        )
        try:
            chain = prompt | self.parse_llm.with_structured_output(JD, method="json_schema")
            result_obj = chain.invoke({"jd": state["job_description"]})
            result = result_obj.model_dump()
            return {"jd_structured": result, "status": "scoring"}
        except OutputParserException:
            # Do NOT catch or handle here, just let it propagate
            raise
        except Exception as e:
            logger.error(f"JD parsing failed: {e}")
            raise

    def _score_resume_node(self, state: AgentState) -> dict:
        logger.debug("Scoring resume against job description...")
        try:
            prompt = PromptTemplate(
                template="""You are an expert ATS (Applicant Tracking System) that analyzes resumes against job descriptions.
                Your goal is to provide a comprehensive score and detailed feedback that a human can use to significantly improve their resume.
                Analyze how well the provided resume matches the job description, considering keywords, skills, experience, and overall relevance.

                **CRITICAL KEYWORD ANALYSIS REQUIREMENTS:**
                - You MUST identify and flag EVERY SINGLE keyword from the job description that is missing from the resume
                - Do not overlook ANY technical term, tool, or concept mentioned in the job requirements
                - Treat each keyword as critical, even if it appears only once in the job description
                - Flag ALL missing keywords, regardless of how minor they may seem

                **CRITICAL RULES:**
                - If previous_score provided: Only give feedback on NEW issues or PERSISTENT problems
                - Return empty arrays for sections with no new issues

                Provide a score from 0-100, a summary, and detailed lists of strengths, weaknesses, missing keywords, suggested skills, actionable recommendations, and specific sections to improve.

                Job Requirements:
                ```json
                Required Skills: {required_skills}
                Preferred Skills: {preferred_skills}
                Experience: {experience_requirements}
                Education: {education_requirements}
                Responsibilities: {key_responsibilities}
                Company Info: {company_info}
                ```

                Resume Content:
                ```json
                  {resume}
                ```

                Previous Analysis (for reference only):
                ```json
                    Previous Score: {previous_score}
                    Previous Summary: {previous_summary}
                ```

                SCORING (0-100):
                - Keywords: 40pts | Skills: 25pts | Experience: 25pts | Education: 10pts

                **SECTION-BY-SECTION ANALYSIS REQUIREMENTS:**

                **1. SUMMARY SECTION:**
                - Analyze current professional summary effectiveness
                - Identify missing keywords from JD that should be in summary
                - List missing value propositions from JD
                - Provide specific rewrite instructions
                - Recommend target length and tone

                **2. CONTACT_INFO SECTION:**
                - Check for missing professional contact elements
                - Suggest improvements to professional presentation
                - Identify any formatting issues

                **3. SKILLS SECTION:**
                - List technical skills from JD missing in resume
                - List soft skills from JD missing in resume
                - Identify existing skills that need more prominence
                - Suggest better skill categorization/organization
                - Recommend priority order based on JD importance

                **4. CERTIFICATIONS SECTION:**
                - Identify certifications mentioned in JD but missing from resume
                - Suggest existing certifications to emphasize more
                - Recommend additional relevant certifications

                **5. PROFESSIONAL_EXPERIENCE SECTION:**
                - For EACH job in experience, provide specific instructions:
                * Missing keywords from JD for that role
                * Responsibilities to add or modify
                * Areas needing quantification (metrics, percentages, dollar amounts)
                * Achievements to highlight
                * Better action verbs from JD language
                - Provide overall experience section improvements
                - Suggest experience ordering for relevance

                **6. EDUCATION SECTION:**
                - **CRITICAL FORMATTING ANALYSIS:**
                * Compare degree format in resume vs JD requirements (e.g., "MS" vs "Master of Science" vs "M.S.")
                * If JD uses "MS" and resume uses "Master of Science", add for format change
                * If JD uses "Bachelor of Science" and resume uses "BS", add for format change
                * Identify field of study emphasis needed (e.g., "Computer Science" vs "CS")

                **7. PROJECTS SECTION:**
                - For EACH project, analyze alignment with JD requirements
                - Identify missing technologies from JD to highlight
                - Suggest outcomes/metrics to add
                - Recommend description improvements
                - Suggest new projects that would strengthen application

                **CRITICAL ANTI-REDUNDANCY REQUIREMENTS:**
                You *MUST* thoroughly analyze `Previous ATS Feedback History` and provide *ONLY*:
                1. **NEW issues** not previously identified
                2. **PERSISTENT issues** that remain despite previous attempts (mark as "PERSISTENT:")
                3. **REFINED guidance** for issues that were partially addressed but need improvement

                *DO NOT* repeat feedback that has been successfully implemented. If a section shows no new or persistent issues, provide empty instructions for that section.

                """,
                input_variables=[
                    "required_skills",
                    "preferred_skills",
                    "experience_requirements",
                    "education_requirements",
                    "key_responsibilities",
                    "company_info",
                    "resume",
                    "previous_score",
                    "previous_summary",
                ],
            )

            chain = prompt | self.score_llm.with_structured_output(ATSScoreOutput, method="json_schema")
            feedback_obj = chain.invoke(
                {
                    "required_skills": state["jd_structured"].get("required_skills", []),
                    "preferred_skills": state["jd_structured"].get("preferred_skills", []),
                    "experience_requirements": state["jd_structured"].get("experience_requirements", []),
                    "education_requirements": state["jd_structured"].get("education_requirements", []),
                    "key_responsibilities": state["jd_structured"].get("key_responsibilities", []),
                    "company_info": state["jd_structured"].get("company_info", {}),
                    "resume": state["current_resume"],
                    "previous_score": state["ats_feedback"].get("score", "") if state["ats_feedback"] else "",
                    "previous_summary": state["ats_feedback"].get("summary", "") if state["ats_feedback"] else "",
                }
            )
            feedback = feedback_obj.model_dump()

            updates = {
                "ats_feedback": feedback,
                "ats_feedback_history": state["ats_feedback_history"] + [feedback],
                "iterations": state["iterations"] + 1,
                "status": "rewriting"
                if feedback["score"] < TARGET_SCORE and state["iterations"] < MAX_REWRITE_ATTEMPTS
                else "complete",
            }

            if state["iterations"] == 0:
                updates["initial_ats_score"] = feedback["score"]
            updates["final_ats_score"] = feedback["score"]

            return updates
        except OutputParserException:
            # Do NOT catch or handle here, just let it propagate
            raise
        except Exception as e:
            logger.error(f"Scoring failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _orchestrated_rewrite_node(self, state: AgentState) -> dict:
        """
        Parent orchestration node that coordinates specialized agents for each resume section.
        This ensures maximum quality and coherence across all sections while maintaining
        temporal accuracy and ATS optimization.
        """
        logger.debug("Orchestrating rewrite with specialized agents...")
        try:
            # Extract coordination data from state
            ats_feedback = state["ats_feedback"]
            section_tasks = ats_feedback.get("section_instructions", {})
            coordination_strategy = state.get("coordination_strategy", {})

            current_resume = json.loads(state["current_resume"])

            # Phase 1: Execute specialized agents in priority order
            logger.info("Starting orchestrated rewrite with specialized agents")

            # Get rewrite priorities from coordination strategy
            rewrite_priorities = coordination_strategy.get("rewrite_priorities", [])
            if not rewrite_priorities:
                # rewrite_priorities = ["summary", "skills", "professional_experience", "projects", "certifications", "education", "contact_info"]
                rewrite_priorities = ["summary", "skills", "professional_experience", "projects", "education"]

            # Execute agents in coordinated manner
            section_results = self._execute_specialized_agents(
                current_resume=current_resume,
                jd_structured=state["jd_structured"],
                ats_feedback=ats_feedback,
                section_tasks=section_tasks,
                rewrite_priorities=rewrite_priorities,
                coordination_strategy=coordination_strategy,
            )
            # Phase 2: Integrate and validate results
            integrated_resume = self._integrate_section_results(
                original_resume=current_resume, section_results=section_results, ats_feedback=ats_feedback
            )
            # Phase 3: Final coherence and quality check
            final_resume = self._final_coherence_validation(
                integrated_resume=integrated_resume, jd_structured=state["jd_structured"], ats_feedback=ats_feedback
            )

            return {
                "current_resume": json.dumps(final_resume),
                "status": "scoring",
                "rewrite_summary": {
                    "sections_processed": list(section_results.keys()),
                    "total_improvements": sum(
                        len(result.get("improvements", [])) for result in section_results.values()
                    ),
                    "coordination_applied": True,
                },
            }

        except OutputParserException:
            # Do NOT catch or handle here, just let it propagate
            raise

        except Exception as e:
            logger.error(f"Orchestrated rewriting failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _execute_specialized_agents(
        self,
        current_resume: dict,
        jd_structured: dict,
        ats_feedback: dict,
        section_tasks: dict,
        rewrite_priorities: list[str],
        coordination_strategy: dict,
    ) -> dict:
        """Execute specialized agents for each section with proper coordination"""
        logger.debug("Executing specialized agents for each section...")

        # Pre-build shared context without processed_sections
        base_shared_context = {
            "jd_structured": jd_structured,
            "ats_feedback": ats_feedback,
            "coordination_strategy": coordination_strategy,
            "temporal_context": self._build_temporal_context(current_resume),
        }

        def process_section(section_info: dict[str, Any]) -> tuple[str, dict]:
            """Process a single section with its own context"""
            section_name = section_info["name"]
            section_data = section_info["data"]
            section_instructions = section_info["instructions"]

            # Create section-specific context
            section_context = base_shared_context.copy()

            try:
                result = self._execute_section_agent(
                    section_name=section_name,
                    current_section_data=section_data,
                    section_instructions=section_instructions,
                    shared_context=section_context,
                )
                return section_name, result
            except Exception as e:
                logger.error(f"Failed to process section {section_name}: {str(e)}")
                return section_name, {"status": "error", "data": section_data, "error": str(e)}

        # Prepare section processing tasks
        section_tasks_list = [
            {
                "name": section_name,
                "data": current_resume.get(section_name),
                "instructions": section_tasks.get(section_name),
            }
            for section_name in rewrite_priorities
            if section_name in section_tasks
        ]

        # Process sections in parallel with a thread pool
        section_results = {}
        with ThreadPoolExecutor(max_workers=min(4, len(section_tasks_list))) as executor:
            # Submit all tasks
            future_to_section = {
                executor.submit(process_section, section_info): section_info["name"]
                for section_info in section_tasks_list
            }

            # Collect results in order of completion
            for future in concurrent.futures.as_completed(future_to_section):
                section_name, result = future.result()
                section_results[section_name] = result
                logger.debug(f"Completed processing section: {section_name}")

        return section_results

    def _calculate_role_information(self, section_name: str, current_section_data: Any) -> dict:
        """Calculate role count and bullet counts for professional experience section with validation and order preservation"""
        logger.debug(f"Calculating role information for section: {section_name}")

        role_info = {
            "role_count": 0,
            "role_bullet_counts": {},
            "role_order": [],  # preserve order
        }

        if section_name == "professional_experience" and isinstance(current_section_data, list):
            role_count = len(current_section_data)

            # Validation: Ensure we have at least one role
            if role_count == 0:
                logger.warning("WARNING: No professional experience roles found. This may indicate a parsing issue.")
                return role_info

            role_info["role_count"] = role_count

            # Calculate bullet counts for each role and preserve order
            for i, role in enumerate(current_section_data):
                if not isinstance(role, dict):
                    logger.warning(f"WARNING: Role {i + 1} is not a dictionary. Skipping.")
                    continue
                role_key = f"role_{i}"
                role_info["role_order"].append(role_key)
                responsibilities = role.get("responsibilities", [])
                # Validation: Ensure responsibilities is a list
                if not isinstance(responsibilities, list):
                    logger.warning(f"WARNING: Responsibilities for role {i + 1} is not a list. Setting to empty list.")
                    responsibilities = []
                role_info["role_bullet_counts"][role_key] = {
                    "title": role.get("title", f"Role {i + 1}"),
                    "company": role.get("company", "Unknown Company"),
                    "bullet_count": len(responsibilities),
                }
                logger.debug(
                    f"Role {i + 1}: {role.get('title', 'Unknown')} at {role.get('company', 'Unknown')} - {len(responsibilities)} bullets"
                )

        return role_info

    def _validate_experience_output(
        self, output_data: dict, expected_role_count: int, expected_bullet_counts: dict
    ) -> bool:
        """Validate that the LLM output contains the expected number of roles and bullet counts"""
        logger.debug("Validating experience output...")

        try:
            updated_section = output_data.get("updated_section", [])

            # Check if output is a list
            if not isinstance(updated_section, list):
                logger.error("ERROR: updated_section is not a list")
                return False

            # Check role count
            actual_role_count = len(updated_section)
            if actual_role_count < expected_role_count:
                logger.error(f"ERROR: Expected {expected_role_count} roles, got {actual_role_count}")
                return False

            # Check bullet counts for each role
            for i, role in enumerate(updated_section):
                if not isinstance(role, dict):
                    logger.error(f"ERROR: Role {i + 1} is not a dictionary")
                    return False

                role_key = f"role_{i}"
                expected_bullets = expected_bullet_counts.get(role_key, {}).get("bullet_count", 0)
                actual_bullets = len(role.get("responsibilities", []))

                if actual_bullets < (expected_bullets - 1):
                    logger.error(f"ERROR: Role {i + 1} expected {expected_bullets} bullets, got {actual_bullets}")
                    return False

            logger.debug("Experience output validation passed")
            return True

        except Exception as e:
            logger.error(f"ERROR: Validation failed with exception: {str(e)}")
            return False

    def _split_roles_by_years(self, professional_experience, max_years=7):
        """Split roles into those within the most recent max_years and the rest, preserving order."""
        from datetime import datetime

        from dateutil.parser import parse

        processed_roles = []
        untouched_roles = []
        total_months = 0
        current_keywords = {"present", "current", "now"}
        for role in professional_experience:
            start = parse(role.get("start_date", ""))
            end_date_value = role.get("end_date", "")
            # Handle None values safely
            end_str = "" if end_date_value is None else str(end_date_value).strip()
            end = datetime.now() if end_str.lower() in current_keywords or not end_str else parse(end_str)
            months = (end.year - start.year) * 12 + (end.month - start.month)
            if total_months < max_years * 12:
                processed_roles.append(role)
                total_months += months
            else:
                untouched_roles.append(role)
        return processed_roles, untouched_roles

    def _execute_section_agent(
        self, section_name: str, current_section_data: Any, section_instructions: dict, shared_context: dict
    ) -> dict:
        """Execute a specialized agent for a specific resume section. Only 'professional_experience' uses retry/validation logic."""
        logger.debug(f"Executing specialized agent for section: {section_name}")
        prompt_template = self.agent_prompts.get(section_name)
        if not prompt_template:
            logger.warning(f"No specialized agent found for section: {section_name}")
            return {"status": "skipped", "data": current_section_data}
        try:
            if section_name == "professional_experience":
                # Step 1: Embed cloud context into roles BEFORE processing
                enriched_roles = self._embed_cloud_context_in_roles(current_section_data)

                # Step 2: Split by years (cloud context travels with roles)
                roles_to_process, untouched_roles = self._split_roles_by_years(enriched_roles, max_years=7)

                # Step 3: Calculate role info
                role_info = self._calculate_role_information(section_name, roles_to_process)

                # Step 4: Extract cloud context summary for prompt
                role_cloud_summary = []
                for idx, role in enumerate(roles_to_process):
                    ctx = role.get("__cloud_context__", {})
                    role_cloud_summary.append(
                        {
                            "index": idx,
                            "title": role.get("title"),
                            "company": role.get("company"),
                            "cloud_providers": ctx.get("cloud_providers", ["Cloud-Agnostic"]),
                            "bullet_count": ctx.get("original_bullet_count", 0),
                        }
                    )
                chain = prompt_template | self.rewrite_llm | JsonOutputParser()
                max_retries = 3
                validation_passed = False
                result = None
                best_result = None
                best_role_count = 0
                best_bullet_count = 0
                contamination_for_next: list[str] = []
                for attempt in range(max_retries):
                    logger.debug(f"Attempt {attempt + 1} for section: {section_name}")
                    retry_context = ""
                    if attempt > 0:
                        retry_context = f"""
                                            **CRITICAL RETRY INSTRUCTION - ATTEMPT {attempt + 1}:**
                                            The previous response was missing roles or had incorrect bullet counts.
                                            You MUST return EXACTLY {role_info["role_count"]} roles with the following bullet counts and preserve the original order (role_0, role_1, ...):
                                        """
                        for _idx, role_key in enumerate(role_info["role_order"]):
                            role_data = role_info["role_bullet_counts"][role_key]
                            retry_context += f"- {role_key}: {role_data['title']} at {role_data['company']}: {role_data['bullet_count']} bullets\n"
                        retry_context += (
                            "\nDO NOT skip any roles. Process ALL roles from the input. Preserve the original order."
                        )
                        if contamination_for_next:
                            retry_context += "\n\n**CONTAMINATION WARNING:**\n"
                            retry_context += "For EACH role, use ONLY technologies from the role's allowed cloud_providers shown below. If allowed is ['Cloud-Agnostic'], DO NOT add AWS/Azure/GCP terms; keep it cloud-agnostic. If allowed is ['AWS'], use only AWS or cloud-agnostic tech (no Azure/GCP). If unsure, keep the original bullets.\n"
                            for note in contamination_for_next:
                                retry_context += f"- {note}\n"
                    result = chain.invoke(
                        {
                            "role_count": role_info["role_count"],
                            "role_bullet_counts": json.dumps(role_info["role_bullet_counts"]),
                            "role_order": json.dumps(role_info["role_order"]),
                            "role_cloud_summary": json.dumps(role_cloud_summary),  # NEW: Cloud context
                            "current_section": json.dumps(roles_to_process),
                            "section_instructions": json.dumps(section_instructions),
                            "jd_structured": json.dumps(shared_context["jd_structured"]),
                            "ats_feedback": json.dumps(shared_context["ats_feedback"]),
                            "temporal_context": json.dumps(shared_context["temporal_context"]),
                            "processed_sections": json.dumps(shared_context.get("processed_sections", {})),
                            "coordination_strategy": json.dumps(shared_context["coordination_strategy"]),
                            "retry_context": retry_context,
                        }
                    )
                    output_roles = result.get("updated_section", [])

                    # Step 5: Re-attach cloud context to output roles for validation
                    for idx, out_role in enumerate(output_roles):
                        if idx < len(roles_to_process):
                            out_role["__cloud_context__"] = roles_to_process[idx].get("__cloud_context__", {})

                    # Step 6: Validate and fix contamination
                    output_roles, contamination_notes = self._validate_and_fix_contamination(output_roles)

                    role_count = len(output_roles)
                    bullet_count = sum(len(r.get("responsibilities", [])) for r in output_roles)
                    if (role_count > best_role_count) or (
                        role_count == best_role_count and bullet_count > best_bullet_count
                    ):
                        best_result = result
                        best_role_count = role_count
                        best_bullet_count = bullet_count
                    validation_passed = self._validate_experience_output(
                        result, role_info["role_count"], role_info["role_bullet_counts"]
                    )

                    # Success only if structure valid and no contamination
                    if validation_passed and not contamination_notes:
                        logger.debug(f"Validation passed on attempt {attempt + 1}")
                        contamination_for_next = []
                        break
                    else:
                        logger.debug(f"Validation failed on attempt {attempt + 1}")
                        # Prepare contamination guidance for next attempt
                        contamination_for_next = contamination_notes if contamination_notes else []

                        if attempt == max_retries - 1:
                            logger.error(f"All {max_retries} attempts failed for section: {section_name}")
                            # Fallback: merge missing roles/points from best LLM output, preserving order
                            original_roles = roles_to_process
                            output_roles = best_result.get("updated_section", []) if best_result else []

                            def role_key(role):
                                return (role.get("title", ""), role.get("company", ""))

                            output_roles_map = {role_key(r): r for r in output_roles}
                            merged_roles = []
                            for _idx, orig_role in enumerate(original_roles):
                                key = role_key(orig_role)
                                if key in output_roles_map:
                                    out_role = output_roles_map[key]
                                    orig_bullets = orig_role.get("responsibilities", [])
                                    out_bullets = out_role.get("responsibilities", [])
                                    if len(out_bullets) < len(orig_bullets):
                                        out_bullets += orig_bullets[len(out_bullets) :]
                                        out_role["responsibilities"] = out_bullets
                                    merged_roles.append(out_role)
                                else:
                                    merged_roles.append(orig_role)
                            # Merge with untouched roles, preserving original order
                            merged_roles.extend(untouched_roles)
                            return {
                                "status": "success",
                                "data": merged_roles,
                                "improvements": best_result.get("improvements", []) if best_result else [],
                                "keywords_added": best_result.get("keywords_added", []) if best_result else [],
                                "validation_notes": (best_result.get("validation_notes", []) if best_result else [])
                                + [
                                    "Fallback: Original roles/points used for missing data. Order preserved. Used best LLM output."
                                ],
                            }
                        continue
                # On success, merge processed and untouched roles, preserving order
                merged_roles = output_roles + untouched_roles

                # Step 7: Strip cloud context before returning
                final_roles = self._strip_cloud_context_from_roles(merged_roles)

                return {
                    "status": "success",
                    "data": final_roles,
                    "improvements": result.get("improvements", []),
                    "keywords_added": result.get("keywords_added", []),
                    "validation_notes": result.get("validation_notes", []),
                }
            else:
                # For all other sections, just run the agent once and return the result
                chain = prompt_template | self.rewrite_llm | JsonOutputParser()
                result = chain.invoke(
                    {
                        "current_section": json.dumps(current_section_data),
                        "section_instructions": json.dumps(section_instructions),
                        "jd_structured": json.dumps(shared_context["jd_structured"]),
                        "ats_feedback": json.dumps(shared_context["ats_feedback"]),
                        "temporal_context": json.dumps(shared_context["temporal_context"]),
                        "processed_sections": json.dumps(shared_context.get("processed_sections", {})),
                        "coordination_strategy": json.dumps(shared_context["coordination_strategy"]),
                    }
                )
                return {
                    "status": "success",
                    "data": result.get("updated_section"),
                    "improvements": result.get("improvements", []),
                    "keywords_added": result.get("keywords_added", []),
                    "validation_notes": result.get("validation_notes", []),
                }
        except Exception as e:
            logger.error(f"Section agent failed for {section_name}: {str(e)}")
            return {"status": "error", "data": current_section_data, "error": str(e)}

    def _get_summary_agent_prompt(self) -> PromptTemplate:
        """Specialized agent for summary section optimization"""
        logger.debug("Creating specialized agent for summary section optimization...")
        return PromptTemplate(
            template="""You are an expert Summary Section Specialist focused on creating compelling professional summaries that maximize ATS compatibility and recruiter appeal.

    **INPUTS:**
    - Current Summary: {current_section}
    - Section Instructions: {section_instructions}
    - Job Description: {jd_structured}
    - Temporal Context: {temporal_context}
    - Processed Sections: {processed_sections}
    - Coordination Strategy: {coordination_strategy}

    **STRICT REQUIREMENT:**
    You MUST follow ALL instructions provided in section_instructions exactly and completely. Do not skip, ignore, or modify any instruction from section_instructions. If there is any conflict between section_instructions and other guidelines, always prioritize and strictly follow section_instructions.

    **YOUR MISSION:**
    Create a powerful, keyword-optimized professional summary that serves as the resume's hook while perfectly aligning with the job requirements and strictly following section_instructions.

    **SUMMARY OPTIMIZATION RULES:**
    1. **Keyword Integration**: Seamlessly incorporate ALL missing keywords from instructions
    2. **Experience Alignment**: Ensure years of experience match the temporal context exactly
    3. **Value Proposition**: Lead with the strongest value propositions from instructions
    4. **Length Optimization**: Follow target length guidance (typically 3-4 impactful sentences)
    5. **Professional Tone**: Maintain confident, results-oriented language
    6. **Quantifiable Impact**: Include 1-2 key metrics that demonstrate value
    7. **Role Alignment**: Mirror language and priorities from job description

    **WRITING STYLE REQUIREMENTS:**
    - NEVER use first-person pronouns (I, my, me, mine)
    - Write in third-person or use direct statements without pronouns
    - Use active voice and strong action-oriented language
    - Example: Instead of "I have 5+ years of experience" use "Professional with 5+ years of experience"
    - Example: Instead of "My expertise includes" use "Expertise includes" or "Specializes in"

    **TEMPORAL ACCURACY REQUIREMENTS:**
    - Calculate total experience from temporal_context
    - Use appropriate technology terms for the experience level
    - Avoid anachronistic references (e.g., don't mention 2023 AI tools for 2018 experience)

    **OUTPUT FORMAT(STRICT JSON):**
    Return ONLY valid JSON in the EXACT format below, with NO extra keys, NO markdown, and NO explanations:
    ```json
    {{
        "updated_section": "Your optimized summary text here",
        "improvements": ["List of specific improvements made"],
        "keywords_added": ["List of keywords successfully integrated"],
        "validation_notes": ["Notes on temporal accuracy and alignment"]
    }}
    ```

    Generate an exceptional summary that makes recruiters want to read more, and that strictly follows all section_instructions provided.""",
            input_variables=[
                "current_section",
                "section_instructions",
                "jd_structured",
                "ats_feedback",
                "temporal_context",
                "processed_sections",
                "coordination_strategy",
            ],
        )

    def _get_skills_agent_prompt(self) -> PromptTemplate:
        """Specialized agent for skills section optimization"""
        logger.debug("Creating specialized agent for skills section optimization...")
        return PromptTemplate(
            template="""You are an expert Skills Section Specialist focused on creating strategically organized, ATS-optimized skills sections that perfectly align with job requirements.

    **INPUTS:**
    - Current Skills: {current_section}
    - Section Instructions: {section_instructions}
    - Job Description: {jd_structured}
    - Temporal Context: {temporal_context}
    - Processed Sections: {processed_sections}

    **YOUR MISSION:**
    Transform the skills section into a powerful, strategically organized showcase that maximizes ATS scoring and demonstrates perfect job alignment.

    **SKILLS OPTIMIZATION RULES:**
    1. **Priority Integration**: Add ALL missing technical and soft skills from instructions
    2. **Strategic Categorization**: Organize skills by relevance and impact (NO generic "Technical Skills" category)
    3. **JD Language Matching**: Use exact terminology from job description
    4. **Skill Prioritization**: Place most critical skills first within each category
    5. **Temporal Validation**: Ensure all skills are appropriate for the candidate's experience timeline
    6. **Complementary Planning**: Consider how skills will be demonstrated in experience/projects sections

    **CATEGORIZATION STRATEGY:**
    - Create specific, meaningful categories (e.g., "Programming Languages", "Cloud Platforms", "Data Analysis Tools")
    - Group related skills logically
    - Prioritize categories by job description importance
    - Ensure comprehensive coverage of job requirements

    **OUTPUT FORMAT(STRICT JSON):**
    Return ONLY valid JSON in the EXACT format below, with NO extra keys, NO markdown, and NO explanations:
    ```json
    {{
        "updated_section": [
            {{
                "category": "Category Name",
                "skills": ["skill1", "skill2", "skill3"]
            }}
        ],
        "improvements": ["List of specific improvements made"],
        "keywords_added": ["List of new skills added"],
        "validation_notes": ["Notes on categorization and prioritization strategy"]
    }}
    ```

    Create a skills section that immediately demonstrates perfect job fit, while remaining strictly truthful and ATS-friendly.""",
            input_variables=[
                "current_section",
                "section_instructions",
                "jd_structured",
                "ats_feedback",
                "temporal_context",
                "processed_sections",
            ],
        )

    def _get_experience_agent_prompt(self) -> PromptTemplate:
        """Specialized agent for professional experience optimization"""
        logger.debug("Creating specialized agent for professional experience optimization...")
        return PromptTemplate(
            template="""
You are an elite Professional Experience Section Specialist. Your job is to strictly analyze the candidate's entire professional experience and make the most contextually appropriate, detailed, and achievement-oriented changes to align the experience with the job description (JD).

    **INPUTS:**
    - Full Professional History: {current_section}
    - Section Instructions: {section_instructions}
    - Role Bullet Counts: {role_bullet_counts}
    - Role Cloud Summary: {role_cloud_summary}
    - Retry Context: {retry_context}

**CRITICAL CLOUD CONSTRAINT (MUST FOLLOW):**

Each role has a native cloud stack that you MUST respect. The role_cloud_summary shows which cloud(s) each role uses.

**RULES BY CLOUD TYPE:**

1. **Role with cloud_providers: ["AWS"]**
   - ✅ CAN add: AWS services (S3, Redshift, Glue, Lambda, etc.)
   - ✅ CAN add: Cloud-agnostic (Python, SQL, Spark, Airflow, etc.)
   - ❌ CANNOT add: GCP services (BigQuery, Dataflow, etc.)
   - ❌ CANNOT add: Azure services (Synapse, Data Factory, etc.)

2. **Role with cloud_providers: ["GCP"]**
   - ✅ CAN add: GCP services (BigQuery, Dataflow, Pub/Sub, etc.)
   - ✅ CAN add: Cloud-agnostic (Python, SQL, Spark, Airflow, etc.)
   - ❌ CANNOT add: AWS services (Redshift, Glue, etc.)
   - ❌ CANNOT add: Azure services (Synapse, Data Factory, etc.)

3. **Role with cloud_providers: ["Azure"]**
   - ✅ CAN add: Azure services (Synapse, Data Factory, Databricks, etc.)
   - ✅ CAN add: Cloud-agnostic (Python, SQL, Spark, Airflow, etc.)
   - ❌ CANNOT add: AWS services (Redshift, Glue, etc.)
   - ❌ CANNOT add: GCP services (BigQuery, Dataflow, etc.)

4. **Role with cloud_providers: ["AWS", "GCP"] (multi-cloud)**
   - ✅ CAN add: AWS services
   - ✅ CAN add: GCP services
   - ✅ CAN add: Cloud-agnostic
   - ❌ CANNOT add: Azure services (not in this role's stack)

5. **Role with cloud_providers: ["Cloud-Agnostic"] or ["On-Premise"]**
   - ✅ CAN add: Cloud-agnostic technologies (Python, SQL, Spark, etc.)
   - ✅ CAN add: On-premise technologies (Hadoop, Kafka, etc.)
   - ⚠️ AVOID adding specific cloud services unless clearly appropriate

**WHAT TO DO WHEN JD REQUIRES A CLOUD THE ROLE DOESN'T HAVE:**

If JD requires Azure but role is AWS:
- ❌ DON'T add Azure technologies to this role
- ✅ DO enhance the role with better metrics, achievements, action verbs
- ✅ DO add cloud-agnostic skills that transfer (Python, SQL, data modeling)
- ✅ DO improve bullet quality without changing cloud stack

**STRICT EXECUTION PROTOCOL:**
1. **CONTEXT ANALYSIS PHASE:**
   - Review Full Professional History
   - Identify missing JD requirements: {jd_structured}
   - Verify technology timelines: {temporal_context}

2. **CRITICAL COMPLETENESS REQUIREMENTS:**
   - Input Role Count: {role_count}
   - Required Output: Process ALL {role_count} roles without exception
   - You MUST include EVERY SINGLE JOB ROLE from the original experience section
   - DO NOT skip or omit ANY job roles, regardless of relevance
   - Each role MUST maintain its EXACT original bullet point count as specified in role_bullet_counts

3. **BULLET REPLACEMENT & ELABORATION RULES:**
   For each role:
   - Maintain EXACT original bullet count as specified in role_bullet_counts
   - Only replace bullets that:
     * Are least relevant to JD requirements
     * Contain redundant/outdated information
     * Don't demonstrate measurable impact
   - New bullets MUST:
     * Be 15-30 words, elaborated and achievement-oriented
     * Clearly state context, action, and measurable result (use STAR: Situation, Task, Action, Result)
     * MATCH the role's actual technology stack
     * Use past tense for completed roles
     * Contain at least one quantifiable result ($, %, #, time)
     * INTEGRATE missing JD keywords naturally and meaningfully

4. **TECHNOLOGY CONTEXT & CLOUD STACK CONSISTENCY RULE:**
   - For each role, only use technologies from the same cloud provider as evidenced in the candidate's actual experience for that role.
   - NEVER mix AWS and GCP (or Azure) technologies in the same bullet or project unless the candidate's history explicitly shows multi-cloud experience.
   - If the candidate has no GCP/BigQuery experience, do NOT add BigQuery to any bullet, even if the JD requires it. Instead, bridge with relevant AWS experience if possible, or leave it out.
   - NEVER add technologies to incompatible roles:
     ```python
     if (role_tech == "Azure/SQL Server") and (new_tech == "Redshift"):
         REJECT addition
     if (role_tech == "Kafka") and (new_tech == "Azure Synapse"):
         REJECT addition
     ```
   - Only add technology if:
     * It existed during role timeframe
     * It's logically compatible with stack
     * Candidate has verifiable experience

5. **KEYWORD INTEGRATION:**
   - Embed missing keywords ONLY through replacement bullets
   - Each "keywords_added" MUST appear verbatim in new bullets

**EXAMPLE EXECUTION:**
JD Requires: Redshift, Data Warehousing, BigQuery
Role Context: Azure Data Engineer (SQL Server/ADF)

**WRONG:**
Normalized and staged extracted data using AWS Glue Scripts into S3, streamlining the data pipeline and enhancing accessibility for downstream operations, thereby improving data usability and reliability by leveraging BigQuery for advanced analytics.

**RIGHT:**
Normalized and staged extracted data using AWS Glue Scripts into S3, streamlining the data pipeline and enhancing accessibility for downstream operations, enabling advanced analytics with AWS-native tools.

If the candidate has GCP experience in another role, keep the clouds separate per role.

**ABSOLUTE PROHIBITIONS:**
- ✗ Adding bullets without removing existing ones
- ✗ Mixing conflicting or cross-cloud technologies in same role
- ✗ Exceeding original bullet point count
- ✗ Including unverifiable technologies
- ✗ Listing keywords not present in output

**OUTPUT FORMAT (STRICT JSON):**
Return ONLY valid JSON in the EXACT format below, with NO extra keys, NO markdown, and NO explanations.
The number of bullet points in 'responsibilities' for each role MUST be the same as in the original.
**IMPORTANT:** Each keyword in "keywords_added" MUST be present as a descriptive point in the bullet points above, and you MUST show which bullet it was integrated into.
```json
{{
    "updated_section": [
        {{
            "title": "Job Title",
            "company": "Company Name",
            "location": "Location",
            "start_date": "MM/YYYY",
            "end_date": "MM/YYYY or Present",
            "responsibilities": ["Elaborated, achievement-oriented bullet point 1", "Elaborated, achievement-oriented bullet point 2"]
        }}
    ],
    "improvements": ["List of specific improvements made (e.g., replaced X with Y)"],
    "keywords_added": ["List of keywords integrated"],
    "validation_notes": ["Notes on contextual appropriateness, technology alignment, and bullet count preservation"]
}}
```

Transform the professional experience section into a detailed, truthful, and ATS-optimized showcase that aligns with the job description, demonstrates measurable impact, and always preserves the original bullet count per role.""",
            input_variables=[
                "role_count",
                "role_bullet_counts",
                "role_order",
                "role_cloud_summary",
                "current_section",
                "section_instructions",
                "jd_structured",
                "ats_feedback",
                "temporal_context",
                "processed_sections",
                "coordination_strategy",
                "retry_context",
            ],
        )

    def _get_projects_agent_prompt(self) -> PromptTemplate:
        """Specialized agent for projects section optimization"""
        logger.debug("Creating specialized agent for projects section optimization...")
        return PromptTemplate(
            template="""You are an expert Projects Section Specialist focused on showcasing technical projects that perfectly demonstrate job-relevant skills and capabilities.

    **INPUTS:**
    - Current Projects: {current_section}
    - Section Instructions: {section_instructions}
    - Job Description: {jd_structured}
    - Temporal Context: {temporal_context}
    - Processed Sections: {processed_sections}

    **YOUR MISSION:**
    Transform the projects section into compelling evidence of technical capability, ensuring each project demonstrates relevant skills and perfect alignment with job requirements.

    **PROJECT OPTIMIZATION RULES:**
    1. **Skills Demonstration**: Each project must showcase skills from the processed skills section
    2. **Technology Alignment**: Incorporate missing technologies identified in instructions, but ONLY if they are relevant to the project and can be truthfully added based on the candidate's background
    3. **Outcome Focus**: Transform descriptions into achievement-focused narratives with metrics
    4. **Temporal Accuracy**: Ensure all technologies mentioned were available during project timelines
    5. **JD Language**: Use terminology and concepts directly from job description
    6. **Technical Depth**: Show appropriate technical complexity for the role level
    7. **Business Impact**: Connect technical achievements to business value
    8. **RELEVANCY & CONSISTENCY**: When adding missing technologies, ensure they are highly relevant to the project and that the new point is consistent with the other project points. Do NOT add technologies that do not fit the context or would be out of place with the rest of the project description.
    9. **COMPLETENESS**: Make sure no relevant missing technology is left unadded, as long as it is truthful and contextually appropriate.

    **PROJECT ENHANCEMENT STRATEGY:**
    For each project:
    - Address alignment issues from instructions
    - Integrate missing technologies naturally, but ONLY if they are relevant and consistent with the rest of the project
    - Add quantifiable outcomes and metrics
    - Improve descriptions per guidance
    - Ensure complementarity with professional experience
    - Please don't add redundant technologies in one single project (eg: Spark Streaming and Azure Stream Analytics can't be used in single project as they both are streaming technologies)

    **VALIDATION REQUIREMENTS:**
    - Verify technology stack timeline accuracy
    - Ensure project complexity matches experience level
    - Validate that outcomes are realistic and measurable
    - Check for complementarity with work experience
    - Ensure all additions are truthful and ATS-friendly

    **OUTPUT FORMAT(STRICT JSON):**
    Return ONLY valid JSON in the EXACT format below, with NO extra keys, NO markdown, and NO explanations:
    ```json
    {{
        "updated_section": [
            {{
                "title": "Project Title",
                "description": "Enhanced project description with outcomes and metrics"
            }}
        ],
        "improvements": ["List of specific improvements made"],
        "keywords_added": ["List of technologies/keywords integrated"],
        "validation_notes": ["Notes on temporal accuracy and technical alignment"]
    }}
    ```

    Create projects that serve as powerful proof points for technical capability and job readiness, while remaining strictly truthful and ATS-friendly.""",
            input_variables=[
                "current_section",
                "section_instructions",
                "jd_structured",
                "ats_feedback",
                "temporal_context",
                "processed_sections",
            ],
        )

    def _get_contact_info_agent_prompt(self) -> PromptTemplate:
        """Specialized agent for contact info optimization"""
        logger.debug("Creating specialized agent for contact information optimization...")
        return PromptTemplate(
            template="""You are an expert Contact Information Specialist focused on ensuring professional, complete, and ATS-optimized contact details.

    **INPUTS:**
    - Current Contact Info: {current_section}
    - Section Instructions: {section_instructions}

    **YOUR MISSION:**
    Optimize contact information for maximum professionalism and ATS compatibility.

    **OPTIMIZATION RULES:**
    1. **Completeness**: Ensure all required elements are present
    2. **Professional Format**: Use industry-standard formatting
    3. **ATS Compatibility**: Ensure format is machine-readable
    4. **Missing Elements**: Add any missing professional contact elements from instructions

    **OUTPUT FORMAT(STRICT JSON):**
    Return ONLY valid JSON in the EXACT format below, with NO extra keys, NO markdown, and NO explanations:
    ```json
    {{
        "updated_section": {{
            "full_name": "Full Name",
            "email": "email@domain.com",
            "phone": "Phone Number",
            "location": "City, State",
            "linkedin": "LinkedIn URL or null"
        }},
        "improvements": ["List of improvements made"],
        "validation_notes": ["Professional formatting notes"]
    }}
    ```""",
            input_variables=["current_section", "section_instructions"],
        )

    def _get_certifications_agent_prompt(self) -> PromptTemplate:
        """Specialized agent for certifications optimization"""
        logger.debug("Creating specialized agent for certifications section optimization...")
        return PromptTemplate(
            template="""You are an expert Certifications Specialist focused on highlighting relevant certifications that support job candidacy.

    **INPUTS:**
    - Current Certifications: {current_section}
    - Section Instructions: {section_instructions}
    - Job Description: {jd_structured}

    **YOUR MISSION:**
    Optimize certifications section to highlight most relevant credentials and address missing certifications identified in feedback.

    **OPTIMIZATION RULES:**
    1. **Relevance Priority**: Prioritize certifications most relevant to job requirements
    2. **Missing Additions**: Note missing certifications from JD (but don't fabricate)
    3. **Emphasis Strategy**: Highlight certifications to emphasize per instructions
    4. **Professional Format**: Ensure clean, professional presentation

    **OUTPUT FORMAT(STRICT JSON):**
    Return ONLY valid JSON in the EXACT format below, with NO extra keys, NO markdown, and NO explanations:
    ```json
    {{
        "updated_section": [
            {{
                "name": "Certification Name",
                "date": "MM/YYYY or null"
            }}
        ],
        "improvements": ["List of improvements made"],
        "validation_notes": ["Notes on relevance and emphasis"]
    }}
    ```""",
            input_variables=["current_section", "section_instructions", "jd_structured"],
        )

    def _get_education_agent_prompt(self) -> PromptTemplate:
        """Specialized agent for education section optimization"""
        logger.debug("Creating specialized agent for education section optimization...")
        return PromptTemplate(
            template="""You are an expert Education Section Specialist focused on presenting educational background in the most professionally relevant manner.

    **INPUTS:**
    - Current Education: {current_section}
    - Section Instructions: {section_instructions}
    - Job Description: {jd_structured}

    **YOUR MISSION:**
    Optimize education section to ensure perfect formatting alignment with job requirements, and maximize ATS compatibility.

    **STRICT RULES:**
    1. **NO HALLUCINATION:** If the original resume has NO education details (i.e., {current_section} is empty or null), DO NOT add, infer, or hallucinate any education entries. Return an empty list for "updated_section".
    2. **NO FABRICATION:** Only use education details present in the original resume. Do NOT create new institutions, degrees, or dates.
    3. **Professional Format:** Ensure clean, professional presentation for existing entries.
    4. **JD Alignment:** Emphasize educational elements that support job candidacy, but ONLY if present in the original resume.

    **OUTPUT FORMAT (STRICT JSON):**
    Return ONLY valid JSON in the EXACT format below, with NO extra keys, NO markdown, and NO explanations.

    If education is present:
    ```json
    {{
        "updated_section": [
            {{
                "institution": "Institution Name or null",
                "degree": "Degree Information",
                "location": "Location or null",
                "start_date": "MM/YYYY or null",
                "end_date": "MM/YYYY or null"
            }}
        ],
        "improvements": ["List of improvements made"],
        "validation_notes": ["Notes on relevance and emphasis"]
    }}
    ```
    If education is NOT present:
    ```json
    {{
        "updated_section": [],
        "improvements": [],
        "validation_notes": ["No education details present in original resume; no entries added."]
    }}
    ```
    """,
            input_variables=["current_section", "section_instructions", "jd_structured"],
        )

    def _build_temporal_context(self, resume: dict) -> dict:
        logger.debug("Building temporal context for resume...")
        experience = resume.get("professional_experience", [])
        total_months = 0
        career_start = None
        career_end = None
        technology_timeline = {}

        for exp in experience:
            start_date_str = exp.get("start_date", "")
            end_date_str = exp.get("end_date", "Present")
            try:
                start_date = parse(start_date_str)
            except Exception:
                start_date = None
            try:
                end_date = parse(end_date_str) if end_date_str and end_date_str != "Present" else datetime.now()
            except Exception:
                end_date = datetime.now()

            # Update career start/end
            if start_date and (not career_start or start_date < career_start):
                career_start = start_date
            if end_date and (not career_end or end_date > career_end):
                career_end = end_date
            # Calculate months for this experience
            if start_date and end_date:
                months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                total_months += max(0, months)

            # Technology timeline
            for responsibility in exp.get("responsibilities", []):
                tech_keywords = [
                    "Python",
                    "JavaScript",
                    "React",
                    "AWS",
                    "Docker",
                    "Kubernetes",
                    "AI",
                    "ML",
                    "LangChain",
                    "LlamaIndex",
                ]
                for tech in tech_keywords:
                    if tech.lower() in responsibility.lower():
                        year = start_date.year if start_date else datetime.now().year
                        if tech not in technology_timeline:
                            technology_timeline[tech] = {"first_used": year, "last_used": year}
                        else:
                            technology_timeline[tech]["first_used"] = min(technology_timeline[tech]["first_used"], year)
                            technology_timeline[tech]["last_used"] = max(technology_timeline[tech]["last_used"], year)
        return {
            "total_experience_years": total_months // 12 if total_months else 0,
            "career_start": career_start.strftime("%Y-%m-%d") if career_start else None,
            "career_end": career_end.strftime("%Y-%m-%d") if career_end else None,
            "technology_timeline": technology_timeline,
            "current_year": datetime.now().year,
        }

    def _integrate_section_results(self, original_resume: dict, section_results: dict, ats_feedback: dict) -> dict:
        """Integrate all section results into a cohesive resume"""
        logger.debug("Integrating section results into final resume...")
        integrated_resume = original_resume.copy()

        # Apply successful section updates
        for section_name, result in section_results.items():
            if result["status"] == "success":
                integrated_resume[section_name] = result["data"]

        # Ensure structural integrity
        integrated_resume = self._validate_resume_structure(integrated_resume)

        return integrated_resume

    def _validate_resume_structure(self, resume: dict) -> dict:
        """Validate and correct resume structure to match Resume model"""
        logger.debug("Validating resume structure...")
        # Ensure all required fields exist
        required_fields = [
            "summary",
            "contact_info",
            "skills",
            "certifications",
            "professional_experience",
            "education",
            "projects",
        ]

        for field in required_fields:
            if field not in resume:
                resume[field] = self._get_default_field_value(field)

        # Validate contact_info structure
        if not isinstance(resume["contact_info"], dict):
            resume["contact_info"] = {"full_name": "", "email": "", "phone": "", "location": ""}

        # Ensure skills is list of SkillCategory objects
        if not isinstance(resume["skills"], list):
            resume["skills"] = []

        return resume

    def _get_default_field_value(self, field_name: str):
        """Get default value for missing resume fields"""
        logger.debug(f"Getting default value for {field_name}...")
        defaults = {
            "summary": "",
            "contact_info": {"full_name": "", "email": "", "phone": "", "location": ""},
            "skills": [],
            "certifications": [],
            "professional_experience": [],
            "education": [],
            "projects": [],
        }
        return defaults.get(field_name, "")

    def _final_coherence_validation(self, integrated_resume: dict, jd_structured: dict, ats_feedback: dict) -> dict:
        """Final validation pass to ensure coherence and quality.

        Placeholder hook — temporal / keyword / formatting validators were
        removed in the dead-code cleanup. Re-introduce validators here when
        needed.
        """
        logger.debug("Performing final coherence validation on integrated resume...")
        return integrated_resume

    def _should_rewrite(self, state: AgentState) -> Literal["rewrite", "complete"]:
        logger.debug("Determining whether to rewrite resume based on ATS feedback and iterations...")
        if state["status"] == "error":
            return "complete"
        return (
            "rewrite"
            if state["ats_feedback"]["score"] < TARGET_SCORE and state["iterations"] < MAX_REWRITE_ATTEMPTS
            else "complete"
        )

    def _error_handler(self, state: AgentState) -> dict:
        logger.error(f"Pipeline failed at state: {state}")
        return {"status": "error"}

    async def optimize(self, resume_text: str, job_description: str) -> dict:
        logger.debug("Starting resume optimization process...")
        initial_state = {
            "raw_resume_text": resume_text,
            "job_description": job_description,
            "jd_structured": None,
            "parsed_resume": None,
            "current_resume": None,
            "ats_feedback": {},
            "ats_feedback_history": [],
            "iterations": 0,
            "initial_ats_score": None,
            "final_ats_score": None,
            "status": "parsing",
        }
        result = await self.graph.ainvoke(initial_state)
        return {
            "initial_score": result["initial_ats_score"],
            "final_score": result["final_ats_score"],
            "parsed_resume": result["parsed_resume"],
            "optimized_resume": json.loads(result["current_resume"]),
            "ats_feedback": result["ats_feedback"],
            "ats_feedback_history": result["ats_feedback_history"],
            "iterations": result["iterations"],
            "status": result["status"],
        }

    async def optimize_from_parsed(self, parsed_resume: dict, job_description: str) -> dict:
        """Optimize resume using pre-parsed resume data, skipping parsing step"""
        logger.debug("Starting resume optimization from pre-parsed data...")
        initial_state = {
            "raw_resume_text": json.dumps(parsed_resume),  # Not used but required
            "job_description": job_description,
            "jd_structured": None,
            "parsed_resume": parsed_resume,
            "current_resume": json.dumps(parsed_resume),
            "ats_feedback": {},
            "ats_feedback_history": [],
            "iterations": 0,
            "initial_ats_score": None,
            "final_ats_score": None,
            "status": "scoring",  # Skip parsing, go directly to scoring
        }

        # Start from JD parsing since resume is already parsed
        result = await self.graph_parsed.ainvoke(initial_state, {"configurable": {"thread_id": "optimize_parsed"}})

        return {
            "initial_score": result["initial_ats_score"],
            "final_score": result["final_ats_score"],
            "parsed_resume": result["parsed_resume"],
            "optimized_resume": json.loads(result["current_resume"]),
            "ats_feedback": result["ats_feedback"],
            "ats_feedback_history": result["ats_feedback_history"],
            "iterations": result["iterations"],
            "status": result["status"],
        }
