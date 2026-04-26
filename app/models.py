from enum import StrEnum

from pydantic import BaseModel, EmailStr, Field, field_validator


class ContactInfo(BaseModel):
    full_name: str
    email: EmailStr
    phone: str
    location: str | None = None
    linkedin: str | None = None


class SkillCategory(BaseModel):
    category: str = Field(
        ..., description="The category of skills, e.g., Programming Languages, Framework and Libraries "
    )
    skills: list[str]


class Certification(BaseModel):
    name: str
    date: str | None = None  # Optional date field, e.g., "08/2024"


class ExperienceItem(BaseModel):
    title: str
    company: str
    location: str | None = None
    start_date: str
    end_date: str | None = None
    responsibilities: list[str] = Field(
        ...,
        description="List of experience/action/achievement bullet points for this role. Do NOT include tech stack, environment, or non-relevant lines. Only include true responsibilities and accomplishments.",
    )

    @field_validator("responsibilities", mode="before")
    def filter_environment_lines(cls, v):
        if not isinstance(v, list):
            return v
        filtered = []
        for line in v:
            # Heuristic: skip lines that start with 'Environment:', 'Tech Stack:', etc.
            if isinstance(line, str) and (
                line.strip().lower().startswith("environment:")
                or line.strip().lower().startswith("tech stack:")
                or line.strip().lower().startswith("tools:")
                or line.strip().lower().startswith("technologies:")
            ):
                continue
            filtered.append(line)
        return filtered


class EducationItem(BaseModel):
    institution: str | None = Field(
        None,
        description="The name of the educational institution (e.g., 'Stanford University'). Do not fill with placeholders if the content is not available.",
    )
    degree: str = Field(
        ...,
        description="The degree of the education item e.g., Bachelor's: Computer Science, Master of Science: Information Technology, etc.",
    )
    location: str | None = Field(
        None,
        description="The physical location of the institution (e.g., 'Stanford, CA'). Do not fill with placeholders if the content is not available.",
    )
    start_date: str | None = None
    end_date: str | None = None


class ProjectItem(BaseModel):
    title: str
    description: str


class Resume(BaseModel):
    summary: str | None = ""
    contact_info: ContactInfo
    skills: list[SkillCategory]
    certifications: list[Certification]
    professional_experience: list[ExperienceItem]
    education: list[EducationItem]
    projects: list[ProjectItem]

    @field_validator("summary", mode="before")
    def coerce_summary_none(cls, v):
        return v or ""


class SummaryInstructions(BaseModel):
    current_issues: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    missing_value_propositions: list[str] = Field(default_factory=list)
    actionable_steps: list[str] = Field(default_factory=list)
    target_length: str = Field(default="")
    tone_guidance: str = Field(default="")


class SkillsInstructions(BaseModel):
    missing_technical_skills: list[str] = Field(default_factory=list)
    missing_soft_skills: list[str] = Field(default_factory=list)
    # skills_to_emphasize: List[str] = Field(default_factory=list)
    categorization_improvements: list[str] = Field(default_factory=list)
    actionable_steps: list[str] = Field(default_factory=list)
    # priority_order: List[str] = Field(default_factory=list)


class JobSpecificInstruction(BaseModel):
    job_title: str
    missing_keywords: list[str] = Field(default_factory=list)
    responsibilities_to_add: list[str] = Field(default_factory=list)
    responsibilities_to_modify: list[str] = Field(default_factory=list)
    quantification_needed: list[str] = Field(default_factory=list)
    achievements_to_highlight: list[str] = Field(default_factory=list)
    action_verbs_to_use: list[str] = Field(default_factory=list)


class ProfessionalExperienceInstructions(BaseModel):
    job_specific_instructions: list[JobSpecificInstruction] = Field(default_factory=list)
    overall_improvements: list[str] = Field(default_factory=list)
    experience_ordering: list[str] = Field(default_factory=list)
    actionable_steps: list[str] | None = None


class EducationInstructions(BaseModel):
    formatting_requirements: list[str] | None = Field(
        default=None, description="Formatting requirements derived from job description"
    )
    actionable_steps: list[str] | None = None


class ProjectSpecificInstruction(BaseModel):
    project_name: str
    alignment_issues: list[str] = Field(default_factory=list)
    missing_technologies: list[str] = Field(default_factory=list)
    missing_outcomes: list[str] | None = None
    description_improvements: list[str] = Field(default_factory=list)


class ProjectsInstructions(BaseModel):
    project_specific_instructions: list[ProjectSpecificInstruction] = Field(default_factory=list)
    projects_to_add: list[str] = Field(default_factory=list)
    projects_to_emphasize: list[str] = Field(default_factory=list)
    overall_improvements: list[str] = Field(default_factory=list)
    actionable_steps: list[str] = Field(default_factory=list)


class SectionInstructions(BaseModel):
    summary: SummaryInstructions | None = None
    # contact_info: ContactInfoInstructions | None = None
    skills: SkillsInstructions | None = None
    # certifications: CertificationsInstructions | None = None
    professional_experience: ProfessionalExperienceInstructions | None = None
    education: EducationInstructions | None = None
    projects: ProjectsInstructions | None = None


class ScoreBreakdown(BaseModel):
    keywords_score: float = Field(..., description="Score for keyword matching (0-40 points)")
    keywords_reasoning: str = Field(..., description="Detailed explanation of keyword matching score")
    skills_score: float = Field(..., description="Score for skills alignment (0-25 points)")
    skills_reasoning: str = Field(..., description="Detailed explanation of skills score")
    experience_score: float = Field(..., description="Score for experience relevance (0-25 points)")
    experience_reasoning: str = Field(..., description="Detailed explanation of experience score")
    education_score: float = Field(..., description="Score for education alignment (0-10 points)")
    education_reasoning: str = Field(..., description="Detailed explanation of education score")


class ATSScoreOutput(BaseModel):
    score: float  # 0-100
    summary: str  # Brief explanation of the score
    score_breakdown: ScoreBreakdown = Field(..., description="Detailed breakdown of how the score was calculated")
    strengths: list[str]
    weaknesses: list[str]
    section_instructions: SectionInstructions = Field(default_factory=SectionInstructions)
    model_config = {"extra": "ignore"}


# Database Models
class GeneratePDFRequest(BaseModel):
    resume_generation_id: str
    resume_template: str = "resume_template_7.html"


class FeedbackCategory(StrEnum):
    FEEDBACK = "feedback"
    ISSUE = "issue"
    FEATURE = "feature"


class FeedbackRequest(BaseModel):
    category: FeedbackCategory
    message: str
    user_email: EmailStr | None = None
    user_name: str | None = None


class SkillsGroup(BaseModel):
    technical_skills: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    domain_skills: list[str] = Field(default_factory=list)


class ExperienceRequirements(BaseModel):
    minimum_years: str | None = ""
    domain_specific: list[str] = Field(default_factory=list)
    conditional_requirements: list[str] = Field(default_factory=list)
    alternative_paths: list[str] = Field(default_factory=list)


class EducationRequirements(BaseModel):
    degree_requirements: list[str] = Field(default_factory=list)
    alternative_experience: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


class CompanyInfo(BaseModel):
    industry: str | None = ""


class JD(BaseModel):
    target_job_title: str = Field(default="")
    required_skills: SkillsGroup = Field(default_factory=SkillsGroup)
    preferred_skills: SkillsGroup = Field(default_factory=SkillsGroup)
    experience_requirements: ExperienceRequirements = Field(default_factory=ExperienceRequirements)
    education_requirements: EducationRequirements = Field(default_factory=EducationRequirements)
    key_responsibilities: list[str] = Field(default_factory=list)
    company_info: CompanyInfo = Field(default_factory=CompanyInfo)
