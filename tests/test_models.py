"""Tests for ``app.models`` — pydantic schema validators.

We focus on the one validator with real logic:
``ExperienceItem.filter_environment_lines`` — a ``@field_validator`` that
strips environment/tech-stack lines from responsibility bullets.

Without this filter, resume parses would include noise lines like
"Environment: Java, Spring Boot, AWS" mixed in with real accomplishments,
degrading downstream ATS scoring.
"""

import pytest
from app.models import ContactInfo, ExperienceItem, Resume
from pydantic import ValidationError


class TestExperienceItemFilter:
    def _make(self, bullets):
        return ExperienceItem(
            title="SWE",
            company="ACME",
            start_date="2023-01",
            end_date="2024-01",
            responsibilities=bullets,
        )

    def test_environment_line_filtered(self):
        item = self._make(
            [
                "Shipped feature X",
                "Environment: Java, Spring Boot, AWS",
                "Led team of 3",
            ]
        )
        assert item.responsibilities == ["Shipped feature X", "Led team of 3"]

    def test_tech_stack_line_filtered(self):
        item = self._make(["Built API", "Tech Stack: Python, FastAPI"])
        assert item.responsibilities == ["Built API"]

    def test_tools_line_filtered(self):
        item = self._make(["Did work", "Tools: IntelliJ, Docker"])
        assert item.responsibilities == ["Did work"]

    def test_technologies_line_filtered(self):
        item = self._make(["Delivered", "Technologies: K8s, Terraform"])
        assert item.responsibilities == ["Delivered"]

    def test_case_insensitive_filtering(self):
        item = self._make(
            [
                "Built",
                "ENVIRONMENT: Java",
                "tech stack: Python",
                "TOOLS: Docker",
            ]
        )
        assert item.responsibilities == ["Built"]

    def test_leading_whitespace_still_filtered(self):
        item = self._make(["Kept", "   Environment: Java"])
        assert item.responsibilities == ["Kept"]

    def test_no_environment_lines_all_preserved(self):
        bullets = [
            "Led cross-functional team",
            "Reduced latency by 40%",
            "Shipped MVP in 3 weeks",
        ]
        item = self._make(bullets)
        assert item.responsibilities == bullets

    def test_environment_substring_not_at_start_preserved(self):
        # Line mentioning "environment" mid-sentence should survive.
        item = self._make(["Improved deployment environment for 5 services"])
        assert item.responsibilities == ["Improved deployment environment for 5 services"]

    def test_all_filtered_yields_empty_list(self):
        item = self._make(["Environment: X", "Tools: Y"])
        assert item.responsibilities == []

    def test_empty_bullets(self):
        item = self._make([])
        assert item.responsibilities == []

    def test_non_list_input_passes_through(self):
        # Validator short-circuits on non-list input; pydantic then rejects.
        with pytest.raises(ValidationError):
            ExperienceItem(
                title="X",
                company="Y",
                start_date="2023",
                responsibilities="not a list",
            )


class TestContactInfo:
    def test_valid_contact_info(self):
        ci = ContactInfo(
            full_name="Jane Doe",
            email="jane@example.com",
            phone="555-1234",
        )
        assert ci.full_name == "Jane Doe"
        assert ci.location is None

    def test_invalid_email_rejected(self):
        with pytest.raises(ValidationError):
            ContactInfo(full_name="X", email="not-an-email", phone="1")

    def test_optional_fields_default_none(self):
        ci = ContactInfo(full_name="X", email="x@y.com", phone="1")
        assert ci.location is None
        assert ci.linkedin is None


class TestResumeSchema:
    def test_minimal_valid_resume(self):
        resume = Resume(
            summary="Experienced SWE.",
            contact_info=ContactInfo(full_name="X", email="x@y.com", phone="1"),
            skills=[],
            certifications=[],
            professional_experience=[],
            education=[],
            projects=[],
        )
        assert resume.summary == "Experienced SWE."
        assert resume.skills == []

    def test_missing_required_fields_rejected(self):
        with pytest.raises(ValidationError):
            Resume(
                summary="x",
                contact_info=ContactInfo(full_name="X", email="x@y.com", phone="1"),
                # missing: skills, certifications, experience, education, projects
            )
