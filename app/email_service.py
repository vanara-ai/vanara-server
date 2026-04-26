"""Optional email delivery for user feedback.

All SMTP_* env vars are optional. If any of SMTP_SERVER, SMTP_PORT,
SMTP_USERNAME, SMTP_PASSWORD is missing the service simply reports that
email is disabled and nothing is sent — the endpoint caller continues
normally (feedback is still stored in Supabase if configured).
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from .logger import logger


class EmailService:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER") or None
        raw_port = os.getenv("SMTP_PORT")
        try:
            self.smtp_port = int(raw_port) if raw_port else None
        except ValueError:
            logger.warning("Invalid SMTP_PORT; email disabled", extra={"value": raw_port})
            self.smtp_port = None
        self.smtp_username = os.getenv("SMTP_USERNAME") or None
        self.smtp_password = os.getenv("SMTP_PASSWORD") or None
        self.from_email = os.getenv("FROM_EMAIL") or None
        self.to_email = os.getenv("TO_EMAIL") or None

    @property
    def enabled(self) -> bool:
        return all(
            [
                self.smtp_server,
                self.smtp_port,
                self.smtp_username,
                self.smtp_password,
                self.from_email,
                self.to_email,
            ]
        )

    async def send_feedback_email(self, category, message, user_email=None, user_name=None):
        if not self.enabled:
            logger.info("Email delivery disabled (SMTP vars not set); skipping send")
            return {"success": False, "message": "Email delivery not configured"}

        try:
            msg = MIMEMultipart()
            msg["From"] = self.from_email
            msg["To"] = self.to_email

            category_labels = {
                "feedback": "User Feedback",
                "issue": "Bug Report",
                "feature": "Feature Request",
            }
            msg["Subject"] = f"Vanara.ai: {category_labels.get(category, 'Feedback')}"

            body = f"Category: {category_labels.get(category, category)}\n\n"
            if user_name:
                body += f"From: {user_name}\n"
            if user_email:
                body += f"Email: {user_email}\n"
            body += f"\nMessage:\n{message}\n"

            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            logger.info(
                "Feedback email sent",
                extra={"category": category, "user_email": user_email},
            )
            return {"success": True, "message": "Feedback sent successfully"}

        except Exception as e:
            logger.error(
                "Failed to send feedback email",
                extra={"error": str(e), "category": category},
            )
            return {"success": False, "message": f"Failed to send email: {e}"}
