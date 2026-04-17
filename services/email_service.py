import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import config


def send_result_email(to_email: str, download_url: str, job_id: str):
    """Send an HTML email with the processed video download link via Gmail SMTP."""

    if not config.SMTP_USER or not config.SMTP_PASSWORD:
        print(f"[EMAIL] SMTP credentials not set. Skipping email to {to_email}")
        print(f"[EMAIL] Download link would be: {download_url}")
        return

    subject = f"Your processed video is ready! (Job {job_id})"

    html_body = f"""\
    <html>
    <body style="font-family: 'Segoe UI', Arial, sans-serif; background: #0f0f23; color: #e0e0e0; padding: 40px;">
      <div style="max-width: 560px; margin: 0 auto; background: linear-gradient(135deg, #1a1a3e, #16213e); border-radius: 16px; padding: 40px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);">
        <h1 style="color: #818cf8; margin-top: 0; font-size: 24px;">🎬 Video Processing Complete</h1>
        <p style="line-height: 1.7; color: #c0c0d0;">
          Great news! Your video has been processed through all three stages:
        </p>
        <ul style="line-height: 2; color: #a0a0c0;">
          <li>✅ Video Stabilization</li>
          <li>✅ Object Tracking</li>
          <li>✅ CSV Postprocessing</li>
        </ul>
        <p style="line-height: 1.7; color: #c0c0d0;">
          Click the button below to download your processed video:
        </p>
        <div style="text-align: center; margin: 32px 0;">
          <a href="{download_url}"
             style="background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; text-decoration: none; padding: 14px 32px; border-radius: 10px; font-weight: 600; font-size: 16px; display: inline-block;">
            ⬇️ Download Video
          </a>
        </div>
        <p style="font-size: 13px; color: #666; margin-top: 32px; border-top: 1px solid #2a2a4a; padding-top: 16px;">
          Job ID: {job_id}<br>
          This is an automated message from TTGUI Video Processor.
        </p>
      </div>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = config.SMTP_USER
    msg["To"] = to_email

    # Plain-text fallback
    text_body = (
        f"Your video (Job {job_id}) has been processed!\n\n"
        f"Download it here: {download_url}\n\n"
        f"Stages completed: Stabilization, Object Detection, Object Tracking."
    )
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(config.SMTP_USER, config.SMTP_PASSWORD)
        server.sendmail(config.SMTP_USER, to_email, msg.as_string())

    print(f"[EMAIL] Sent result email to {to_email}")
