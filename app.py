from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

from utils.resume_parser import extract_text_from_pdf
from utils.matcher import extract_skills, match_jobs, skill_gap_analysis


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    if "resume" not in request.files:
        return "No resume uploaded"

    file = request.files["resume"]

    if file.filename == "":
        return "Please select a resume"

    if not file.filename.lower().endswith(".pdf"):
        return "Please upload a PDF file"

    filename = secure_filename(file.filename)

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(file_path)

    # Extract resume text
    resume_text = extract_text_from_pdf(file_path)

    # Extract skills
    detected_skills = extract_skills(resume_text)

    # Match resume with jobs
    matched_jobs = match_jobs(resume_text)

    # Convert top jobs to records
    job_results = matched_jobs.head(5).to_dict("records")

    # Analyze skill gaps for each recommended job
    for job in job_results:

        matching_skills, missing_skills = skill_gap_analysis(
            detected_skills,
            job["required_skills"]
        )

        job["matching_skills"] = matching_skills
        job["missing_skills"] = missing_skills

    # Get the top recommended job
    top_job = job_results[0] if job_results else None

    return render_template(
        "index.html",
        resume_text=resume_text,
        detected_skills=detected_skills,
        job_results=job_results,
        top_job=top_job
    )


if __name__ == "__main__":
    app.run()