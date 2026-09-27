import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# SKILLS
# ============================================================

SKILLS = [

    # Programming
    "Python",
    "Java",
    "C++",
    "SQL",
    "JavaScript",
    "C",

    # Web Development
    "HTML",
    "CSS",
    "React",
    "Node.js",
    "Flask",
    "Django",
    "REST API",

    # Databases
    "MySQL",
    "MongoDB",
    "SQL Server",
    "Oracle",

    # Data & AI
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "Machine Learning",
    "Deep Learning",
    "Artificial Intelligence",
    "Data Science",
    "Data Analysis",
    "Data Analytics",

    # Business / Analytics
    "Excel",
    "Power BI",
    "Tableau",

    # Cloud
    "AWS",
    "Azure",
    "Google Cloud",
    "Cloud Computing",
    "Cloud Services",

    # Development Tools
    "Git",
    "GitHub",
    "Docker",

    # Enterprise Technologies
    "Salesforce",
    "AEM",
    "AEM Assets",
    "AEM Developer",

    # Security / Biometrics
    "Face Recognition",
    "Facial Recognition",
    "Biometrics",
    "Computer Vision",
    "Authentication",
    "Biometric Authentication"
]


# ============================================================
# SKILL ALIASES
# ============================================================

SKILL_ALIASES = {

    "Face Recognition": [
        "face recognition",
        "facial recognition",
        "face recognizing",
        "face recognizing system",
        "face and biometric",
        "face biometric"
    ],

    "Facial Recognition": [
        "facial recognition",
        "face recognition",
        "face recognizing"
    ],

    "Biometrics": [
        "biometric",
        "biometrics",
        "biometric-based",
        "biometric based",
        "biometric traits",
        "biometric sensors",
        "biometric sensor",
        "face and biometric"
    ],

    "Biometric Authentication": [
        "biometric authentication",
        "biometric-based authentication",
        "biometric based authentication",
        "biometric authentication protocol"
    ],

    "Cloud Services": [
        "cloud services",
        "cloud service",
        "cloud server",
        "cloud-based services",
        "cloud based services"
    ],

    "Cloud Computing": [
        "cloud computing",
        "cloud services",
        "cloud service",
        "cloud server",
        "cloud-based services",
        "cloud based services"
    ],

    "Authentication": [
        "authentication",
        "authenticated",
        "authentication protocol"
    ],

    "AEM Developer": [
        "aem developer",
        "aem development"
    ],

    "Data Analytics": [
        "data analytics",
        "data analytic",
        "analytics"
    ],

    "Data Analysis": [
        "data analysis",
        "data analytics",
        "data analyst"
    ],

    "Salesforce": [
        "salesforce",
        "salesforce developer",
        "salesforce development"
    ]
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    # Convert hyphens and underscores to spaces
    text = re.sub(r"[-_]", " ", text)

    # Keep letters, numbers, + and #
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# PHRASE CHECK
# ============================================================

def phrase_exists(text, phrase):

    phrase = normalize_text(phrase)

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(phrase)
        + r"(?![a-z0-9])"
    )

    return re.search(pattern, text) is not None


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(resume_text):

    detected_skills = []

    text = normalize_text(resume_text)

    for skill in SKILLS:

        # Check original skill name
        if phrase_exists(text, skill):
            detected_skills.append(skill)
            continue

        # Check aliases
        aliases = SKILL_ALIASES.get(skill, [])

        for alias in aliases:

            if phrase_exists(text, alias):
                detected_skills.append(skill)
                break

    return detected_skills


# ============================================================
# SKILL NORMALIZATION FOR MATCHING
# ============================================================

def normalize_skill(skill):

    skill = normalize_text(skill)

    # Treat equivalent skills as the same for matching
    aliases = {

        "cloud service": "cloud computing",
        "cloud services": "cloud computing",
        "cloud server": "cloud computing",

        "data analysis": "data analytics",

        "facial recognition": "face recognition",

        "face recognizing": "face recognition",

        "biometric": "biometrics",

        "biometric based authentication":
            "biometric authentication",

        "biometric based":
            "biometrics"
    }

    return aliases.get(skill, skill)


# ============================================================
# SKILL MATCH SCORE
# ============================================================

def calculate_skill_match(resume_skills, required_skills):

    required_list = [
        skill.strip()
        for skill in required_skills.split(",")
        if skill.strip()
    ]

    if not required_list:
        return 0

    resume_skills_normalized = {
        normalize_skill(skill)
        for skill in resume_skills
    }

    matched_count = 0

    for skill in required_list:

        normalized_required = normalize_skill(skill)

        if normalized_required in resume_skills_normalized:
            matched_count += 1

    skill_score = (
        matched_count / len(required_list)
    ) * 100

    return skill_score


# ============================================================
# JOB MATCHING
# ============================================================

def match_jobs(resume_text):

    jobs = pd.read_csv("data/jobs.csv")

    # --------------------------------------------------------
    # Extract resume skills
    # --------------------------------------------------------

    resume_skills = extract_skills(resume_text)

    # --------------------------------------------------------
    # Combine required skills + description
    # --------------------------------------------------------

    jobs["job_text"] = (
        jobs["required_skills"].fillna("")
        + " "
        + jobs["description"].fillna("")
    )

    job_texts = jobs["job_text"].tolist()

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    documents = [resume_text] + job_texts

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity_scores = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    )[0]

    text_scores = similarity_scores * 100

    # --------------------------------------------------------
    # Calculate scores
    # --------------------------------------------------------

    final_scores = []
    skill_scores = []
    similarity_score_list = []

    for index, job in jobs.iterrows():

        # Skill score
        skill_score = calculate_skill_match(
            resume_skills,
            job["required_skills"]
        )

        # Text similarity
        text_score = text_scores[index]

        # Final score
        final_score = (
            0.70 * skill_score
            + 0.30 * text_score
        )

        skill_scores.append(skill_score)

        similarity_score_list.append(
            text_score
        )

        final_scores.append(
            final_score
        )

    # --------------------------------------------------------
    # Store scores
    # --------------------------------------------------------

    jobs["skill_match_score"] = skill_scores

    jobs["text_similarity_score"] = (
        similarity_score_list
    )

    jobs["match_score"] = final_scores

    # Remove temporary column
    jobs = jobs.drop(
        columns=["job_text"]
    )

    # Sort jobs
    jobs = jobs.sort_values(
        by="match_score",
        ascending=False
    )

    return jobs


# ============================================================
# SKILL GAP ANALYSIS
# ============================================================

def skill_gap_analysis(
    resume_skills,
    required_skills
):

    resume_skills_normalized = {
        normalize_skill(skill)
        for skill in resume_skills
    }

    required_skills_list = [
        skill.strip()
        for skill in required_skills.split(",")
        if skill.strip()
    ]

    matching_skills = []
    missing_skills = []

    for skill in required_skills_list:

        normalized_required = normalize_skill(skill)

        if normalized_required in resume_skills_normalized:

            matching_skills.append(skill)

        else:

            missing_skills.append(skill)

    return matching_skills, missing_skills