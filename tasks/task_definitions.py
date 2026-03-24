from crewai import Task

def create_tasks(agents, inputs):
    jd_text      = inputs.get("jd_text", "")
    jd_url       = inputs.get("jd_url", "")
    resume_path  = inputs.get("resume_path", "")
    linkedin_url = inputs.get("linkedin_url", "")

    jd_source = f"Job posting URL: {jd_url}" if jd_url else f"Job description:\n{jd_text}"
    profile_source = f"Resume PDF path: {resume_path}" if resume_path else "No resume provided."
    if linkedin_url:
        profile_source += f"\nLinkedIn URL: {linkedin_url}"

    retrieve_memory = Task(
        description=(
            f"Search ChromaDB for past applications similar to: {jd_text or jd_url}. "
            "Return up to 3 similar past applications. If none, return 'No past applications found.'"
        ),
        expected_output="Summary of similar past applications or 'No past applications found.'",
        agent=agents["memory"]
    )

    analyse_jd = Task(
        description=(
            f"Analyse this job description. DO NOT use any tools.\n\n{jd_source}\n\n"
            "Extract: job title, company name, required skills, preferred skills, "
            "key responsibilities, seniority level, and top 20 ATS keywords."
        ),
        expected_output="Structured analysis: job title, company, required skills, preferred skills, top 20 ATS keywords, seniority level.",
        agent=agents["jd_analyser"],
        context=[retrieve_memory]
    )

    parse_profile = Task(
        description=(
            f"Parse the candidate profile.\n{profile_source}\n\n"
            "IMPORTANT: You have access to the 'PDF Resume Parser' tool. "
            "Call this tool with the file_path to extract resume text, then build a structured profile with: name, summary, skills, experience, education, projects."
        ),
        expected_output="Structured candidate profile: Summary, Skills, Experience, Education, Projects.",
        agent=agents["profile_parser"],
        context=[retrieve_memory]
    )

    score_ats = Task(
        description=(
            "DO NOT use any tools. Using the JD analysis and candidate profile from context:\n"
            "1. Count how many top 20 ATS keywords appear in the candidate profile\n"
            "2. Calculate match percentage\n"
            "3. List missing keywords\n"
            "4. List matched keywords\n"
            "5. Give 5 specific recommendations to close the gaps"
        ),
        expected_output="ATS Score: X%. Missing: [list]. Matched: [list]. Recommendations: [numbered list].",
        agent=agents["ats_scorer"],
        context=[analyse_jd, parse_profile]
    )

    tailor_resume = Task(
        description=(
            "DO NOT use any tools. Using the JD analysis, candidate profile, and ATS report from context:\n"
            "- Rewrite the professional summary to mirror JD language\n"
            "- Reorder bullet points so most relevant achievements appear first\n"
            "- Weave in missing ATS keywords naturally\n"
            "- Quantify vague claims where possible\n"
            "- Output clean markdown format"
        ),
        expected_output="Complete tailored resume in clean markdown format.",
        agent=agents["resume_tailor"],
        context=[analyse_jd, parse_profile, score_ats]
    )

    write_cover_letter = Task(
        description=(
            "DO NOT use any tools. Using context only, write a personalised cover letter:\n"
            "- Opening: hook connecting to company mission\n"
            "- Body (2 paragraphs): map 2-3 achievements to JD requirements\n"
            "- Closing: confident call to action\n"
            "- Length: 250-320 words"
        ),
        expected_output="Complete cover letter in markdown, 250-320 words.",
        agent=agents["cover_letter_writer"],
        context=[analyse_jd, parse_profile, score_ats]
    )

    write_cold_email = Task(
        description=(
            "DO NOT use any tools. Using context only, write a cold recruiter email:\n"
            "- Subject line: under 8 words\n"
            "- Body: 3 sentences — who you are, why this role, call to action\n"
            "- Tone: peer-level, confident\n"
            "- Total: under 100 words"
        ),
        expected_output="Subject line on first line, then email body. Under 100 words.",
        agent=agents["cold_email_writer"],
        context=[analyse_jd, parse_profile]
    )

    prep_interview = Task(
        description=(
            "DO NOT use any tools. Using context only, generate interview prep:\n"
            "- 5 behavioural questions (STAR format) from JD responsibilities\n"
            "- 5 technical questions from JD skills\n"
            "- Model answer for each using candidate's real experience"
        ),
        expected_output="10 questions with answers. Q1: [question]\nA1: [answer] for all 10.",
        agent=agents["interview_prep"],
        context=[analyse_jd, parse_profile]
    )

    critique_outputs = Task(
        description=(
            "DO NOT use any tools. Review resume, cover letter, and cold email from context.\n"
            "Score each on (1-10):\n"
            "1. Keyword alignment\n"
            "2. Tone consistency\n"
            "3. ATS safety\n"
            "4. Specificity\n"
            "If any score below 7 give rewrite instructions. If all 7+ mark APPROVED."
        ),
        expected_output="Scores for all 3 outputs across 4 dimensions. APPROVED or rewrite instructions.",
        agent=agents["critic"],
        context=[tailor_resume, write_cover_letter, write_cold_email, score_ats]
    )

    save_application = Task(
        description=(
            "Save this application to ChromaDB memory using the Save Application tool.\n"
            "Store: company name, role, JD summary, ATS score, resume snippet, cover letter snippet."
        ),
        expected_output="Confirmation the application was saved with its document ID.",
        agent=agents["memory"],
        context=[analyse_jd, tailor_resume, write_cover_letter, score_ats]
    )

    return {
        "retrieve_memory":    retrieve_memory,
        "analyse_jd":         analyse_jd,
        "parse_profile":      parse_profile,
        "score_ats":          score_ats,
        "tailor_resume":      tailor_resume,
        "write_cover_letter": write_cover_letter,
        "write_cold_email":   write_cold_email,
        "prep_interview":     prep_interview,
        "critique_outputs":   critique_outputs,
        "save_application":   save_application,
    }