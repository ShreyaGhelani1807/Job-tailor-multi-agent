from crewai import Agent

def create_jd_analyser(llm):
    return Agent(
        role="Job Description Analyser",
        goal="Extract required skills, keywords, responsibilities, and ATS terms from a job description.",
        backstory=(
            "You are an expert at reading job postings and identifying exactly what "
            "employers are looking for. You never use external tools — you work purely "
            "from the text provided to you."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )