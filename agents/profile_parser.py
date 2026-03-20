from crewai import Agent

def create_profile_parser(llm, tools):
    return Agent(
        role="Candidate Profile Parser",
        goal="Parse the candidate's resume PDF and LinkedIn profile into a clean, structured profile containing all experience, skills, achievements, and education.",
        backstory=(
            "You are a professional resume consultant who can read any resume format and "
            "extract structured, usable information. You know how to merge data from multiple "
            "sources like PDFs and LinkedIn profiles into a single comprehensive candidate profile."
        ),
        llm=llm,
        tools=tools,
        verbose=True,
        allow_delegation=False
    )