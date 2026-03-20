from crewai import Agent

def create_cold_email_writer(llm):
    return Agent(
        role="Recruiter Outreach Specialist",
        goal="Write a short confident cold email to the recruiter that gets a reply.",
        backstory=(
            "You write cold emails under 100 words that get responses. "
            "You work purely from the context provided — no tools needed."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )