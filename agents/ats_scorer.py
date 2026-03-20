from crewai import Agent

def create_ats_scorer(llm):
    return Agent(
        role="ATS Compatibility Scorer",
        goal="Compare candidate profile against job description, calculate ATS match percentage, and list gaps.",
        backstory=(
            "You are an ATS systems expert. You calculate keyword match rates and identify "
            "exactly what is missing. You work only from context — no external tools needed."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )