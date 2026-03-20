from crewai import Agent

def create_critic(llm):
    return Agent(
        role="Quality Critic",
        goal="Score resume, cover letter and cold email on 4 dimensions. Request rewrites if any score is below 7.",
        backstory=(
            "You are a brutally honest quality reviewer. You score outputs on keyword alignment, "
            "tone consistency, ATS safety, and specificity. You work from context only."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )