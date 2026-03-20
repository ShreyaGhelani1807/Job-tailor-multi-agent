from crewai import Agent

def create_interview_prep_agent(llm):
    return Agent(
        role="Interview Preparation Coach",
        goal="Generate 10 role-specific interview questions with model answers from the candidate's experience.",
        backstory=(
            "You are a veteran interview coach. You generate questions specific to the JD "
            "and craft answers using the candidate's actual experience."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )