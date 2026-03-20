from crewai import Agent

def create_manager(llm):
    return Agent(
        role="Application Manager",
        goal="Orchestrate all sub-agents to produce a complete high-quality job application package.",
        backstory=(
            "You are a senior career strategist with 15 years of experience helping "
            "candidates land jobs at top companies."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=True
    )