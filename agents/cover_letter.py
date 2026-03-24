from crewai import Agent

def create_cover_letter_writer(llm):
    return Agent(
        role="Cover Letter Writer",
        goal="Write a personalised cover letter connecting the candidate's achievements to the job requirements.",
        backstory=(
            "You are a storytelling expert who writes cover letters that hiring managers "
            "actually read. You work purely from the context provided — no tools needed."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )