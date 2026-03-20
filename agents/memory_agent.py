from crewai import Agent
from tools.chroma_store import SaveApplicationTool, RetrieveSimilarTool

def create_memory_agent(llm):
    return Agent(
        role="Application Memory Manager",
        goal="Store completed applications to ChromaDB and retrieve similar past applications for context.",
        backstory=(
            "You manage the long-term memory of the application system using ChromaDB."
        ),
        llm=llm,
        tools=[SaveApplicationTool(), RetrieveSimilarTool()],
        verbose=True,
        allow_delegation=False
    )