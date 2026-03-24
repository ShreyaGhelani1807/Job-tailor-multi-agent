from crewai import Agent
from tools.pdf_parser import PDFParserTool

def create_profile_parser(llm, tools=None):
    tools = tools or [PDFParserTool()]
    return Agent(
        role="Candidate Profile Parser",
        goal="Parse the candidate resume PDF into a structured profile with skills, experience, and education.",
        backstory=(
            "You are a professional resume consultant who extracts structured information "
            "from resumes. You use the PDF Parser tool to read the resume file."
        ),
        llm=llm,
        tools=tools,
        verbose=True,
        allow_delegation=False
    )