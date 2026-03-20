import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Route CrewAI's default OpenAI provider directly to Groq's OpenAI-compatible endpoint!
# OpenAI v1.x uses OPENAI_BASE_URL instead of OPENAI_API_BASE
os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY", "dummy-key")

import requests
from crewai import Crew, Process, LLM

from agents.jd_analyser      import create_jd_analyser
from agents.profile_parser   import create_profile_parser
from agents.ats_scorer       import create_ats_scorer
from agents.resume_tailor    import create_resume_tailor
from agents.cover_letter     import create_cover_letter_writer
from agents.cold_email       import create_cold_email_writer
from agents.interview_prep   import create_interview_prep_agent
from agents.critic           import create_critic
from agents.memory_agent     import create_memory_agent

from tasks.task_definitions import create_tasks

def build_llm():
    # Use standard crewai LLM without provider prefix, so it routes to our proxied OpenAI provider
    return LLM(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=4096
    )

def send_n8n_webhook(payload: dict):
    webhook_url = os.getenv("N8N_WEBHOOK_URL")
    if not webhook_url:
        print("N8N_WEBHOOK_URL not set. Skipping.")
        return
    try:
        response = requests.post(webhook_url, json=payload, timeout=20)
        print(f"N8N status: {response.status_code}")
    except Exception as e:
        print(f"N8N error: {str(e)}")

def run_crew(inputs: dict) -> dict:
    try:
        llm = build_llm()

        from tools.pdf_parser import PDFParserTool
        from tools.scraper import ScraperTool
        from tools.search import SearchTool
        from tools.chroma_store import SaveApplicationTool, RetrieveSimilarTool
        
        # Tools available to the agents
        tools = [
            PDFParserTool(),
            ScraperTool(),
            SearchTool(),
            SaveApplicationTool(),
            RetrieveSimilarTool()
        ]

        agents = {
            "jd_analyser":         create_jd_analyser(llm),
            "profile_parser":      create_profile_parser(llm, tools),
            "ats_scorer":          create_ats_scorer(llm),
            "resume_tailor":       create_resume_tailor(llm),
            "cover_letter_writer": create_cover_letter_writer(llm, tools),
            "cold_email_writer":   create_cold_email_writer(llm),
            "interview_prep":      create_interview_prep_agent(llm),
            "memory":              create_memory_agent(llm),
            "critic":              create_critic(llm),
        }

        tasks_dict = create_tasks(agents, inputs)

        crew = Crew(
            agents=list(agents.values()),
            tasks=list(tasks_dict.values()),
            process=Process.sequential,
            verbose=True,
            embedder={
                "provider": "huggingface",
                "config": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            }
        )

        crew.kickoff()

        output_payload = {
            "ats_score":       getattr(tasks_dict["score_ats"].output,          'raw', str(tasks_dict["score_ats"].output)),
            "tailored_resume": getattr(tasks_dict["tailor_resume"].output,      'raw', str(tasks_dict["tailor_resume"].output)),
            "cover_letter":    getattr(tasks_dict["write_cover_letter"].output, 'raw', str(tasks_dict["write_cover_letter"].output)),
            "cold_email":      getattr(tasks_dict["write_cold_email"].output,   'raw', str(tasks_dict["write_cold_email"].output)),
            "interview_prep":  getattr(tasks_dict["prep_interview"].output,     'raw', str(tasks_dict["prep_interview"].output)),
            "critic_review":   getattr(tasks_dict["critique_outputs"].output,   'raw', str(tasks_dict["critique_outputs"].output)),
        }

        send_n8n_webhook(output_payload)
        return output_payload

    except Exception as e:
        print(f"Crew error: {str(e)}")
        raise
