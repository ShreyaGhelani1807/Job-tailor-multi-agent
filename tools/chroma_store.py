import chromadb
import os
from datetime import datetime
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "memory", "chroma_db")

from chromadb.utils import embedding_functions

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Explicitly use the default embedding function so it doesn't try to use OpenAI
    ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection("job_applications", embedding_function=ef)

class SaveApplicationInput(BaseModel):
    company: str = Field(description="Company name")
    role: str = Field(description="Job role title")
    jd_summary: str = Field(description="Short summary of the job description")
    ats_score: str = Field(description="ATS match score as percentage string")
    resume: str = Field(description="Tailored resume text")
    cover_letter: str = Field(description="Cover letter text")

class SaveApplicationTool(BaseTool):
    name: str = "Save Application"
    description: str = "Saves a completed job application to ChromaDB memory for future reference."
    args_schema: Type[BaseModel] = SaveApplicationInput

    def _run(self, company: str, role: str, jd_summary: str,
             ats_score: str, resume: str, cover_letter: str) -> str:
        try:
            collection = get_collection()
            doc_id = f"{company}_{role}_{datetime.now().strftime('%Y%m%d%H%M%S')}".replace(" ", "_")
            collection.add(
                documents=[jd_summary],
                metadatas=[{
                    "company": company,
                    "role": role,
                    "ats_score": ats_score,
                    "resume": resume[:500],
                    "cover_letter": cover_letter[:500],
                    "date": datetime.now().strftime("%Y-%m-%d")
                }],
                ids=[doc_id]
            )
            return f"Application saved successfully with ID: {doc_id}"
        except Exception as e:
            return f"Error saving application: {str(e)}"

class RetrieveSimilarInput(BaseModel):
    jd_summary: str = Field(description="Job description summary to find similar past applications")

class RetrieveSimilarTool(BaseTool):
    name: str = "Retrieve Similar Applications"
    description: str = "Retrieves the 3 most similar past applications from memory based on job description."
    args_schema: Type[BaseModel] = RetrieveSimilarInput

    def _run(self, jd_summary: str) -> str:
        try:
            collection = get_collection()
            count = collection.count()
            if count == 0:
                return "No past applications found in memory."
            results = collection.query(
                query_texts=[jd_summary],
                n_results=min(3, count)
            )
            output = []
            for i, meta in enumerate(results["metadatas"][0]):
                output.append(
                    f"Past Application {i+1}:\n"
                    f"  Company: {meta['company']} | Role: {meta['role']} | "
                    f"Date: {meta['date']} | ATS: {meta['ats_score']}\n"
                    f"  Resume snippet: {meta['resume']}\n"
                    f"  Cover letter snippet: {meta['cover_letter']}"
                )
            return "\n\n".join(output)
        except Exception as e:
            return f"Error retrieving applications: {str(e)}"