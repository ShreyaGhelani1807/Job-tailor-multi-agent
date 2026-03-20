"""
Search Tool for the AI Job Application Tailor.
Uses Serper API to perform web searches for company info and recruiters.
"""

import os
import json
import requests
from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

load_dotenv()

class SearchInput(BaseModel):
    """Schema for the SearchTool inputs."""
    query: str = Field(..., description="The search query to look up on the web.")

class SearchTool(BaseTool):
    """
    A CrewAI tool that performs a web search using the Serper API.
    Use this to find company news, recruiter names, or ATS systems used by companies.
    """
    name: str = "web_search"
    description: str = "Searches the web for recent information. Ideal for finding company news or recruiter names."
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        """
        Executes the web search using Serper API.
        
        Args:
            query (str): The search query.
            
        Returns:
            str: A formatted string containing the top search results, or an error message.
        """
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return "Error: SERPER_API_KEY is not defined in the environment variables."
            
        try:
            url = "https://google.serper.dev/search"
            payload = json.dumps({"q": query, "gl": "us", "hl": "en", "num": 5})
            headers = {
                'X-API-KEY': api_key,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(url, headers=headers, data=payload, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                organic_results = data.get("organic", [])
                
                if not organic_results:
                    return f"No results found for query: {query}"
                    
                formatted_results = []
                for idx, result in enumerate(organic_results, start=1):
                    title = result.get("title", "No Title")
                    snippet = result.get("snippet", "No Snippet")
                    formatted_results.append(f"{idx}. {title}\nSummary: {snippet}")
                    
                return "\n\n".join(formatted_results)
            else:
                return f"Error: Serper API request failed with status code {response.status_code}: {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"Error: Network exception occurred while contacting Serper API: {str(e)}"
        except Exception as e:
            return f"Error: An unexpected error occurred during search: {str(e)}"
