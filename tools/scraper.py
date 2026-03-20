"""
Scraper Tool for the AI Job Application Tailor.
Uses Firecrawl to scrape public LinkedIn profiles or job posting URLs.
"""

import os
import requests
from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

load_dotenv()

class ScraperInput(BaseModel):
    """Schema for the ScraperTool inputs."""
    url: str = Field(..., description="The URL of the webpage to scrape (e.g., Job posting or LinkedIn profile).")

class ScraperTool(BaseTool):
    """
    A CrewAI tool that extracts text content from a web page using the Firecrawl API.
    Use this to scrape job descriptions or public LinkedIn profiles.
    """
    name: str = "web_scraper"
    description: str = "Scrapes the content of a given URL and returns its markdown text. Ideal for job postings and LinkedIn public profiles."
    args_schema: type[BaseModel] = ScraperInput

    def _run(self, url: str) -> str:
        """
        Executes the scraping using Firecrawl API.
        
        Args:
            url (str): The URL to scrape.
            
        Returns:
            str: The scraped markdown text content or an error message.
        """
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            return "Error: FIRECRAWL_API_KEY is not defined in the environment variables."
            
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            # Firecrawl v1 API endpoint for single page scrape
            endpoint = "https://api.firecrawl.dev/v1/scrape"
            payload = {
                "url": url,
                "formats": ["markdown"]
            }
            
            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    markdown_content = data.get("data", {}).get("markdown", "")
                    if markdown_content:
                        return markdown_content
                    return "Error: Scrape successful but no markdown content found in response."
                else:
                    return f"Error from Firecrawl: {data.get('error', 'Unknown error')}"
            else:
                return f"Error: Firecrawl API request failed with status code {response.status_code}: {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"Error: Network exception occurred while contacting Firecrawl API: {str(e)}"
        except Exception as e:
            return f"Error: An unexpected error occurred during scraping: {str(e)}"
