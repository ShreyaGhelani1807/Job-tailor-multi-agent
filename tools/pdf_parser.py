"""
PDF Parser Tool for the AI Job Application Tailor.
Extracts text from PDF resumes.
"""

import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Load environment variables (coding rule #3)
load_dotenv()

class PDFParserInput(BaseModel):
    """
    Schema for the PDFParserTool inputs.
    """
    file_path: str = Field(..., description="The absolute path to the PDF resume file to parse.")


class PDFParserTool(BaseTool):
    """
    A CrewAI tool that extracts text from a PDF resume file.
    Use this tool to parse the user's uploaded resume.
    """
    name: str = "PDF Resume Parser"
    description: str = "Extracts and structures text from a PDF resume. Returns a clean string of the full resume text. Requires the absolute path to the PDF file."
    args_schema: type[BaseModel] = PDFParserInput

    def _run(self, file_path: str) -> str:
        """
        Executes the tool to extract text from the provided PDF file.
        
        Args:
            file_path (str): The path to the PDF file to be parsed.
            
        Returns:
            str: A clean string of the full resume text, or an error message if parsing fails.
        """
        try:
            if not os.path.exists(file_path):
                return f"Error: The file at {file_path} does not exist."

            try:
                # Open the document using PyMuPDF (fitz)
                doc = fitz.open(file_path)
            except fitz.FileDataError:
                return f"Error: The PDF file at {file_path} is corrupted or cannot be opened."
            except Exception as doc_e:
                return f"Error: Failed to open the PDF file. Details: {str(doc_e)}"

            if doc.page_count == 0:
                doc.close()
                return "Error: The provided PDF file is empty (contains 0 pages)."

            extracted_text_blocks = []
            for page_num in range(doc.page_count):
                try:
                    page = doc.load_page(page_num)
                    # Extract raw text, preserving some vertical structure naturally
                    text = page.get_text("text")
                    if text and text.strip():
                        extracted_text_blocks.append(text.strip())
                except Exception as page_e:
                    # Log or handle specific page errors gracefully, continue to next page
                    extracted_text_blocks.append(f"[Error extracting page {page_num}: {str(page_e)}]")

            doc.close()

            final_text = "\n\n".join(extracted_text_blocks).strip()

            if not final_text:
                return "Error: Scanned the PDF, but no extractable text was found."

            return final_text

        except Exception as e:
            # Catch-all for any unexpected errors per coding rule #2
            return f"Error: An unexpected error occurred while parsing the PDF: {str(e)}"
