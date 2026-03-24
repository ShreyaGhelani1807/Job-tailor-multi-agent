"""
Streamlit UI for the AI Job Application Tailor.
Accepts inputs, renders status logs, and displays the 5 generated outputs.
"""

import os
os.environ["OPENAI_API_KEY"] = "sk-dummy-not-used"


import streamlit as st
import time
from tools.scraper import ScraperTool
from crew import JobApplicationCrew

# Streamlit Page Config
st.set_page_config(page_title="AI Job Application Tailor", page_icon="💼", layout="wide")

def main():
    st.title("💼 AI Job Application Tailor")
    st.markdown("Automate your end-to-end job application tailoring with a CrewAI multi-agent system.")
    
    # Sidebar
    st.sidebar.header("Agent Inputs")
    uploaded_resume = st.sidebar.file_uploader("Resume PDF (*Required*)", type=["pdf"])
    linkedin_url = st.sidebar.text_input("LinkedIn Profile URL (Optional)")
    job_url = st.sidebar.text_input("Job Posting URL")
    job_text = st.sidebar.text_area("Job Description (Paste if no URL)")
    
    if st.sidebar.button("Tailor my application", type="primary"):
        if not uploaded_resume:
            st.sidebar.error("Please upload a Resume PDF.")
            return
        if not job_url and not job_text:
            st.sidebar.error("Please provide either a Job Posting URL or paste the Job Description.")
            return
            
        # Save temp file
        temp_pdf_path = os.path.join(os.getcwd(), "temp_resume.pdf")
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_resume.getbuffer())
            
        st.toast("Starting job tailoring workflow...", icon="🚀")
        
        with st.status("Agents are evaluating your application...", expanded=True) as status:
            try:
                # 1. Setup job context
                jd_context = job_text
                if job_url:
                    st.write("- Scraping Job Description URL...")
                    scraper = ScraperTool()
                    scraped_jd = scraper._run(job_url)
                    if not scraped_jd.startswith("Error"):
                        jd_context = scraped_jd + "\n\n" + job_text
                        
                # 2. Instantiate and run crew
                st.write("- Initialising CrewAI Manager and specialized agents...")
                st.write("- Running hierarchical task execution (this might take a few minutes)...")
                
                crew_job = JobApplicationCrew(
                    resume_path=temp_pdf_path,
                    linkedin_url=linkedin_url if linkedin_url else "Not provided",
                    jd_text_or_url=jd_context
                )
                
                outputs = crew_job.run()
                
                if "error" in outputs:
                    status.update(label="Workflow failed.", state="error", expanded=True)
                    st.error(outputs["error"])
                else:
                    status.update(label="Workflow Complete! Webhook sent.", state="complete", expanded=False)
                    st.toast("All documents generated and delivered via N8N!", icon="✅")
                    # Store outputs
                    st.session_state["outputs"] = outputs
                    
            except Exception as e:
                status.update(label="An error occurred.", state="error", expanded=True)
                st.error(f"Execution Error: {str(e)}")
            finally:
                # Cleanup
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

    # Render Tabs
    if "outputs" in st.session_state:
        outputs = st.session_state["outputs"]
        t1, t2, t3, t4, t5 = st.tabs(["ATS Score", "Tailored Resume", "Cover Letter", "Cold Email", "Interview Prep"])
        
        with t1:
            st.subheader("ATS Score & Gap Analysis")
            st.markdown(outputs.get("ats_score", "No data generated."))
            
        with t2:
            st.subheader("Tailored Resume")
            st.markdown(outputs.get("tailored_resume", "No data generated."))
            
        with t3:
            st.subheader("Personalised Cover Letter")
            st.markdown(outputs.get("cover_letter", "No data generated."))
            
        with t4:
            st.subheader("Recruiter Cold Email")
            st.markdown(outputs.get("cold_email", "No data generated."))
            
        with t5:
            st.subheader("Role-specific Interview Prep")
            st.markdown(outputs.get("interview_prep", "No data generated."))

if __name__ == "__main__":
    main()
