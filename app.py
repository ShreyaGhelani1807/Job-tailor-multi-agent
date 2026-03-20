import streamlit as st
import tempfile, os
from crew import run_crew

st.set_page_config(page_title="AI Job Tailor", page_icon="🎯", layout="wide")
st.title("AI Job Application Tailor")
st.caption("Powered by CrewAI · Groq · ChromaDB · N8N")

with st.sidebar:
    st.header("Your inputs")
    resume_file  = st.file_uploader("Resume PDF", type=["pdf"])
    linkedin_url = st.text_input("LinkedIn profile URL (optional)")
    jd_url       = st.text_input("Job posting URL (optional)")
    jd_text      = st.text_area("Or paste job description here", height=200)
    company      = st.text_input("Company name")
    role         = st.text_input("Job title / role")
    run_btn      = st.button("Tailor my application", type="primary", use_container_width=True)

if run_btn:
    if not resume_file and not linkedin_url:
        st.error("Please upload a resume PDF or provide a LinkedIn URL.")
    elif not jd_url and not jd_text:
        st.error("Please provide a job posting URL or paste the job description.")
    else:
        resume_path = None
        if resume_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(resume_file.read())
                resume_path = tmp.name

        inputs = {
            "resume_path":  resume_path,
            "linkedin_url": linkedin_url,
            "jd_url":       jd_url,
            "jd_text":      jd_text,
            "company":      company,
            "role":         role
        }

        with st.status("Running agents...", expanded=True) as status:
            st.write("Starting CrewAI crew...")
            try:
                outputs = run_crew(inputs)
                status.update(label="All agents complete!", state="complete")
            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"Something went wrong: {str(e)}")
                outputs = None

        if resume_path:
            os.unlink(resume_path)

        if outputs:
            n8n = outputs.get("n8n_status", {})
            if n8n.get("status") == "delivered":
                st.success("Outputs delivered to your email and saved to Google Drive!")

            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "ATS Score", "Tailored Resume", "Cover Letter",
                "Cold Email", "Interview Prep", "Critic Review"
            ])

            with tab1:
                st.markdown("### ATS Compatibility Report")
                st.markdown(outputs.get("ats_score", "Not generated."))

            with tab2:
                st.markdown("### Tailored Resume")
                resume_text = outputs.get("tailored_resume", "Not generated.")
                st.markdown(resume_text)
                st.download_button("Download resume (.txt)", resume_text,
                                   file_name=f"resume_{company}_{role}.txt")

            with tab3:
                st.markdown("### Cover Letter")
                cover = outputs.get("cover_letter", "Not generated.")
                st.markdown(cover)
                st.button("Copy to clipboard", key="copy_cover",
                          on_click=lambda: st.write(cover))

            with tab4:
                st.markdown("### Cold Recruiter Email")
                st.markdown(outputs.get("cold_email", "Not generated."))

            with tab5:
                st.markdown("### Interview Preparation")
                qa = outputs.get("interview_prep", "Not generated.")
                for i, block in enumerate(qa.split("\n\n")):
                    if block.strip():
                        with st.expander(f"Q{i+1}" if not block.startswith("Q") else block.split("\n")[0]):
                            st.markdown(block)

            with tab6:
                st.markdown("### Critic Review")
                st.markdown(outputs.get("critic_review", "Not generated."))
