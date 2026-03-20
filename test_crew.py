import os
import sys

# Change directory so imports work
current_dir = r"d:\Projects\AI Job Tailor Agent"
os.chdir(current_dir)
sys.path.insert(0, current_dir)

from crew import run_crew

if __name__ == "__main__":
    inputs = {
        "jd_text": "Looking for a software engineer with Python experience.",
        "company": "TestCorp",
        "role": "Software Engineer"
    }
    try:
        run_crew(inputs)
    except Exception as e:
        import traceback
        traceback.print_exc()
