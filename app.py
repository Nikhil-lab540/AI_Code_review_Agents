import streamlit as st
import os
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
LLM_MODEL = "gemini/gemini-2.0-flash"
API_KEY = "AIzaSyBGS7z6Z8TZ7vQJWmTNLb2WqUuNxxJxz70"

if not API_KEY:
    st.error("API key not found. Please set the GEMINI_API_KEY environment variable.")
    st.stop()


# --- Helper Functions ---
def create_agent(role: str, goal: str, backstory: str, llm_model: LLM) -> Agent:
    """
    Creates an agent with the specified role, goal, backstory, and LLM.

    Args:
        role (str): The role of the agent.
        goal (str): The goal of the agent.
        backstory (str): The backstory of the agent.
        llm_model (LLM): The language model to use for the agent.

    Returns:
        Agent: The created agent.
    """
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm_model
    )


def create_task(agent: Agent, description: str, expected_output: str) -> Task:
    """
    Creates a task for the specified agent with the given description and expected output.

    Args:
        agent (Agent): The agent to assign the task to.
        description (str): The description of the task.
        expected_output (str): The expected output of the task.

    Returns:
        Task: The created task.
    """
    return Task(
        agent=agent,
        description=description,
        expected_output=expected_output
    )


def run_code_review(code: str):
    """
    Runs the code review process using a crew of agents.

    Args:
        code (str): The code to review.

    Returns:
        str: The results of the code review.
    """
    llm_model = LLM(model=LLM_MODEL, api_key=API_KEY)

    # Agents
    syntax_agent = create_agent(
        role='Syntax Stylist',
        goal='Enforce clean code style and formatting',
        backstory='You are a strict code style reviewer using standard guidelines.',
        llm_model=llm_model
    )

    bug_agent = create_agent(
        role='Bug Hunter',
        goal='Find logical or runtime errors in the code',
        backstory='You analyze flow and catch potential bugs or poor logic.',
        llm_model=llm_model
    )

    security_agent = create_agent(
        role='Security Analyst',
        goal='Identify security vulnerabilities or bad practices',
        backstory='You know OWASP and scan code for security risks.',
        llm_model=llm_model
    )

    refactor_agent = create_agent(
        role='Refactoring Advisor',
        goal='Improve the code with clean, modular suggestions',
        backstory='You help developers make cleaner and more maintainable code.',
        llm_model=llm_model
    )

    # Tasks
    tasks = [
        create_task(
            agent=syntax_agent,
            description=f"Review this code for syntax and style:\n\n{code}",
            expected_output="A clear explanation of any syntax/style issues, following standard guidelines."
        ),
        create_task(
            agent=bug_agent,
            description=f"Review this code for bugs and errors:\n\n{code}",
            expected_output="A list of bugs or logical issues, and where they occur."
        ),
        create_task(
            agent=security_agent,
            description=f"Review this code for security issues:\n\n{code}",
            expected_output="Any security vulnerabilities in the code and suggestions to fix them."
        ),
        create_task(
            agent=refactor_agent,
            description=f"Suggest refactorings for this code:\n\n{code}",
            expected_output="Refactoring suggestions to improve code readability, reusability, or efficiency."
        )
    ]

    crew = Crew(agents=[syntax_agent, bug_agent, security_agent, refactor_agent], tasks=tasks)
    results = crew.kickoff()
    return results


# --- Streamlit UI ---
st.title("🧠 AI Code Review Crew")

code = st.text_area("Paste your code here:", height=300)

if st.button("🔍 Run Code Review"):
    code = code.strip()
    if not code:
        st.warning("Please paste some code before running.")
    else:
        results = run_code_review(code)
        st.subheader("🧾 Review Summary")
        st.info(results)