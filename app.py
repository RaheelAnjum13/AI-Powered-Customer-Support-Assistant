import streamlit as st
from crewai import Task, Crew, Agent
from crewai_tools.tools import ScrapeWebsiteTool
from bs4 import BeautifulSoup
import requests, os, time


class ProgressTracker:
    def __init__(self):
        self.steps = []

    def add_step(self, message):
        self.steps.append(message)

    def render(self):
        with st.expander("📋 Agent Step-by-Step Progress", expanded=False):
            for i, step in enumerate(self.steps[-15:], 1):
                st.markdown(f"**{i}.** {step}")


st.set_page_config(page_title="CrewAI Chatbot", layout="centered")
st.title("🤖 Chat Support Assistant")


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions only using the provided website documentation.",
        }
    ]

# Sidebar inputs
with st.sidebar:
    st.markdown("🔐 **Enter your OpenAI API Key**")
    openai_key = st.text_input("API Key", type="password")

    st.markdown("🌐 **Website to Scrape**")
    website_url = st.text_input("🔗 Enter website URL")


if not openai_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()


os.environ["OPENAI_API_KEY"] = openai_key
os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"


for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


inquiry = st.chat_input("Ask your support question here...")

if inquiry and website_url:
    st.session_state.messages.append({"role": "user", "content": inquiry})

    with st.chat_message("user"):
        st.markdown(inquiry)

    progress = ProgressTracker()

    with st.spinner("🛠️ Agents working on your request..."):
        try:
            progress.add_step("🌐 Fetching website content...")
            soup = BeautifulSoup(
                requests.get(website_url, timeout=10).text, "html.parser"
            )
            text_content = soup.get_text(separator=" ", strip=True)
            preview = text_content[:1000] + "..."
            progress.add_step("✅ Website content retrieved successfully!")

            progress.add_step("🧠 Preparing context from chat messages...")
            context_block = "\n".join(
                f"<|{msg['role']}|> {msg['content']}"
                for msg in st.session_state.messages
                if msg["role"] in ["user", "assistant"]
            )

            progress.add_step("🛠️ Initializing support and QA agents...")
            support_agent = Agent(
                role="Senior Support Representative",
                goal="Answer user questions using only provided website documentation.",
                backstory="You're a helpful support agent. Use the scraped website to assist users. Don't hallucinate.",
                allow_delegation=False,
                verbose=True,
                model_parameters={"temperature": 0.5, "max_tokens": 500},
            )
            qa_agent = Agent(
                role="Support QA Reviewer",
                goal="Ensure support quality and accuracy based strictly on website content.",
                backstory="You review support responses to make sure they are helpful and accurate.",
                allow_delegation=False,
                verbose=True,
                model_parameters={"temperature": 0.5, "max_tokens": 300},
            )
            progress.add_step("🤖 Agents initialized.")

            progress.add_step("📝 Creating tasks for support and QA agents...")
            docs_scrape_tool = ScrapeWebsiteTool(website_url=website_url)

            inquiry_resolution = Task(
                description=(
                    f"You are a support assistant responding only using information from a scraped website.\n\n"
                    f"Previous conversation:\n{context_block or 'None'}\n\n"
                    f"<|user|> {inquiry}\n\n"
                    f"Respond with a clear and complete answer in plain text, based only on the website content."
                ),
                expected_output="Plain text answer only.",
                tools=[docs_scrape_tool],
                agent=support_agent,
            )

            qa_review = Task(
                description=(
                    f"Review the support agent's answer for accuracy and clarity based only on website content.\n\n"
                    f"Question:\n<|user|> {inquiry}\n\n"
                    f"Return a final improved plain text answer. No extra formatting."
                ),
                expected_output="Plain text final answer only.",
                agent=qa_agent,
            )

            progress.add_step("📋 Tasks created successfully.")

            progress.add_step("🚀 Executing the CrewAI workflow...")
            crew = Crew(
                agents=[support_agent, qa_agent],
                tasks=[inquiry_resolution, qa_review],
                memory=True,
                verbose=True,
            )

            for msg in [
                "🔍 Support agent is analyzing website...",
                "✍️ Support agent is generating response...",
                "🧐 QA agent is reviewing the response...",
            ]:
                progress.add_step(msg)
                time.sleep(0.6)

            result = crew.kickoff(inputs={"inquiry": inquiry})
            final_response = result.output if hasattr(result, "output") else str(result)

            st.session_state.messages.append(
                {"role": "assistant", "content": final_response}
            )

            with st.chat_message("assistant"):
                st.markdown(final_response)

            progress.add_step("✅ All tasks completed successfully.")

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )
            with st.chat_message("assistant"):
                st.markdown(error_msg)
            progress.add_step(error_msg)

    progress.render()
