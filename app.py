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
        with st.expander("Agent Workflow", expanded=False):
            for i, step in enumerate(self.steps[-15:], 1):
                st.markdown(f"**{i}.** {step}")


st.set_page_config(page_title="Customer Support Assistant", layout="centered")
st.title("🤖 AI Powered Chat Support Assistant")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


with st.sidebar:
    st.markdown("🔐 **Enter your OpenAI API Key**")
    openai_key = st.text_input("API Key", type="password")

    st.markdown("🌐 **Website to Scrape**")
    website_url = st.text_input("🔗 Enter website URL")


if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"
else:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()


st.markdown("💬 Chat Start from Here")
for entry in st.session_state.chat_history[-10:]:
    st.markdown("🧑‍💬 **You said:**")
    st.write(f"💬 {entry['user']}")

    st.markdown("🤖 **Assistant replied:**")
    st.write(f"📢 {entry['assistant']}")
    st.markdown("---")


inquiry = st.chat_input("Ask your support question here...")

if inquiry and website_url:
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

            progress.add_step("🧠 Preparing context from last 10 messages...")
            context_block = "\n".join(
                f"User: {entry['user']}\nAssistant: {entry['assistant']}"
                for entry in st.session_state.chat_history[-10:]
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
                    f"User Prompt:\n{inquiry}\n\n"
                    f"Respond with a clear and complete answer in plain text, based only on the website content."
                ),
                expected_output="Plain text answer only.",
                tools=[docs_scrape_tool],
                agent=support_agent,
            )
            qa_review = Task(
                description=(
                    f"Review the support agent's answer for accuracy and clarity based only on website content.\n\n"
                    f"Question:\n{inquiry}\n\n"
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

            st.session_state.chat_history.append(
                {"user": inquiry, "assistant": final_response}
            )
            st.session_state.chat_history = st.session_state.chat_history[-10:]
            progress.add_step("✅ All tasks completed successfully.")

            st.success("🎉 Answer ready!")
            st.markdown("**📨 Final Answer:**")
            st.write(final_response)

        except Exception as e:
            progress.add_step(f"❌ Error: {str(e)}")
            st.error(f"❌ Error: {str(e)}")

    progress.render()
