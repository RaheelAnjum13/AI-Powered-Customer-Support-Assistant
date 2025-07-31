# 🤖 CrewAI Chat Support Assistant

This project is a **multi-agent customer support assistant** built using **CrewAI**, Streamlit, and web scraping tools. It scrapes a website, processes user inquiries with LLM agents, and returns refined responses after a QA check.

---

## 🚀 Features

- Scrapes website content via `BeautifulSoup`
- Uses **CrewAI agents**: Support & QA Reviewer
- Chat-like interface with session history
- Step-by-step agent progress tracker
- Configurable via `.env` or sidebar input
- Secure API key handling

---

## 🛠 Tech Stack

- 🐍 Python
- 🧠 CrewAI (Agents, Tasks, Crew)
- 🌍 `BeautifulSoup`, `requests`
- 🧪 OpenAI LLMs (via `gpt-3.5-turbo`)
- 🧰 Streamlit for UI

---

## 📂 File Structure

```bash
📁 crewai-support-bot/
├── .venv/                 # Virtual environment (excluded from Git)
├── .env                  # Store your OpenAI key securely
├── app.py                # Streamlit + CrewAI app logic
├── requirements.txt      # Python dependencies
```

---

## ✅ Usage

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Set your OpenAI API key** in `.env` or sidebar.

3. **Run the app**:

```bash
streamlit run app.py
```

4. **Enter a website URL** and ask your support question.

---

## 🤖 Agent Roles

- **Support Agent**: Answers user questions from the scraped website only.
- **QA Agent**: Reviews and refines the response to ensure clarity and accuracy.

---

## 🧠 How It Works

- Website is scraped using BeautifulSoup.
- Agents work via CrewAI tasks:
  1. Support agent responds based on site.
  2. QA agent reviews and improves the reply.
- Final answer shown with interaction history.

---

## 🔐 Environment Variables

`.env`

```
OPENAI_API_KEY=your-openai-key
```

---

## 🧪 Example Requirements

```txt
streamlit
crewai
crewai-tools
beautifulsoup4
requests
```

---

## 🛡 Security Tips

- API key is taken at runtime via sidebar and not hardcoded.
- `.env` is ignored from Git.

---

## 📤 Future Improvements

- Add long-term memory with ChromaDB or Redis
- Integrate with multiple knowledge sources (PDF, Notion, etc.)
- Add user feedback & analytics

---
