# 🚀 LangChain AI Search API Integration

[![PyPI version](https://badge.fury.io/py/langchain-aisearchapi.svg)](https://pypi.org/project/langchain-aisearchapi/)
[![Python Support](https://img.shields.io/pypi/pyversions/langchain-aisearchapi.svg)](https://pypi.org/project/langchain-aisearchapi/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official [LangChain](https://python.langchain.com/) integration for the [AI Search API](https://aisearchapi.io/).**  
> Use semantic search, contextual answers, and intelligent agents in your LangChain projects with **just one package**.

---

## ✨ Features

- 🔑 **One Package Setup** – `pip install langchain-aisearchapi` and you’re ready  
- 🤖 **LLM Interface** – Use AI Search API as a LangChain LLM  
- 💬 **Chat Model** – Build conversational agents with context memory  
- 🛠️ **Tools for Agents** – Add AI Search directly into LangChain workflows  
- 📚 **Prebuilt Chains** – Research, Q&A, fact-checking out of the box  

👉 To start, create an account and get your API key:  
- [🆕 Sign Up](https://app.aisearchapi.io/join)  
- [🔑 Log In](https://app.aisearchapi.io/login)  
- [📊 Dashboard](https://app.aisearchapi.io/dashboard)  

---

## ⚡ Installation

Install the integration from [PyPI](https://pypi.org/project/langchain-aisearchapi/):

```bash
pip install langchain-aisearchapi
```

That’s it — no extra setup required.  

---

## 🚀 Quick Start

### 1. Basic LLM Usage

```python
from langchain_aisearchapi import AISearchLLM

llm = AISearchLLM(api_key="your-key")
response = llm("Explain semantic search in simple terms")
print(response)
```

### 2. Conversational Chat

```python
from langchain_aisearchapi import AISearchChat
from langchain.schema import HumanMessage

chat = AISearchChat(api_key="your-key")

messages = [
    HumanMessage(content="What is LangChain?"),
    HumanMessage(content="Why do developers use it?")
]

response = chat(messages)
print(response.content)
```

### 3. AI Search as a Tool in Agents

```python
from langchain_aisearchapi import AISearchTool, AISearchLLM
from langchain.agents import initialize_agent, AgentType

search_tool = AISearchTool(api_key="your-key")
llm = AISearchLLM(api_key="your-key")

agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("Find the latest SpaceX launch details")
print(result)
```

### 4. Research Assistant

```python
from langchain_aisearchapi import create_research_chain

research = create_research_chain(api_key="your-key")
result = research.run("Breakthroughs in AI search technology 2024")
print(result)
```

---

## 🛠️ Components

| Component | Description | Use Case |
|-----------|-------------|----------|
| `AISearchLLM` | AI Search API as an LLM | Completions, text generation |
| `AISearchChat` | Chat model with context | Conversational AI, assistants |
| `AISearchTool` | Search as LangChain tool | Agents, workflows |
| `create_research_chain()` | Ready-made chain | Research and reporting |

Full API reference: [AI Search API Docs](https://docs.aisearchapi.io/).

---

## ❗ Troubleshooting

- **No API key?** → [Sign up](https://app.aisearchapi.io/join) or [log in](https://app.aisearchapi.io/login).  
- **Key issues?** → Check your [dashboard](https://app.aisearchapi.io/dashboard).  
- **Rate limited?** → Use retry logic with [tenacity](https://tenacity.readthedocs.io/).  

---

## 📚 Resources

- [AI Search API Homepage](https://aisearchapi.io/)  
- [Join / Sign Up](https://app.aisearchapi.io/join)  
- [Log In](https://app.aisearchapi.io/login)  
- [Dashboard](https://app.aisearchapi.io/dashboard)  
- [Documentation](https://docs.aisearchapi.io/)  
- [PyPI Package](https://pypi.org/project/langchain-aisearchapi/)  

---

## 🎉 Start Now

Install the package, get your API key, and build **powerful LangChain apps** with the AI Search API:  
```bash
pip install langchain-aisearchapi
```

👉 [Join now](https://app.aisearchapi.io/join) to claim your free API key and start building!  

---

*Made with ❤️ for the AI Search API + LangChain developer community*

---

### SEO Keywords  
*LangChain AI Search API integration, AI Search API Python package, semantic search LangChain, contextual AI LangChain, AI chatbot LangChain, AI Search API key setup*
