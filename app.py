import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import Tool
import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Arxiv and Wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# Google News API Wrapper
def google_news_search(query: str) -> str:
    api_key = st.secrets['NEWS_API_KEY']
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&pageSize=3"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        articles = data.get("articles", [])
        return "\n\n".join([f"{article['title']} - {article['url']}" for article in articles])
    else:
        return "Error fetching news."

google_news_tool = Tool(
    name="GoogleNews",
    func=google_news_search,
    description="Fetches top news articles from Google News."
)

# Streamlit App Title
st.title("🔎 LangChain - Chat with Search & Reasoning")

# Sidebar for Settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:",value=st.secrets['GROQ_API_KEY'], type="password")

# Initialize Session State for Messages and Chat History
if "session_histories" not in st.session_state:
    st.session_state["session_histories"] = {}

# Select or Create a Session
session_id = st.sidebar.text_input("Enter session ID (for unique chats):", value="default")
if session_id not in st.session_state["session_histories"]:
    st.session_state["session_histories"][session_id] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web and hold conversations. How can I help you?"}
    ]

# Display Chat History
for msg in st.session_state["session_histories"][session_id]:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat Input
if prompt := st.chat_input(placeholder="Ask me anything..."):
    st.session_state["session_histories"][session_id].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki, google_news_tool]

    # Initialize Agent
    search_agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
    )

    # Determine if the query needs tools or reasoning
    keywords = ["search", "find", "news", "wikipedia", "arxiv", "today", "current", "how"]
    if any(kw in prompt.lower() for kw in keywords):
        # Use tools for search-based queries
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(prompt, callbacks=[st_cb])
    else:
        # Use LLM for reasoning or conversational responses
        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["session_histories"][session_id]])
        full_prompt = f"The following is a conversation between a user and an AI assistant. Use context to respond.\n\n{chat_history}\n\nUser: {prompt}\nAssistant:"
        response_json = llm.invoke(full_prompt)
        response = response_json.content

    # Store and Display Response
    st.session_state["session_histories"][session_id].append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
