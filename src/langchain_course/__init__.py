from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that search over internet

    Args:
        query (str): The query to search for

    Returns:
        str: The search result
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query)

llm = ChatOllama(model="gpt-oss")
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 job postings for an ai engineer using langchain in the egypt area on linkedin and provide the links and details")})
    print(result)   
    
    
if __name__ == "__main__":
    main()