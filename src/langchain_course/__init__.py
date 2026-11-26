import json

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field


class Source(BaseModel):
    title: str
    url: str
    snippet: str


class AgentResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)


class RelevantResults(BaseModel):
    result_indexes: list[int] = Field(
        description="One-based indexes of relevant job postings from the search results.",
        max_length=3,
    )


load_dotenv()

llm = ChatOllama(model="gpt-oss", temperature=0)
search = TavilySearch(
    include_domains=["linkedin.com"],
    search_depth="advanced",
    include_images=False,
    max_results=10,
    topic="general",
)
result_selector = llm.with_structured_output(
    RelevantResults,
    method="json_schema",
)


def search_jobs(query: str) -> AgentResponse:
    raw_results = search.invoke({"query": query})
    if isinstance(raw_results, str):
        search_data = json.loads(raw_results)
    elif isinstance(raw_results, dict):
        search_data = raw_results
    else:
        raise TypeError("Tavily returned search results in an unsupported format.")

    if not isinstance(search_data, dict) or not isinstance(
        search_data.get("results"), list
    ):
        raise ValueError("Tavily returned an unexpected search-results format.")

    indexed_results = [
        {
            "index": index,
            "title": result["title"],
            "url": result["url"],
            "snippet": result["content"],
        }
        for index, result in enumerate(search_data["results"], start=1)
        if isinstance(result, dict)
        and all(isinstance(result.get(key), str) for key in ("title", "url", "content"))
    ]
    if not indexed_results:
        return AgentResponse(answer="No search results were returned.")

    selection = result_selector.invoke(
        [
            SystemMessage(
                content=(
                    "Select up to three search results that are actual AI Engineer "
                    "job postings in Egypt (or explicitly remote and open to Egypt) "
                    "and mention LangChain. Return only their one-based indexes. "
                    "Use only the supplied results; do not infer missing details. "
                    "Return fewer results if fewer match."
                )
            ),
            HumanMessage(content=json.dumps(indexed_results)),
        ]
    )
    results_by_index = {item["index"]: item for item in indexed_results}
    if any(index not in results_by_index for index in selection.result_indexes):
        raise ValueError(
            "The model selected a search result index that does not exist."
        )

    sources = [
        Source(
            title=results_by_index[index]["title"],
            url=results_by_index[index]["url"],
            snippet=results_by_index[index]["snippet"],
        )
        for index in dict.fromkeys(selection.result_indexes)
    ]
    return AgentResponse(
        answer=f"Found {len(sources)} matching search result(s).",
        sources=sources,
    )


def main() -> None:
    structured_response = search_jobs(
        "AI Engineer LangChain job Egypt site:linkedin.com"
    )
    print(structured_response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
