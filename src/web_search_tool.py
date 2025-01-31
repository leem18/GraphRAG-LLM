# lib/web_search_tool.py
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import GoogleSearchAPIWrapper

class WebSearchTool:
    def __init__(self, engine, max_results=5):
        self.max_results = max_results
        self.api = self._get_api_wrapper(engine)

    def _get_api_wrapper(self, engine):
        if engine.lower() == "tavily":
            return TavilyAPIWrapper(max_results=self.max_results)
        elif engine.lower() == "google":
            return GoogleAPIWrapper(max_results=self.max_results)
        else:
            raise ValueError(f"Unknown search engine: {engine}")

    def run(self, query):
        results = self.api.search(query)
        standardized_results = self._standardize_results(results)
        return standardized_results

    def _standardize_results(self, results):
        standardized = []
        for result in results:
            if isinstance(result, dict):
                standardized.append({
                    "url": result.get("url") or result.get("link"),
                    "content": result.get("content") or result.get("snippet")
                })
            else:
                print(f"Unexpected result format: {result}")
        return standardized


class TavilyAPIWrapper:
    def __init__(self, max_results):
        self.max_results = max_results

    def search(self, query):
        web_search_tool = TavilySearchResults(max_results=self.max_results)
        docs = web_search_tool.invoke({"query": query})
        return docs


class GoogleAPIWrapper:
    def __init__(self, max_results):
        self.max_results = max_results

    def search(self, query):
        search = GoogleSearchAPIWrapper()
        result = search.results(query, num_results=self.max_results)
        return result
