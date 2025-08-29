from typing import List, Dict, Any

from langchain_tavily import TavilySearch
from asset_analyst.configs.config_init import Config


class WebSearch:
    def __init__(self):
        self.config = Config()
        self.api_key = self.config.TAVILY_API_KEY

    def get_tavily_tool(self, k: int = None):
        # Returns a LangChain Tool compatible with agent tool-calling
        max_results = int(k if k is not None else self.config.TAVILY_SEARCH_COUNT)
        return TavilySearch(max_results=max_results, tavily_api_key=self.api_key)

    def web_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        tool = self.get_tavily_tool(k)
        return tool.run(query)
