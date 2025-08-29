# src/asset_analyst/agents/data_gathering.py
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from asset_analyst.agents import make_llm
from asset_analyst.configs.config_init import Config
from asset_analyst.schemas import ResearchData
import json

from asset_analyst.schemas.prompts import DATA_GATHERING_PROMPT

try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None

SYSTEM = DATA_GATHERING_PROMPT


class DataGathering:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "initialized"):
            self.llm = make_llm()
            self.config = Config()
            self.tools: List[BaseTool] = []
            self.parser = PydanticToolsParser(tools=[ResearchData])
            self.initialized = True

    def add_tools(self, tools: Optional[List[BaseTool]] = None):
        try:
            if not tools:
                return
            existing_tool_ids = {id(tool) for tool in self.tools}
            new_tools = [tool for tool in tools if id(tool) not in existing_tool_ids]
            if new_tools:
                self.tools.extend(new_tools)
                print("New tools added", str(new_tools))
            else:
                print("No new tools to be added.")
        except Exception as e:
            print("Exception occured while adding tool ", str(e))
            raise e

    def _build_data_gathering_agent(self, system: str = SYSTEM) -> Runnable:
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    (
                        "human",
                        "User question:\n{task}\n\nSearch results:\n{results_json}",
                    ),
                ]
            )

            # Use structured output with tool choice enforcement
            chain = (
                prompt
                | self.llm.bind_tools(tools=[ResearchData], tool_choice="ResearchData")
                | self.parser
            )
            return chain
        except Exception as e:
            print("Error while building data gathering agent. ", str(e))
            raise e

    def _normalize_results(self, raw: Any) -> List[Dict[str, Any]]:
        if not raw:
            return []
        if isinstance(raw, list):
            return raw
        if (
            isinstance(raw, dict)
            and "results" in raw
            and isinstance(raw["results"], list)
        ):
            return raw["results"]
        return []

    def _call_search_tools(self, task: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        # Try each tool via invoke/run with both input shapes
        for tool in self.tools:
            try:
                if hasattr(tool, "invoke"):
                    out = tool.invoke(task)
                    results.extend(self._normalize_results(out))
                if hasattr(tool, "invoke"):
                    out2 = tool.invoke({"query": task})
                    results.extend(self._normalize_results(out2))
            except Exception:
                pass
            try:
                if hasattr(tool, "run"):
                    out3 = tool.run(task)
                    results.extend(self._normalize_results(out3))
            except Exception:
                pass

        # Hard fallback: direct Tavily client
        if not results and TavilyClient and self.config.TAVILY_API_KEY:
            try:
                client = TavilyClient(api_key=self.config.TAVILY_API_KEY)
                out = client.search(
                    task,
                    search_depth="advanced",
                    max_results=int(self.config.TAVILY_SEARCH_COUNT or 5),
                )
                results.extend(self._normalize_results(out))
            except Exception as e:
                print("Direct Tavily client error:", str(e))
        return results

    def run_data_gathering(self, task: str) -> Dict[str, Any]:
        try:
            search_results = self._call_search_tools(task)
            sources = []
            for r in search_results:
                title = r.get("title") or r.get("name") or ""
                url = r.get("url") or r.get("link") or ""
                snippet = r.get("content") or r.get("snippet") or r.get("text") or ""
                if url:
                    sources.append({"title": title, "url": url, "snippet": snippet})

            chain = self._build_data_gathering_agent()
            result = chain.invoke(
                {
                    "task": task,
                    "results_json": json.dumps(sources[:10], ensure_ascii=False),
                }
            )

            # result is now a validated ResearchData object
            return self._parse_result(result)

        except Exception as e:
            print("Error while gathering Data. ", str(e))
            # Return fallback structure if parsing fails
            return {
                "sources": [],
                "summary": f"Error during research: {str(e)}",
                "key_metrics": {},
            }

    def _parse_result(self, result: Any):
        if isinstance(result, list) and len(result) > 0:
            # Take the first result if it's a list
            result = result[0]

        # result should now be a validated ResearchData object
        if hasattr(result, "dict"):
            return result.dict()
        elif isinstance(result, dict):
            return result
        else:
            # Fallback if result is unexpected
            return {"sources": [], "summary": str(result), "key_metrics": {}}
