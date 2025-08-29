# src/asset_analyst/agents/supervisor.py
from typing import TypedDict, Optional, Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate

from asset_analyst.configs.config_init import Config
from asset_analyst.agents import make_llm
from asset_analyst.tools.web_search import WebSearch
from asset_analyst.agents.analysis import Analysis
from asset_analyst.agents.data_gathering import DataGathering
from asset_analyst.schemas import ResearchData, InvestmentAnalysis


class AnalysisState(TypedDict, total=False):
    question: str
    research: Dict[str, Any]
    analysis: Dict[str, Any]
    status: Literal["init", "researching", "analyzing", "done"]
    error: Optional[str]


class SuperVise:
    def __init__(self):
        self.data_gathering_instance = DataGathering()
        self.analysis_instance = Analysis()
        self.config = Config()

    def gather_node(self, state: AnalysisState) -> AnalysisState:
        try:
            web_search = WebSearch()
            tavily_tool = web_search.get_tavily_tool(
                k=int(self.config.TAVILY_SEARCH_COUNT or 5)
            )
            self.data_gathering_instance.add_tools([tavily_tool])

            # This now returns a validated ResearchData.dict()
            research = self.data_gathering_instance.run_data_gathering(
                state["question"]
            )

            return {**state, "research": research, "status": "analyzing"}
        except Exception as e:
            return {**state, "error": f"Research failed: {str(e)}", "status": "done"}

    def analysis_node(self, state: AnalysisState) -> AnalysisState:
        try:
            # research is now guaranteed to have the ResearchData structure
            analysis = self.analysis_instance.run_analysis_agent(
                state["question"], state["research"]
            )
            # analysis is now a validated InvestmentAnalysis.dict()
            return {**state, "analysis": analysis, "status": "done"}
        except Exception as e:
            return {**state, "error": f"Analysis failed: {str(e)}", "status": "done"}

    def supervisor_router(self, state: AnalysisState) -> str:
        if state.get("status") in (None, "init"):
            return "gather"
        if state.get("status") == "researching" or ("research" not in state):
            return "gather"
        if state.get("status") == "analyzing" or (
            "analysis" not in state and "research" in state
        ):
            return "analyze"
        return "finish"

    def llm_supervisor_router(
        self, state: AnalysisState
    ) -> Literal["gather", "analyze", "finish"]:
        PROMPT = ChatPromptTemplate.from_template(
            """You are the supervisor. State:
            {state}

            Which node should run next? Options: GATHER, ANALYZE, FINISH.
            Answer with a single word."""
        )
        llm = make_llm(temperature=0.0)
        res = (PROMPT | llm).invoke({"state": state}).content.strip().upper()
        if "GATHER" in res:
            return "gather"
        if "ANALYZE" in res:
            return "analyze"
        return "finish"


def build_graph():
    supervise_instance = SuperVise()
    g = StateGraph(AnalysisState)
    g.add_node("gather", supervise_instance.gather_node)
    g.add_node("analyze", supervise_instance.analysis_node)
    g.add_edge(START, "gather")
    g.add_conditional_edges(
        "gather",
        supervise_instance.supervisor_router,
        {"gather": "gather", "analyze": "analyze", "finish": END},
    )
    g.add_conditional_edges(
        "analyze",
        supervise_instance.supervisor_router,
        {"gather": "gather", "analyze": "analyze", "finish": END},
    )
    return g.compile()
