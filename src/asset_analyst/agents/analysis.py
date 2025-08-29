from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from asset_analyst.agents import make_llm
from asset_analyst.schemas.investment import InvestmentAnalysis
from asset_analyst.schemas.prompts import ANALYSIS_PROMT

SYSTEM = ANALYSIS_PROMT


class Analysis:
    def __init__(self):
        self.llm = make_llm(temperature=0.2)

    def _build_analysis_agent(self, system: str = SYSTEM) -> Runnable:
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    ("human", "Research:\n{research}\n\nUser question:\n{question}"),
                ]
            )
            chain = prompt | self.llm.bind_tools(
                tools=[InvestmentAnalysis], tool_choice="InvestmentAnalysis"
            )
            return chain
        except Exception as e:
            print("Error while building Analysis Agent", str(e))
            raise

    def run_analysis_agent(
        self, question: str, research: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            chain = self._build_analysis_agent()
            result = chain.invoke({"question": question, "research": research})
            return result.dict()
        except Exception as e:
            print("Error while running analysis", str(e))
            raise
