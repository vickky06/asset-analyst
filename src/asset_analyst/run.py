from asset_analyst.agents.supervisor import build_graph
from asset_analyst.configs.config_init import Config
import sys
import json


def _to_jsonable(obj):
    try:
        # If it's a LangChain message, use its content
        content = getattr(obj, "content", obj)
        # If it's already basic JSON types
        if isinstance(content, (dict, list, str, int, float, bool)) or content is None:
            return content
        # Try to coerce dict-like objects
        if hasattr(content, "dict"):
            return content.dict()
        return str(content)
    except Exception:
        return str(obj)


def main():
    config = Config()
    config.validate()
    # if len(sys.argv) < 2:
    #     print(
    #         'Usage: asset-analyst "Your asset question (e.g., Should I buy NVDA now?)"'
    #     )
    #     sys.exit(1)
    question = """TSLA stock analysis 12-18 month outlook"""  # sys.argv[1]
    app = build_graph()
    state = {"question": question, "status": "init"}
    final = app.invoke(state)
    print("\n=== Research ===")
    print(json.dumps(_to_jsonable(final.get("research", {})), indent=2))
    print("\n=== Analysis ===")
    print(json.dumps(_to_jsonable(final.get("analysis", {})), indent=2))


if __name__ == "__main__":
    main()
