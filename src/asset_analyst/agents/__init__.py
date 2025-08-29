from asset_analyst.LLMs import ai_factory
from asset_analyst.configs.config_init import Config

config_instance = Config()

def make_llm(model: str = config_instance.DEFAULT_MODEL, temperature: float = None):
    factory_instance = ai_factory.AIFactory()
    temp = float(temperature if temperature is not None else (config_instance.MODEL_TEMPERATURE or 0.2))
    return factory_instance.return_ai_model(model_name=model, temperature=temp)
