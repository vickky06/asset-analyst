from asset_analyst.LLMs import ai_factory
from asset_analyst.configs.config_init import Config
from asset_analyst.LLMs.model_enums import Model, Provider
from typing import Union

config_instance = Config()


def make_llm(model: Union[Model, str] = None, temperature: float = None):
    factory_instance = ai_factory.AIFactory()
    if model is None:
        model = Model.GEMINI_2_5_FLASH
    temp = float(
        temperature
        if temperature is not None
        else (config_instance.MODEL_TEMPERATURE or 0.2)
    )
    return factory_instance.return_ai_model(model=model, temperature=temp)
