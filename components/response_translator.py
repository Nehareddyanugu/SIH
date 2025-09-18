# components/response_translator.py
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from typing import Any, Dict, Text
from googletrans import Translator

translator = Translator()

@DefaultV1Recipe.register(
    "components.response_translator.ResponseTranslator",
    is_trainable=False
)
class ResponseTranslator(GraphComponent):

    def __init__(self):
        pass

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "ResponseTranslator":
        return cls()

    def process(self, message: Message, **kwargs: Any) -> None:
        # Get detected language
        lang = message.get("language")
        if lang != "en":
            # Translate bot responses back to user language
            responses = message.get("response")
            if responses:
                translated_responses = []
                for r in responses:
                    translated_text = translator.translate(r, src="en", dest=lang).text
                    translated_responses.append(translated_text)
                message.set("response", translated_responses)
