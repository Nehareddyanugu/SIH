# components/language_detector.py
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from typing import Any, Dict, Text
import fasttext
import os
from googletrans import Translator

MODEL_PATH = os.path.join(os.path.dirname(__file__), "lid.176.bin")
translator = Translator()

@DefaultV1Recipe.register(
    "components.language_detector.LanguageDetector",
    is_trainable=False
)
class LanguageDetector(GraphComponent):

    def __init__(self, model):
        self.model = model

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "LanguageDetector":
        model = fasttext.load_model(MODEL_PATH)
        return cls(model)

    def process(self, message: Message, **kwargs: Any) -> None:
        text = message.get("text")
        if not text:
            message.set("language", "unknown", add_to_output=True)
            return

        # Detect language
        prediction = self.model.predict(text)
        lang = prediction[0][0].replace("__label__", "")
        message.set("language", lang, add_to_output=True)

        # Translate to English if not English
        if lang != "en":
            translated = translator.translate(text, src=lang, dest="en")
            message.set("text", translated.text)

        print(f"[LanguageDetector] '{text}' → {lang}, translated → '{message.get('text')}'")
