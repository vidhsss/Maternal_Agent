import re
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from functools import lru_cache

LANGUAGE_PATTERNS = {
    'hindi': re.compile(r'[\u0900-\u097F]'),
    'assamese': re.compile(r'[\u0980-\u09FF]'),
    'english': re.compile(r'[a-zA-Z]')
}

@lru_cache(maxsize=100)
def detect_language(text: str) -> str:
    """
    Detect the primary language of the text (hindi, assamese, english, or mixed).
    """
    counts = {lang: len(pattern.findall(text)) for lang, pattern in LANGUAGE_PATTERNS.items()}
    total = sum(counts.values())
    if total == 0:
        return 'english'
    lang_ratios = {lang: count/total for lang, count in counts.items()}
    primary_lang = max(lang_ratios.items(), key=lambda x: x[1])[0]
    if lang_ratios[primary_lang] < 0.7:
        return 'mixed'
    return primary_lang

@lru_cache(maxsize=100)
def translate_query(query: str) -> str:
    """
    Translate query to English using GPT-3.5-turbo if needed.
    """
    translation_prompt = PromptTemplate(
        template="""Translate the following maternal health query to English, preserving all medical terms and meaning:\n\nQuery: {query}\n\nTranslation:""",
        input_variables=["query"]
    )
    try:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        response = llm(translation_prompt.format_prompt(query=query).to_messages())
        return response.content.strip()
    except Exception as e:
        print(f"Translation failed: {e}")
        return query 