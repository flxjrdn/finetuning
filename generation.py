import os
from typing import List

from groq import Groq

MODEL = "llama-3.3-70b-versatile"
TOKEN = os.environ.get("GROQ_API_TOKEN")
TEMPERATURE = 0
MAX_COMPLETION_TOKENS = 512
SYSTEM_INSTRUCTION = (
    "Du bist ein hilfsbereiter Assistent, der spezialisiert "
    "ist auf Fragen im Versicherungskontext."
)
PROMPT = (
    "Beantworte die folgende ANFRAGE."
    "Nutze für die Beantwortung ausschließlich die im KONTEXT gegebenen Informationen."
    "ANFRAGE: PLACEHOLDER_QUESTION"
    "KONTEXT: PLACEHOLDER_CHUNKS"
)


class AnswerGenerator:
    def __init__(self):
        self.client = Groq(
            api_key=TOKEN,
        )

    def generate(self, question: str, chunks: List[str]) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_INSTRUCTION,
                },
                {
                    "role": "user",
                    "content": self.create_prompt(
                        question=question,
                        chunks=chunks,
                    ),
                },
            ],
            model=MODEL,
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )

        return chat_completion.choices[0].message.content

    @staticmethod
    def create_prompt(question: str, chunks: List[str]) -> str:
        prompt = PROMPT.replace("PLACEHOLDER_QUESTION", question)
        context = "\n".join(chunks)
        prompt = prompt.replace("PLACEHOLDER_CHUNKS", context)
        return prompt
