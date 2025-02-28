import os

from groq import Groq

MODEL = "llama-3.3-70b-versatile"
TOKEN = os.environ.get("GROQ_API_TOKEN")
TEMPERATURE = 0
MAX_COMPLETION_TOKENS = 512


def ask_llm(question: str) -> str:
    client = Groq(
        api_key=TOKEN,
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Du bist ein hilfsbereiter Assistent, der spezialisiert ist auf Fragen im Versicherungskontext.",
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        model=MODEL,
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )

    return chat_completion.choices[0].message.content


if __name__ == "__main__":
    print(ask_llm("Welche Leistung erhalte ich in meinem Tarif 'RisikofreiLeben'?"))
