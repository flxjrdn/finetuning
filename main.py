from src.rag import Rag

if __name__ == "__main__":
    rag = Rag()
    answer = rag.answer(
        "Welche Leistung erhalte ich in meinem Tarif 'RisikofreiLeben'?"
    )
    print(answer.text)
    print()
    print("context: ")
    print("\n\n".join(answer.context))
