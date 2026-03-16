# app.py
from src.generation import generate_answer

def main():
    print("=" * 50)
    print("  DocBot — HR Policy Assistant")
    print("  Type 'quit' to exit")
    print("=" * 50 + "\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            print("Goodbye!")
            break

        print("\nBot: ", end="", flush=True)
        answer = generate_answer(question)
        print(answer)
        print()

if __name__ == "__main__":
    main()