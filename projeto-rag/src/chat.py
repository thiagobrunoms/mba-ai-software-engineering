from search import Config, RAGChain

if __name__ == "__main__":
    config = Config()
    chain = RAGChain(config)

    print("Chat RAG iniciado. Digite 'sair' para encerrar.\n")

    while True:
        query = input("Você: ").strip()
        if query.lower() == "sair":
            print("Encerrando chat. Até logo!")
            break
        if not query:
            continue
        print(f"\nAssistente: {chain.run(query)}\n")
