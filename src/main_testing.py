from loaders.awq_loader import AWQ_ModelLoader


def main():
    tokenizer, model = AWQ_ModelLoader.load_model()
    print("✅ AWQ model and tokenizer loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
