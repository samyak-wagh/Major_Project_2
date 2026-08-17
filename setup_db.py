from main import load_config, init_components, ingest_pdf
import sys

def main():
    print("🔄 Connecting to Qdrant Database...")
    cfg = load_config()
    qdrant_store, embeddings, processor = init_components(cfg)

    print("🗑️  Wiping existing collection (removing sample.pdf)...")
    qdrant_store.delete_collection()

    print("📚 Ingesting the Galvin Textbook (ostxtbook.pdf)...")
    print("⏳ This might take a few minutes. Please wait...")
    
    # Ingest the actual OS textbook
    success = ingest_pdf("ostxtbook.pdf", cfg, qdrant_store, embeddings, processor)
    
    if success:
        print("✅ Database successfully built with the OS Textbook!")
        print("🚀 You can now restart your FastAPI backend.")
    else:
        print("❌ Failed to ingest the textbook.")

if __name__ == "__main__":
    main()
