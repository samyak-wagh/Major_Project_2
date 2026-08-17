import os
import torch
import warnings
from dotenv import load_dotenv

# Suppress warnings for cleaner terminal output
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Import your existing RAG components (this connects to Qdrant automatically!)
from main import load_config, init_components
from rag.qa_chain import SYSTEM_PROMPT, HUMAN_PROMPT

def main():
    load_dotenv()
    
    print("🚀 Initializing Local RAG Testing Environment...")
    
    # 1. Initialize existing Vector Store (No re-ingestion needed!)
    print("📚 Connecting to existing Qdrant Vector Store...")
    cfg = load_config()
    qdrant_store, embeddings, processor = init_components(cfg)
    retriever = qdrant_store.get_retriever(k=cfg.search_k)
    
    # 2. Load Fine-Tuned Model
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter = "rohit21789/OS-tutor"

    print(f"🧠 Loading Tokenizer: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print(f"🧠 Loading Base Model: {base_model_name} (this might take a moment)...")
    # We use torch.float16 and device_map="auto" to ensure it runs fast and fits in memory
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    print(f"🔥 Loading LoRA Adapter: {adapter}...")
    model = PeftModel.from_pretrained(base_model, adapter)

    # 3. Create LangChain HuggingFace Pipeline
    print("⚙️ Building the Text-Generation Pipeline...")
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,       # Limit answer length so it doesn't ramble
        temperature=0.1,          # Keep it deterministic and factual
        do_sample=True,
        repetition_penalty=1.1,
    )

    local_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    # 4. Rebuild the QA Chain using the local LLM
    print("🔗 Connecting Local LLM to the Retrieval Chain...")
    qa_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(HUMAN_PROMPT),
    ])

    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=local_llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        verbose=False,
    )
    
    print("\n" + "="*60)
    print("✅ Local Fine-Tuned OS Tutor Ready!")
    print("Type 'exit' or 'quit' to stop.")
    print("="*60)

    # 5. Interactive Terminal Loop
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            if not user_input.strip():
                continue
                
            print("\nThinking... (Searching Qdrant & Generating Local Answer)\n")
            response = chain.invoke({"question": user_input})
            
            # Extract and format answer
            answer = response.get("answer", "")
            # Sometimes TinyLlama includes the prompt in the output, this cleans it up if it does
            if "Student's question:" in answer:
                answer = answer.split("Student's question:")[-1].split("Answer:")[-1].strip()
                
            print(f"🤖 OS Tutor: {answer}")
            
            # Print sources
            docs = response.get("source_documents", [])
            if docs:
                print("\nSources:")
                seen = set()
                for doc in docs:
                    meta = doc.metadata
                    page = meta.get("page", "?")
                    # Pages are 0-indexed in PyPDF, so we add 1 for human readability
                    if isinstance(page, int):
                        page += 1
                    key = f"{meta.get('source_filename', 'Unknown')} - page {page}"
                    if key not in seen:
                        print(f"  • {key}")
                        seen.add(key)
                        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
