Edited rag_plan.md
Viewed research.md:1-20

Here is a complete, industry-standard tech stack tailored specifically for a Major Project. I have selected these tools because they are highly respected by evaluators, easy to use in Python, and have excellent free tiers for students.

### 1. Document Extraction & Parsing (Handling the Galvin PDF)
*This is where you extract text, tables, and code snippets from the PDF.*
* **Top Choice:** **LlamaParse** (by LlamaIndex)
  * **Why:** It uses AI to read the PDF and is specifically trained to understand complex textbooks, tables, and code blocks perfectly. It is the best tool right now for difficult PDFs.
* **Open-Source Alternative:** **PyMuPDF (`fitz`)**
  * **Why:** If you want to write the extraction script yourself completely offline, this is the fastest and most accurate traditional Python PDF library.

### 2. Text Chunking & Preprocessing
*Breaking the 1000 pages into smaller, overlapping chunks (e.g., 500 tokens).*
* **Top Choice:** **LangChain (`RecursiveCharacterTextSplitter`)**
  * **Why:** It is the industry standard. It intelligently splits text at paragraph or sentence boundaries so it doesn't cut a concept in half.

### 3. Embedding Model
*Converting the text chunks into numbers (vectors).*
* **Top Choice (Cloud):** **OpenAI `text-embedding-3-small`**
  * **Why:** Costs almost nothing (less than 2 cents for the whole book), extremely fast, and incredibly accurate.
* **Top Choice (Local/Free):** **HuggingFace (`BAAI/bge-small-en-v1.5`)**
  * **Why:** If you want your project to run 100% locally without internet or API keys, this open-source model is small, fast, and highly ranked on the MTEB leaderboard.

### 4. Vector Database
*Storing the 2,000 vectors so you can search them in milliseconds.*
* **Top Choice:** **ChromaDB**
  * **Why:** Perfect for Major Projects. It runs locally on your machine and saves the database into a simple folder. No need to set up cloud accounts or complicated Docker containers.
* **Cloud Alternative:** **Pinecone**
  * **Why:** If you want to host your database in the cloud, Pinecone is the industry standard and has a very generous free tier.

### 5. Orchestration Framework
*The "glue" that connects the Vector DB, the prompt, and the LLM together.*
* **Top Choice:** **LangChain**
  * **Why:** Knowing LangChain is a huge resume booster. It has built-in functions for "RetrievalQA" chains and memory (so the bot remembers the chat history).
* **Alternative:** **LlamaIndex**
  * **Why:** Specifically built for RAG. It is often simpler to set up for strictly document-reading tasks than LangChain.

### 6. The LLM (The Chatbot's Brain)
*Takes the context from the Vector DB and answers the student's question.*
* **Top Choice (Cloud):** **Groq (Running `Llama-3-8b`)** or **OpenAI (`gpt-4o-mini`)**
  * **Why:** Groq provides lightning-fast inference (instant answers) and has a great free tier. `gpt-4o-mini` is extremely cheap and strictly follows instructions (good for preventing "internet answers").
* **Top Choice (Local/Offline):** **Ollama**
  * **Why:** Allows you to run LLMs locally on your laptop. Great if your laptop has a decent GPU and you want to show the examiners an entirely offline system.

### 7. Backend & API
*Serving the chatbot logic.*
* **Top Choice:** **FastAPI (Python)**
  * **Why:** It is modern, incredibly fast, and automatically generates API documentation. Evaluators love seeing FastAPI instead of older frameworks like Flask.

### 8. Frontend / User Interface
*What the user actually sees and interacts with.*
* **Top Choice:** **Streamlit**
  * **Why:** You can build a beautiful, ChatGPT-like interface in pure Python in about 50 lines of code. No HTML/CSS/React required. It is the absolute best tool for AI project frontends.

---

### Summary of the Best Stack for Your Project:
1. Parse the Galvin PDF with **LlamaParse**.
2. Chunk it using **LangChain**.
3. Embed and store it locally using **ChromaDB**.
4. Build the API with **FastAPI**.
5. Connect it all to **GPT-4o-mini** (with a strict system prompt).
6. Build the Chat UI using **Streamlit**.