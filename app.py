import streamlit as st
import requests
import time

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="OS Tutor AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Dark UI ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Hide default chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Page background */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    min-height: 100vh;
}

/* Main container */
.block-container {
    max-width: 900px;
    padding-top: 1.5rem;
    padding-bottom: 6rem;
}

/* ── Header ── */
.os-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f3460 50%, #1a1f2e 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.os-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(88,166,255,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.os-header h1 {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    background: linear-gradient(90deg, #58a6ff, #79c0ff, #a5f3fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.os-header p {
    color: #8b949e;
    font-size: 0.95rem;
    margin: 0;
}
.os-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(88,166,255,0.1);
    border: 1px solid rgba(88,166,255,0.3);
    color: #58a6ff;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-top: 0.6rem;
}

/* ── Chat messages ── */
.stChatMessage {
    background: transparent !important;
    border: none !important;
}
[data-testid="stChatMessageContent"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
    color: #e6edf3 !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
}
[data-testid="stChatMessageContent"] p { color: #e6edf3 !important; }

/* ── Chat input ── */
.stChatInputContainer {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
}
.stChatInputContainer:focus-within {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #30363d;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8b949e !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(31,111,235,0.4) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(31,111,235,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(31,111,235,0.5) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117, #161b22) !important;
    border-right: 1px solid #30363d !important;
}
[data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label {
    color: #c9d1d9 !important;
}

/* ── Status boxes ── */
.stSuccess { background: rgba(35,134,54,0.15) !important; border: 1px solid rgba(35,134,54,0.4) !important; border-radius: 8px !important; }
.stWarning { background: rgba(187,128,9,0.15) !important; border: 1px solid rgba(187,128,9,0.4) !important; border-radius: 8px !important; }
.stError   { background: rgba(218,54,51,0.15) !important; border: 1px solid rgba(218,54,51,0.4) !important; border-radius: 8px !important; }
.stInfo    { background: rgba(31,111,235,0.15) !important; border: 1px solid rgba(31,111,235,0.4) !important; border-radius: 8px !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #8b949e !important;
    font-size: 0.85rem !important;
}

/* ── Text input ── */
.stTextInput > div > div > input, .stTextArea textarea {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
}
.stTextInput > div > div > input:focus, .stTextArea textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.2) !important;
}

/* Divider */
hr { border-color: #21262d !important; }

/* Caption text */
.stCaption { color: #6e7681 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
if "llm_backend" not in st.session_state:
    st.session_state.llm_backend = "local"
if "groq_model" not in st.session_state:
    st.session_state.groq_model = "llama-3.3-70b-versatile"

_backend_label = (
    f"⚡ Groq / {st.session_state.groq_model}"
    if st.session_state.llm_backend == "groq"
    else "🤖 Local · TinyLlama + OS-tutor LoRA · Offline"
)
st.markdown(f"""
<div class="os-header">
    <h1>🎓 OS Tutor AI</h1>
    <p>Your intelligent assistant for <strong>Operating Systems</strong> — grounded in the Galvin textbook.</p>
    <span class="os-badge">🔬 {_backend_label}</span>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "👋 Hello! I'm **OS Tutor**, trained on the Galvin *Operating Systems* textbook.\n\n"
            "Ask me anything about:\n"
            "- 🔒 **Deadlocks** — conditions, prevention, banker's algorithm\n"
            "- ⚙️ **CPU Scheduling** — Round Robin, FCFS, SJF\n"
            "- 🧵 **Threads & Processes** — user/kernel threads, PCB\n"
            "- 💾 **Memory Management** — paging, segmentation, TLB\n"
            "- 📁 **File Systems** — FAT, inode, directory structures\n\n"
            "You can type, 🎤 speak, or 🖼️ upload an image of your notes!"
        ),
    }]

if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""

if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Backend selector ──────────────────────────────────────────────────────
    st.markdown("### 🧠 LLM Backend")
    backend_choice = st.radio(
        "Choose model backend",
        options=["local", "groq"],
        format_func=lambda x: "🤖 Local (TinyLlama — Offline)" if x == "local" else "⚡ Groq (Cloud — Fast)",
        index=0 if st.session_state.llm_backend == "local" else 1,
        key="backend_radio",
        label_visibility="collapsed",
    )
    st.session_state.llm_backend = backend_choice

    if backend_choice == "groq":
        from rag.groq_chain import GROQ_MODELS  # noqa: E402
        groq_model_choice = st.selectbox(
            "Groq model",
            options=list(GROQ_MODELS.keys()),
            format_func=lambda k: GROQ_MODELS[k],
            index=list(GROQ_MODELS.keys()).index(st.session_state.groq_model)
                  if st.session_state.groq_model in GROQ_MODELS else 0,
            key="groq_model_select",
        )
        st.session_state.groq_model = groq_model_choice
        st.caption("Free key at [console.groq.com](https://console.groq.com)")
    else:
        st.caption("Runs fully offline after first HuggingFace download.")

    st.markdown("---")
    st.markdown("### 📡 Backend Status")
    try:
        # Increase timeout slightly to allow for model initialization start
        status_res = requests.get(f"{API_URL}/status", timeout=15)
        if status_res.status_code == 200:
            data = status_res.json()
            chain_ready = data.get("chain_ready", False)
            if chain_ready:
                st.success("✅ Model loaded & ready")
                docs = data.get("documents", [])
                if docs:
                    st.info(f"📚 {len(docs)} document(s) in knowledge base")
                    for d in docs:
                        st.caption(f"📄 {d}")
            else:
                st.warning("⏳ Model is loading... Please wait.")
    except requests.exceptions.ReadTimeout:
        st.warning("⏳ Initializing local AI model... This takes ~30 seconds on first run.")
    except requests.exceptions.ConnectionError:
        st.error("❌ Backend offline")
        st.caption("Run: `python api.py` in a terminal")
    except Exception as e:
        st.error(f"❌ {e}")

    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    examples = [
        "What are the four conditions for deadlock?",
        "Explain the Banker's algorithm.",
        "Difference between mutex and semaphore?",
        "How does the TLB work?",
        "What is Belady's Anomaly?",
        "Explain Round Robin scheduling.",
        "What are user-level vs kernel-level threads?",
    ]
    for ex in examples:
        st.markdown(f"<span style='color:#58a6ff;font-size:0.85rem;'>▸</span> *{ex}*", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📤 Upload a PDF")
    uploaded_file = st.file_uploader("Add to knowledge base", type=["pdf"])
    if uploaded_file:
        with st.spinner("Ingesting PDF..."):
            try:
                res = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                    timeout=300,
                )
                if res.status_code == 200:
                    st.success(res.json().get("message", "Ingested!"))
                else:
                    st.error(res.json().get("detail", "Upload failed."))
            except Exception as e:
                st.error(f"Upload error: {e}")

# ── Chat history ──────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 View Source References"):
                for src in message["sources"]:
                    st.markdown(
                        f"<span style='color:#58a6ff;font-size:0.88em;'>📄 {src}</span>",
                        unsafe_allow_html=True,
                    )

# ── Helper: send question ─────────────────────────────────────────────────────
def send_question(prompt: str):
    if not prompt.strip():
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        backend = st.session_state.get("llm_backend", "local")
        if backend == "groq":
            placeholder.markdown("*⚡ Querying Groq...*")
        else:
            placeholder.markdown("*⏳ Thinking... (30–90s on CPU)*")

        try:
            payload = {
                "question":   prompt,
                "backend":    backend,
                "groq_model": st.session_state.get("groq_model", "") if backend == "groq" else "",
            }
            res = requests.post(f"{API_URL}/ask", json=payload, timeout=600)

            if res.status_code == 200:
                data = res.json()
                answer  = data["answer"]
                sources = data.get("sources", [])

                full_response = ""
                placeholder.empty()
                for word in answer.split(" "):
                    full_response += word + " "
                    placeholder.markdown(full_response + "▌")
                    time.sleep(0.025)
                placeholder.markdown(full_response.strip())

                if sources and not any("no source" in s.lower() for s in sources):
                    with st.expander("📚 View Source References"):
                        for src in sources:
                            st.markdown(
                                f"<span style='color:#58a6ff;font-size:0.88em;'>📄 {src}</span>",
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response.strip(),
                    "sources": sources,
                })
            else:
                err = res.json().get("detail", "Unknown error.")
                placeholder.error(f"⚠️ {err}")

        except requests.exceptions.ConnectionError:
            placeholder.error("❌ Cannot connect to backend. Run `python api.py` first.")
        except requests.exceptions.Timeout:
            placeholder.error("⏳ Timed out. Try a shorter question on CPU.")
        except Exception as e:
            placeholder.error(f"❌ {e}")

# ── Input Section ─────────────────────────────────────────────────────────────
st.markdown("---")
tab_text, tab_voice, tab_image = st.tabs(["📝  Text", "🎤  Voice", "🖼️  Image (OCR)"])

# ── Text Tab ──────────────────────────────────────────────────────────────────
with tab_text:
    if prompt := st.chat_input("Ask a question about Operating Systems...", key="text_input"):
        send_question(prompt)

# ── Voice Tab ─────────────────────────────────────────────────────────────────
with tab_voice:
    st.markdown("#### 🎤 Ask by Voice")
    st.caption("Click the mic, speak your question, then click stop. Needs internet only for speech-to-text.")
    
    audio_input = st.audio_input("Record your question", key="voice_recorder")

    if audio_input is not None:
        with st.spinner("🔄 Transcribing..."):
            try:
                from rag.voice_handler import transcribe_audio
                transcript = transcribe_audio(audio_input.read())
                if transcript:
                    st.success(f"✅ Heard: **\"{transcript}\"**")
                    st.session_state.voice_transcript = transcript
                else:
                    st.warning("⚠️ Couldn't understand. Speak clearly and try again.")
            except RuntimeError as e:
                st.error(f"❌ {e}")

    if st.session_state.voice_transcript:
        col1, col2 = st.columns([4, 1])
        with col1:
            edited = st.text_input("Edit if needed:", value=st.session_state.voice_transcript, key="voice_edit")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Send", key="send_voice", use_container_width=True):
                send_question(edited)
                st.session_state.voice_transcript = ""
                st.rerun()

# ── Image Tab ─────────────────────────────────────────────────────────────────
with tab_image:
    st.markdown("#### 🖼️ Ask from an Image")
    st.caption("Upload a photo of handwritten notes, textbook pages, or OS diagrams. OCR extracts the text, then you can ask a question about it.")

    img_file = st.file_uploader("Upload image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg", "bmp", "webp"], key="image_upload")

    if img_file:
        col_img, col_text = st.columns([1, 1])
        with col_img:
            st.image(img_file, caption="Uploaded Image", use_container_width=True)

        with col_text:
            with st.spinner("🔍 Extracting text from image..."):
                try:
                    from rag.image_handler import extract_text_from_image
                    img_file.seek(0)
                    ocr_text = extract_text_from_image(img_file.read())
                    if ocr_text:
                        st.success("✅ Text extracted!")
                        # Automatically frame it as a question
                        default_question = f"Explain these Operating System concepts from the diagram: {ocr_text}"
                        st.session_state.ocr_text = default_question
                    else:
                        st.warning("⚠️ No text found. Try a clearer image.")
                        st.session_state.ocr_text = ""
                except Exception as e:
                    st.error(f"❌ OCR failed: {e}")

        if st.session_state.ocr_text:
            st.markdown("**📋 Question (editable):**")
            edited_ocr = st.text_area(
                label="Question",
                value=st.session_state.ocr_text,
                height=100,
                key="ocr_edit",
                label_visibility="collapsed",
            )
            if st.button("🚀 Send to OS Tutor", key="send_image", use_container_width=True):
                send_question(edited_ocr)
                st.session_state.ocr_text = ""
                st.rerun()
