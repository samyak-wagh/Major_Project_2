# 📊 OS Tutor Diagram Generation Feature Plan

## Objective
To implement a deterministic, prompt-based diagram generation feature in the OS Tutor Streamlit app. This feature will allow students to ask for exact numerical/structural visualizations (e.g., CPU scheduling Gantt charts, Resource Allocation Graphs, Page Tables), which are processed by an LLM into structured code (Graphviz/JSON) and rendered perfectly by Python.

## The Problem with Traditional Text-to-Image AI
Standard image generation models (like Stable Diffusion or FLUX) are terrible at rendering exact text and precise numerical relationships. If asked for a "Gantt chart where P1 starts at 0 and ends at 5", they hallucinate numbers and draw abstract, inaccurate pictures.

## Our Solution: The Code-to-Diagram Pipeline
Instead of generating an image directly, the LLM generates structured data representing the diagram. 

### Supported Diagram Types
1. **Graphviz (DOT Language)**
   - **Use Cases:** Resource Allocation Graphs (Deadlocks), Process State Machines, Tree Structures.
   - **How:** The LLM outputs `digraph G { ... }`. Streamlit natively renders it using `st.graphviz_chart()`.
2. **Plotly / Matplotlib (JSON to Chart)**
   - **Use Cases:** CPU Scheduling Gantt Charts, Memory Allocation blocks.
   - **How:** The LLM outputs a JSON array `[{"task": "P1", "start": 0, "end": 5}]`. Python parses the JSON and renders a beautiful Plotly Gantt chart (`st.plotly_chart()`).

## Proposed Implementation Steps

### Step 1: Update API Endpoint (`api.py`)
- Add a new "intent detection" mechanism to determine if the user's prompt is asking for a diagram.
- If a diagram is requested, route the prompt to a specialized `DiagramGenerator` LLM chain (using the Groq API for extremely fast code generation).
- The `DiagramGenerator` responds with a JSON payload indicating the `type` (e.g., "graphviz" or "gantt") and the `code` (the raw DOT code or JSON array).

### Step 2: Streamlit UI Integration (`app.py`)
- Modify the chat message rendering logic in Streamlit.
- When the backend returns a `diagram` object in the response:
  - If `type == "graphviz"`, use Streamlit's built-in `st.graphviz_chart(code)`.
  - If `type == "gantt"`, use the `plotly` library to draw a timeline chart and call `st.plotly_chart(figure)`.

### Step 3: Prompt Engineering
Create strict system prompts for the Groq model:
- **Graphviz Prompt:** "You are a Graphviz DOT expert. Convert the user's OS problem into a DOT graph. Output ONLY valid DOT code. No markdown formatting."
- **Gantt Prompt:** "You are a JSON generator. Convert the user's CPU scheduling problem into a JSON array with keys 'Task', 'Start', 'Finish'. Output ONLY valid JSON."

## IEEE Paper Impact
This feature provides a massive boost to the project's academic novelty. You can state in the paper:

> *"To overcome the hallucination of text and numerical data inherent in diffusion-based generative AI, OS Tutor implements a deterministic Code-to-Diagram pipeline. By utilizing Large Language Models purely as semantic parsers to extract state variables (into DOT and JSON formats), the system deterministically renders complex operating system constructs—such as Resource Allocation Graphs and CPU scheduling Gantt charts—with 100% numerical and structural fidelity."*
