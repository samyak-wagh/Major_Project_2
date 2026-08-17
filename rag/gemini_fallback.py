"""
gemini_fallback.py — Gemini Fallback Handler
============================================
Handles out-of-context queries. If the primary model (Groq or TinyLlama)
cannot find the answer in the textbook context, it routes the user's
question to Gemini for a general knowledge response.
"""

import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

def ask_gemini(question: str) -> str:
    """
    Sends the question to Gemini when it is out of the scope of the textbook.
    """
    api_key = os.getenv("GEMINI_PAID_API_KEY", "")
    if not api_key:
        logger.error("GEMINI_PAID_API_KEY is not set.")
        return "I don't know the answer to that based on the textbook, and the Gemini fallback is not configured (missing API key)."

    try:
        # We can use gemini-1.5-flash as it's fast and suitable for general knowledge fallback
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )
        
        prompt = (
            "You are a helpful and intelligent AI tutor. "
            "A student asked the following question which was not covered in their textbook. "
            "Please provide a clear, accurate, and concise answer to their question.\n\n"
            f"Question: {question}"
        )
        
        response = llm.invoke(prompt)
        return response.content.strip()

    except Exception as e:
        logger.error("Error in Gemini fallback: %s", e)
        return "I don't know the answer to that based on the textbook, and the Gemini fallback encountered an error."
