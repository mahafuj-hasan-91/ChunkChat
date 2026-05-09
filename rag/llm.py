"""
rag/llm.py
──────────
Handles all Groq LLM API calls for answer generation.

RAG Concept: "Augmented Generation"
  This is the "G" in RAG. Once we've retrieved relevant chunks from the
  vector store, we "augment" the LLM's prompt with that context.

  Instead of asking the LLM: "What is X?"
  We ask: "Given the following context [chunks], answer: What is X?"

  This grounds the LLM's response in the actual document, reducing
  hallucinations and making answers factually tied to your PDF.

Why Groq?
  Groq uses custom hardware (LPUs — Language Processing Units) to run
  inference extremely fast — typically 300-500 tokens/second, compared to
  ~30-50 tokens/second on standard cloud GPUs. And it's FREE (with limits).

Model: llama3-8b-8192
  - "llama3-8b"  → Meta's LLaMA 3 model, 8 billion parameters
  - "8192"       → 8,192 token context window (fits ~3 chunks + question easily)
"""

import os
from groq import Groq
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load .env file so GROQ_API_KEY is available as an environment variable
load_dotenv()

# ─── Model Configuration ───────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024       # Max tokens in the LLM's response
TEMPERATURE = 0.2       # Low temperature = more focused, less creative answers
                        # Range: 0.0 (deterministic) → 1.0 (creative)
                        # For Q&A, 0.1-0.3 works best


def get_groq_client() -> Groq:
    """
    Initialize and return a Groq API client.

    Reads the API key from the GROQ_API_KEY environment variable.
    Set this in your .env file: GROQ_API_KEY=gsk_xxxxxxxxxxxx

    Returns:
        Groq: An authenticated Groq client instance.

    Raises:
        ValueError: If GROQ_API_KEY is not set in the environment.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Please set it in your .env file: GROQ_API_KEY=gsk_xxxxxxxxxxxx\n"
            "Get your free key at: https://console.groq.com"
        )

    return Groq(api_key=api_key)


def build_rag_prompt(question: str, context_chunks: List[str]) -> str:
    """
    Construct the RAG prompt by combining retrieved context with the question.

    RAG Concept: "Context Window Stuffing"
      We inject the retrieved chunks into the prompt as "context". The LLM
      is instructed to ONLY answer from this context, not from its training
      data. This makes answers traceable to the source document.

      Prompt structure:
        [System: Instructions on how to behave]
        [User: Context chunks + User's question]

    Args:
        question (str): The user's original question.
        context_chunks (List[str]): Top-N retrieved text chunks from the PDF.

    Returns:
        str: A fully formatted context string to inject into the prompt.
    """
    # Format each chunk with a numbered label for clarity
    formatted_chunks = "\n\n".join([
        f"[Chunk {i+1}]:\n{chunk.strip()}"
        for i, chunk in enumerate(context_chunks)
    ])

    # This is the "augmented" part of RAG — injecting retrieved knowledge
    context_prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided document context.

CONTEXT FROM DOCUMENT:
────────────────────────────────────────────
{formatted_chunks}
────────────────────────────────────────────

INSTRUCTIONS:
- Answer the question using ONLY the information from the context above.
- If the answer is not in the context, say: "I couldn't find this in the document."
- Be concise and direct. Cite relevant details from the context.
- Do NOT use outside knowledge or make up information.

QUESTION: {question}

ANSWER:"""

    return context_prompt


def generate_answer(
    question: str,
    context_chunks: List[str],
    chat_history: Optional[List[Dict]] = None
) -> str:
    """
    Send a RAG-augmented prompt to Groq and return the LLM's answer.

    The full flow:
      1. Build a prompt with context chunks + question
      2. Send to Groq's llama3-8b-8192 model via API
      3. Return the generated answer text

    Args:
        question (str): The user's question.
        context_chunks (List[str]): Retrieved document chunks (top-3).
        chat_history (Optional[List[Dict]]): Previous conversation turns.
                      Format: [{"role": "user"/"assistant", "content": "..."}]
                      Helps the LLM understand conversation context.

    Returns:
        str: The LLM-generated answer grounded in the retrieved context.

    Raises:
        Exception: If the Groq API call fails (network error, rate limit, etc.)
    """
    try:
        client = get_groq_client()

        # System message defines the LLM's behavior/persona
        system_message = {
            "role": "system",
            "content": (
                "You are a precise document Q&A assistant. "
                "You answer questions strictly based on the provided document context. "
                "Always be truthful — if the context doesn't contain the answer, say so. "
                "Keep answers clear and well-structured."
            )
        }

        # Build the user's message with retrieved context embedded
        rag_prompt = build_rag_prompt(question, context_chunks)
        user_message = {"role": "user", "content": rag_prompt}

        # Assemble messages array — system message first, then history, then current
        messages = [system_message]

        # Include recent chat history for conversational context
        # We limit to last 3 exchanges to avoid exceeding the context window
        if chat_history:
            # Convert our history format to Groq's expected format
            recent_history = chat_history[-6:]  # last 3 Q&A pairs = 6 messages
            for turn in recent_history:
                messages.append({
                    "role": turn["role"],
                    "content": turn["content"]
                })

        messages.append(user_message)

        print(f"[LLM] Sending request to Groq ({GROQ_MODEL})... "
              f"({len(context_chunks)} context chunks)")

        # Call the Groq API
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            # stream=False for simplicity; can be set to True for streaming UI
        )

        # Extract the answer text from the response
        answer = response.choices[0].message.content.strip()

        # Log token usage (useful for monitoring free tier limits)
        usage = response.usage
        print(f"[LLM] Response received. "
              f"Tokens used: {usage.prompt_tokens} prompt + "
              f"{usage.completion_tokens} completion = {usage.total_tokens} total")

        return answer

    except ValueError as e:
        # API key not set
        return f"⚠️ Configuration Error: {str(e)}"

    except Exception as e:
        error_msg = str(e)

        # Provide helpful messages for common errors
        if "rate_limit" in error_msg.lower():
            return ("⚠️ Groq rate limit reached. "
                    "Please wait a moment and try again. "
                    "(Free tier: 30 requests/minute)")
        elif "invalid_api_key" in error_msg.lower():
            return ("⚠️ Invalid Groq API key. "
                    "Check your GROQ_API_KEY in the .env file.")
        else:
            return f"⚠️ LLM Error: {error_msg}"


def test_groq_connection() -> bool:
    """
    Test that the Groq API key is valid and the API is reachable.

    Called on app startup to give early feedback if credentials are wrong.

    Returns:
        bool: True if connection succeeds, False otherwise.
    """
    try:
        client = get_groq_client()
        # Send a minimal test request
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "Say 'OK' in one word."}],
            max_tokens=5
        )
        print(f"[LLM] Groq connection test passed: "
              f"{response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"[LLM] Groq connection test failed: {e}")
        return False
