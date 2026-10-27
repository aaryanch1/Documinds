# DocuMind AI — Groq Edition

Document question-answering application with a Streamlit interface and optional FastAPI backend.

## Setup

1. Install Python 3.10+ (or a version supported by the listed dependencies).
2. Install packages:
   ```bash
   python -m pip install -r requirements_groq.txt
   ```
3. Create a local `.env` file by copying `.env.example`, then set `GROQ_API_KEY` to your own key. **Never commit `.env`.**
4. Start the Streamlit UI:
   ```bash
   streamlit run documind_groq.py
   ```
5. Optional REST API:
   ```bash
   uvicorn documind_groq:api --port 8000
   ```

The source code contains no embedded API key. If no key is configured, document upload and local processing may still work, but AI answers require a valid key and network access.

## GitHub safety

- `.gitignore` excludes `.env`, local data, logs, and Python cache files.
- `.env.example` is safe to commit because it contains only a placeholder.
- If a real API key was previously pasted into a chat or committed, revoke/rotate it in the provider dashboard before using the project publicly. Removing it from the latest file does not remove it from Git history.
