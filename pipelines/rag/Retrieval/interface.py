"""Streamlit-based chat surface for the retrieval pipeline.

The UI currently captures user prompts and routes them to the
`handle_user_query` backend, echoing the stored record back to
the screen. As the retrieval stack matures we can swap the simple
echo response for real hybrid RAG + SQL reasoning.
"""

from __future__ import annotations

import re
import time

import streamlit as st

try:
    from .app_config import AppConfig, load_and_validate
    from .handler import (
        ConversationMessage,
        ConversationStore,
        HandlerResult,
        handle_user_query,
        start_new_conversation,
    )
    from .health import check_health
except ImportError:  # pragma: no cover - fallback when run as a script
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pipelines.rag.Retrieval.app_config import AppConfig, load_and_validate
    from pipelines.rag.Retrieval.handler import (
        ConversationMessage,
        ConversationStore,
        HandlerResult,
        handle_user_query,
        start_new_conversation,
    )
    from pipelines.rag.Retrieval.health import check_health


st.set_page_config(page_title="RAG Retrieval Playground", page_icon="🧠", layout="centered")


def _load_startup_config() -> AppConfig:
    if "_app_config" not in st.session_state:
        st.session_state["_app_config"] = load_and_validate()
    return st.session_state["_app_config"]


try:
    app_config = _load_startup_config()
except EnvironmentError as exc:
    st.error(str(exc))
    st.stop()

if "health" not in st.session_state:
    st.session_state.health = check_health()

chatbot_password = app_config.chatbot_password
if chatbot_password:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "auth_attempts" not in st.session_state:
        st.session_state.auth_attempts = 0

    if st.session_state.authenticated:
        last_activity = st.session_state.get("last_activity")
        if last_activity is not None and time.time() - float(last_activity) > 1800:
            st.session_state.authenticated = False
            st.session_state.auth_attempts = 0
            st.session_state["_session_expired"] = True

    if st.session_state.auth_attempts >= 5 and not st.session_state.authenticated:
        st.title("RAG Retrieval Playground")
        st.error("Too many failed attempts. Please restart the session.")
        st.stop()

    if not st.session_state.authenticated:
        st.title("RAG Retrieval Playground")
        if st.session_state.get("_session_expired"):
            st.warning("Session expired. Please log in again.")
        entered_password = st.text_input("Password", type="password")
        if st.button("Login"):
            if entered_password == chatbot_password:
                st.session_state.authenticated = True
                st.session_state.auth_attempts = 0
                st.session_state.last_activity = time.time()
                st.session_state["_session_expired"] = False
                st.rerun()
            else:
                st.session_state.auth_attempts += 1
                if st.session_state.auth_attempts >= 5:
                    st.error("Too many failed attempts. Please restart the session.")
                else:
                    st.error("Incorrect password.")
        st.stop()
st.title("RAG Retrieval Playground")
st.caption(
    "Experiment with hybrid retrieval ideas. Conversations are persisted so we can wire"
    " decision logic and replay flows later."
)


conversation_store = ConversationStore()

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = start_new_conversation(conversation_store)

if "messages" not in st.session_state:
    st.session_state.messages = list(
        conversation_store.load(st.session_state.conversation_id)
    )


def _append_messages(*msgs: ConversationMessage) -> None:
    st.session_state.messages.extend(msgs)


with st.sidebar:
    st.subheader("Conversations")
    st.text(f"Active ID:\n{st.session_state.conversation_id}")
    if st.session_state.get("user_zip"):
        st.caption(f"ZIP: {st.session_state.user_zip}")
        if st.button("Change ZIP"):
            st.session_state.pop("user_zip", None)
            st.rerun()
    conversation_path = conversation_store.path_for(st.session_state.conversation_id)
    st.caption(f"Log: {conversation_path}")
    if st.button("Start New Conversation"):
        st.session_state.conversation_id = start_new_conversation(conversation_store)
        st.session_state.messages = []
        st.rerun()

    health = st.session_state.get("health", {})

    def _render_bool_status(label: str, value: bool) -> None:
        color = "#16a34a" if value else "#dc2626"
        symbol = "✓" if value else "✗"
        st.markdown(
            f"{label} <span style='color:{color}'>{symbol}</span>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("System Status")
    _render_bool_status("OpenAI Key:", bool(health.get("openai_key")))
    _render_bool_status("Encryption:", bool(health.get("conversation_key")))
    _render_bool_status("Ford DB:", bool(health.get("ford_db")))
    _render_bool_status("Vector Index:", bool(health.get("chroma_index")))
    st.markdown(f"Guardrails: `{health.get('guardrail_config', 'default')}`")

if not st.session_state.get("user_zip"):
    with st.chat_message("assistant"):
        st.write("Welcome to Ford Assistant. To show you nearby dealer offers, could you share your ZIP code?")
    zip_input = st.text_input("ZIP Code", max_chars=5, key="zip_capture_input")
    if st.button("Set Location"):
        entered_zip = (zip_input or "").strip()
        if re.match(r"^\d{5}$", entered_zip):
            st.session_state.user_zip = entered_zip
            st.rerun()
        else:
            st.warning("Please enter a valid 5-digit ZIP code.")
    st.stop()


messages: list[ConversationMessage] = st.session_state.messages

if not messages:
    with st.chat_message("assistant"):
        st.write(
            "Ask about your data sources or request SQL-powered insights. We'll log the prompt now"
            " and connect it to the right retrieval route soon."
        )
else:
    for message in messages:
        with st.chat_message(message.role):
            st.write(message.content)


prompt = st.chat_input("Type your question")

if prompt:
    try:
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result: HandlerResult = handle_user_query(
                    prompt,
                    conversation_id=st.session_state.conversation_id,
                    user_zip=st.session_state.get("user_zip"),
                    enable_judge=True,
                )
            st.session_state.last_activity = time.time()
            st.write(result.assistant_message.content)

        _append_messages(result.user_message, result.assistant_message)
        st.rerun()
    except ValueError as exc:
        with st.chat_message("assistant"):
            st.error(str(exc))
