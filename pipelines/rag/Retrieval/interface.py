"""Streamlit-based chat surface for the retrieval pipeline.

The UI currently captures user prompts and routes them to the
`handle_user_query` backend, echoing the stored record back to
the screen. As the retrieval stack matures we can swap the simple
echo response for real hybrid RAG + SQL reasoning.
"""

from __future__ import annotations

import streamlit as st

try:
    from .handler import (
        ConversationMessage,
        ConversationStore,
        HandlerResult,
        handle_user_query,
        start_new_conversation,
    )
except ImportError:  # pragma: no cover - fallback when run as a script
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pipelines.rag.Retrieval.handler import (
        ConversationMessage,
        ConversationStore,
        HandlerResult,
        handle_user_query,
        start_new_conversation,
    )


st.set_page_config(page_title="RAG Retrieval Playground", page_icon="🧠", layout="centered")
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
    conversation_path = conversation_store.path_for(st.session_state.conversation_id)
    st.caption(f"Log: {conversation_path}")
    if st.button("Start New Conversation"):
        st.session_state.conversation_id = start_new_conversation(conversation_store)
        st.session_state.messages = []
        st.rerun()


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
        result: HandlerResult = handle_user_query(
            prompt,
            conversation_id=st.session_state.conversation_id,
            enable_judge=True,
        )
        _append_messages(result.user_message, result.assistant_message)
        st.rerun()
    except ValueError as exc:
        with st.chat_message("assistant"):
            st.error(str(exc))
