import streamlit as st
import uuid
from typing import Dict, Any, List
import re
import os 

# --- Project imports ---
import config
from src.graph.factory import get_graph_app
from src.schema.constants import NodeName, SimpleQuery, WorkflowSignal
from src.schema.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# --- Page config ---
st.set_page_config(
    page_title="Agent Expert",
    page_icon="🤖",
    layout="wide"
)

# --- UI ---
st.title("🤖 Agent Expert")
st.markdown("""
This app is a multi-agent Q&A system built with LangGraph. Enter a question and each team will work step by step to produce a final answer.
- **Team Query**: Analyzes the user question and creates search queries.
- **Team Search**: Collects and evaluates information through RAG and web research.
- **Team Answer**: Generates and reviews the final answer from the collected context.
""")

with st.sidebar:
    st.header("Runtime")
    st.caption("Current local configuration")
    st.metric("Web research", "Enabled" if config.ENABLE_WEB_RESEARCH else "Disabled")
    st.metric("Vector store", config.VECTOR_STORE_PATH)
    st.metric("OCR language", config.OCR_LANG)

    if st.button("Reset chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# --- Load LangGraph app ---
app = get_graph_app()

# --- Session state for chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Display previous chat history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def _last_named_message(messages: List[Any], name: str):
    return next((msg for msg in reversed(messages) if getattr(msg, "name", None) == name), None)


def summarize_progress(messages: List[Any]) -> Dict[str, Any]:
    """Build structured progress details from LangGraph messages."""
    query_eval = _last_named_message(messages, NodeName.QUERY_EVALUATOR)
    retrieval_eval = _last_named_message(messages, NodeName.RETRIEVAL_EVALUATOR)
    answer_eval = _last_named_message(messages, NodeName.ANSWER_EVALUATOR)
    rag_result = _last_named_message(messages, NodeName.RAG_SEARCH_RESULT)
    web_result = _last_named_message(messages, NodeName.WEB_SEARCH_RESULT)
    answer_started = any(isinstance(msg, AIMessage) for msg in messages)

    is_simple = False
    best_query = ""
    output_format = ""
    if query_eval and query_eval.content == WorkflowSignal.PASS:
        best_query = query_eval.additional_kwargs.get("best_rag_query", "")
        output_format = ", ".join(query_eval.additional_kwargs.get("output_format", []))
        is_simple = not retrieval_eval and not rag_result and answer_started

    rag_count = 0
    web_count = 0
    if retrieval_eval:
        rag_count = retrieval_eval.additional_kwargs.get("accepted_rag", 0)
        web_count = retrieval_eval.additional_kwargs.get("accepted_web", 0)

    return {
        "query_eval": query_eval,
        "retrieval_eval": retrieval_eval,
        "answer_eval": answer_eval,
        "rag_result": rag_result,
        "web_result": web_result,
        "answer_started": answer_started,
        "is_simple": is_simple,
        "best_query": best_query,
        "output_format": output_format,
        "rag_count": rag_count,
        "web_count": web_count,
    }


def render_progress(messages: List[Any]) -> None:
    summary = summarize_progress(messages)

    st.markdown("### Progress")
    cols = st.columns(3)

    query_eval = summary["query_eval"]
    query_state = "Running" if not query_eval else ("Done" if query_eval.content == WorkflowSignal.PASS else "Failed")
    cols[0].metric("Query Team", query_state)

    retrieval_eval = summary["retrieval_eval"]
    if summary["is_simple"]:
        retrieval_state = "Skipped"
    elif retrieval_eval:
        retrieval_state = "Done" if retrieval_eval.content == WorkflowSignal.PASS else "Stopped"
    elif summary["rag_result"] or summary["web_result"]:
        retrieval_state = "Running"
    elif query_eval and query_eval.content == WorkflowSignal.PASS:
        retrieval_state = "Pending"
    else:
        retrieval_state = "Waiting"
    cols[1].metric("Retrieval Team", retrieval_state)

    answer_eval = summary["answer_eval"]
    if answer_eval:
        answer_state = "Done" if answer_eval.content == WorkflowSignal.PASS else "Needs revision"
    elif summary["answer_started"]:
        answer_state = "Running"
    else:
        answer_state = "Waiting"
    cols[2].metric("Answer Team", answer_state)

    details = []
    if summary["best_query"]:
        details.append(f"Best query: `{summary['best_query']}`")
    if summary["output_format"]:
        details.append(f"Output format: `{summary['output_format']}`")
    if retrieval_eval:
        details.append(f"Accepted docs: RAG `{summary['rag_count']}`, Web `{summary['web_count']}`")
    if summary["is_simple"]:
        details.append("Retrieval skipped because this was classified as a simple query.")

    if details:
        st.markdown("\n".join(f"- {detail}" for detail in details))

# --- Main flow: process user input and run graph ---
if prompt := st.chat_input("Enter your question."):
    # Add the user message to chat history and display it.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Start assistant response handling.
    with st.chat_message("assistant"):
        final_answer = ""
        error_message = ""
        
        # UI placeholders for progress and answer output.
        progress_placeholder = st.empty()
        answer_placeholder = st.empty()

        try:
            # Initial state for graph execution.
            initial_state: AgentState = {
                "messages": [HumanMessage(content=prompt)],
                "team1_retries": 0,
                "team2_retries": 0,
                "team3_retries": 0,
                "global_loop_count": 0,
                "is_simple_query": SimpleQuery.NO
            }
            thread = {"configurable": {"thread_id": st.session_state.thread_id}}

            # Process real-time events from the LangGraph stream.
            final_state_messages = []
            for event in app.stream(initial_state, thread, stream_mode="values"):
                # Read messages from the current event.
                messages = event.get("messages", [])
                final_state_messages = messages
                
                # Update progress from messages.
                with progress_placeholder.container():
                    render_progress(messages)
                
            # Process final result.
            if final_state_messages:
                # Treat the last AIMessage as the final answer.
                for msg in reversed(final_state_messages):
                    if isinstance(msg, AIMessage):
                        final_answer = msg.content
                        break
                
                # If no final answer was found, inspect the last message for an error.
                if not final_answer:
                    last_msg = final_state_messages[-1]
                    if last_msg.content != WorkflowSignal.PASS:
                        error_message = f"The workflow ended with a failure. (last step: {last_msg.name}, reason: {last_msg.content})"

        except Exception as e:
            st.error(f"An exception occurred while running the system: {e}")
            error_message = f"System error: {e}"

        # Clear progress UI and show the final result.
        progress_placeholder.empty()
        
        if final_answer:
            # Split Markdown table content from generated image path.
            image_path_marker = "**[View generated table image]"
            
            if image_path_marker in final_answer:
                parts = final_answer.split(image_path_marker)
                markdown_content = parts[0].strip()
                link_part = parts[1]

                # Extract the file path inside parentheses.
                match = re.search(r'\((.*?)\)', link_part)
                if match:
                    image_path = match.group(1)
                    
                    # 1. Display the Markdown table.
                    answer_placeholder.markdown(markdown_content)
                    
                    # 2. Display the generated image from the extracted path.
                    if os.path.exists(image_path):
                        st.image(image_path, caption="Generated table image")
                    else:
                        st.warning(f"Image file not found: {image_path}")
                else:
                    # If link format is invalid, display the full original answer.
                    answer_placeholder.markdown(final_answer)
            else:
                # Display ordinary answers without table images as-is.
                answer_placeholder.markdown(final_answer)

            st.session_state.messages.append({"role": "assistant", "content": final_answer})
        elif error_message:
            st.error(f"Failed to generate an answer: {error_message}")
            st.session_state.messages.append({"role": "assistant", "content": f"Failed: {error_message}"})
        else:
            st.error("Could not generate an answer due to an unknown error.")
            st.session_state.messages.append({"role": "assistant", "content": "Unknown error"})
