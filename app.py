import streamlit as st
import uuid
from typing import Dict, Any, List
import re
import os 

# --- Project imports ---
from src.graph.factory import get_graph_app
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

# --- Helper Function for Progress Tracking ---
def parse_progress(messages: List[Dict[str, Any]]) -> str:
    """Build real-time progress text from the message list."""
    progress_text = "### 🏃‍♂️ Progress\n"
    team1_status, team2_status, team3_status = "⏳ Analyzing question...\n", "⏳ Waiting...\n", "⏳ Waiting...\n"
    rag_query = ""
    team1_failed = False

    for msg in messages:
        # Analyze Team Query status.
        if msg.name == "team1_evaluator":
            if msg.content == "pass":
                team1_status = "✅ Done\n"
                rag_query = msg.additional_kwargs.get("best_rag_query", "")
            else:
                team1_status = f"❌ Failed ({msg.content})\n"
                team1_failed = True
    
    progress_text += f"**Team Query (Question Analysis)**: {team1_status}\n"
    if rag_query:
        progress_text += f"   - Best search query: `{rag_query}`\n\n"

    if team1_status == "✅ Done\n":
        team2_started = any(m.name in ["rag_search_result", "web_search_result"] for m in messages)
        team2_evaluated = any(m.name == "team2_evaluator" for m in messages)

        if not team2_started:
             team2_status = "⏳ Collecting data...\n"
        
        if team2_evaluated:
            team2_eval_msg = next((m for m in reversed(messages) if m.name == "team2_evaluator"), None)
            if team2_eval_msg and team2_eval_msg.content == "pass":
                team2_status = "✅ Done\n"
            else:
                team2_status = f"❌ Failed ({team2_eval_msg.content if team2_eval_msg else 'N/A'})\n"
        
        progress_text += f"**Team Search (Information Retrieval)**: {team2_status}\n"

        if team2_status == "✅ Done\n":
            team3_evaluated = any(m.name == "final_evaluator" for m in messages)
            if not team3_evaluated:
                team3_status = "⏳ Generating answer...\n"
            else:
                team3_eval_msg = next((m for m in reversed(messages) if m.name == "final_evaluator"), None)
                if team3_eval_msg and team3_eval_msg.content == "pass":
                    team3_status = "✅ Done\n"
                else:
                    team3_status = f"❌ Failed ({team3_eval_msg.content if team3_eval_msg else 'N/A'})\n"
            
            progress_text += f"**Team Answer (Answer Generation)**: {team3_status}\n"

    elif team1_failed:
        progress_text += "**Team Search (Information Retrieval)**: 🛑 Stopped\n\n"
        progress_text += "**Team Answer (Answer Generation)**: 🛑 Stopped\n\n"


    return progress_text

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
                "is_simple_query": "No"
            }
            thread = {"configurable": {"thread_id": st.session_state.thread_id}}

            # Process real-time events from the LangGraph stream.
            final_state_messages = []
            for event in app.stream(initial_state, thread, stream_mode="values"):
                # Read messages from the current event.
                messages = event.get("messages", [])
                final_state_messages = messages
                
                # Update progress from messages.
                progress_text = parse_progress(messages)
                progress_placeholder.markdown(progress_text)
                
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
                    if last_msg.content != "pass":
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
