import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq

# ==========================================
# 1. SETUP & PATH RESOLUTION
# ==========================================
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[0]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agent_tools import query_telemetry_db, fetch_corridor_conditions, search_compliance_sop

load_dotenv(project_root / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("CRITICAL: GROQ_API_KEY is missing from your .env file.")

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ==========================================
# 2. LOAD SYSTEM PROMPT
# ==========================================
prompt_path = project_root / "src" / "prompts" / "system_prompt.txt"
try:
    with open(prompt_path, "r", encoding="utf-8") as f:
        SYSTEM_INSTRUCTIONS = f.read()
except FileNotFoundError:
    SYSTEM_INSTRUCTIONS = "You are a Senior Logistics AI Assistant analyzing cold-chain fleet telemetry, weather, and SOP compliance."

# ==========================================
# 3. INITIALIZE GROQ REASONER & TOOLS
# ==========================================
print("🤖 Initializing Groq Reasoner (openai/gpt-oss-120b)...", flush=True)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.1,
    timeout=60,
    max_retries=3
)

fde_tools = [query_telemetry_db, fetch_corridor_conditions, search_compliance_sop]
llm_with_tools = llm.bind_tools(fde_tools)
# ==========================================
# 4. GRAPH NODES & PIPELINE
# ==========================================
def reasoning_node(state: AgentState):
    messages = list(state["messages"])
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_INSTRUCTIONS)] + messages

    print("   [Agent] 🧠 Reasoning with Groq...", flush=True)
    response = llm_with_tools.invoke(messages)

    if getattr(response, "tool_calls", None):
        for tool_call in response.tool_calls:
            print(f"   [Tool Request] 🛠️ {tool_call['name']}({tool_call['args']})", flush=True)
    else:
        print("   [Agent] ✍️ Formulating final response...", flush=True)

    return {"messages": [response]}

print("⚙️ Compiling LangGraph State Machine...", flush=True)
graph_builder = StateGraph(AgentState)
graph_builder.add_node("reasoner", reasoning_node)
graph_builder.add_node("tools", ToolNode(fde_tools))

graph_builder.add_edge(START, "reasoner")
graph_builder.add_conditional_edges("reasoner", tools_condition)
graph_builder.add_edge("tools", "reasoner")

fde_agent = graph_builder.compile(checkpointer=MemorySaver())

def format_clean_text(content) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        extracted = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                extracted.append(item["text"])
            elif isinstance(item, str):
                extracted.append(item)
        return "".join(extracted)
    return str(content)

# ==========================================
# 5. UI EXPORT HELPER
# ==========================================
def run_agent(query: str, thread_id: str = "production_ui_thread") -> str:
    thread_config = {"configurable": {"thread_id": thread_id}}
    final_output = ""
    
    events = fde_agent.stream(
        {"messages": [HumanMessage(content=query)]},
        config=thread_config,
        stream_mode="updates"
    )
    
    for event in events:
        for node_name, node_state in event.items():
            if node_name == "reasoner":
                msg = node_state["messages"][-1]
                if not getattr(msg, "tool_calls", None):
                    text = format_clean_text(msg.content).strip()
                    if text:
                        final_output = text
                        
    return final_output or "System active. Ready for dispatch instructions."

# ==========================================
# 6. CLI INTERACTIVE LOOP
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("🚀 FDE Supply Chain Orchestrator Online (Groq)")
    print("=" * 55 + "\n")

    thread_config = {"configurable": {"thread_id": "cli_session_1"}}

    while True:
        try:
            user_input = input("\nDispatcher > ")
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input.strip():
            continue
        if user_input.lower() in ["exit", "quit"]:
            break

        events = fde_agent.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=thread_config,
            stream_mode="updates"
        )
        for event in events:
            for node_name, node_state in event.items():
                if node_name == "tools":
                    for tool_msg in node_state.get("messages", []):
                        content_preview = str(tool_msg.content)[:120].replace("\n", " ")
                        print(f"   [Tool Result] 📦 {tool_msg.name}: {content_preview}...", flush=True)
                elif node_name == "reasoner":
                    msg = node_state["messages"][-1]
                    if msg.content and not getattr(msg, "tool_calls", None):
                        clean_output = format_clean_text(msg.content)
                        if clean_output.strip():
                            print(f"\n🤖 FDE Agent:\n{clean_output}\n", flush=True)