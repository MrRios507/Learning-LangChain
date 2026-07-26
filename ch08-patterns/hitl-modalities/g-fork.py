import asyncio
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 1. Define the State
class ADRState(TypedDict):
    problem: str
    context_analysis: str
    human_instruction: str
    final_adr: str

# 2. Node 1: The system analyzes the problem (costly operation in real life)
async def analyze_node(state: ADRState) -> ADRState:
    print("🤖 Analyzer: Processing system requirements...")
    await asyncio.sleep(2)  # Simulating the LLM call

    analysis = "The system requires high availability and complex transactional data handling."
    return {"context_analysis": analysis}

# 3. Node 2: The system drafts the document based on the human's decision
async def draft_node(state: ADRState) -> ADRState:
    strategy = state.get("human_instruction", "Default strategy")
    print(f"✍️  Drafter: Writing ADR using the approach: {strategy}")
    await asyncio.sleep(2)

    document = f"# ADR\nContext: {state.get('context_analysis')}\nDecision: {strategy}"
    return {"final_adr": document}

# 4. Compile the graph
def create_adr_graph():
    builder = StateGraph(ADRState)
    builder.add_node("analyze", analyze_node)
    builder.add_node("draft", draft_node)

    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "draft")
    builder.add_edge("draft", END)

    memory = MemorySaver()

    return builder.compile(checkpointer=memory, interrupt_before=["draft"])

async def main():
    graph = create_adr_graph()

    # Configure the original thread
    original_config = {"configurable": {"thread_id": "adr_main_thread"}}
    initial_state = {
        "problem": "We need a system to process massive payments.",
        "human_instruction": ""
    }

    print("\n=== PHASE 1: INITIAL EXECUTION ===")
    # Execute up to the interruption point
    async for event in graph.astream(initial_state, original_config):
        pass

    # Get the state at the pause
    paused_state = await graph.aget_state(original_config)
    print("Context extracted by AI:", paused_state.values.get('context_analysis'))
    print(f"Graph paused. Next node to execute: {paused_state.next}")

    print("\n=== PHASE 2: INTERVENTION IN ORIGINAL THREAD ===")
    print("👤 Human decides to explore Path A: Microservices and DDD")

    await graph.aupdate_state(
        original_config, 
        {"human_instruction": "Use Microservices with Anti-Corruption Layers (ACL)"}, 
        as_node="analyze"   # Simulate the update came from the previous node
    )

    # Resume the original thread
    await graph.ainvoke(None, original_config)
    result_a = await graph.aget_state(original_config)

    print("\n=== PHASE 3: CREATING THE FORK (PARALLEL UNIVERSE) ===")
    print("👤 Human wants to see a Modular Monolith approach without losing previous analysis.")

    # 1. Create a new Thread ID
    fork_config = {"configurable": {"thread_id": "adr_forked_thread"}}
    
    # 2. Inject the values from the paused state into this new thread
    await graph.aupdate_state(fork_config, paused_state.values)
    
    # 3. The human injects their SECOND decision into this new thread
    await graph.aupdate_state(
        fork_config, 
        {"human_instruction": "Use a Modular Monolith with in-memory caching"}, 
        as_node="analyze"
    )
    
    # 4. Resume the forked thread
    await graph.ainvoke(None, fork_config)
    result_b = await graph.aget_state(fork_config)


    print("\n=== PHASE 4: COMPARING RESULTS ===")
    print("--- Path A (Original Thread) ---")
    print(result_a.values.get('final_adr'))
    
    print("\n--- Path B (Forked Thread) ---")
    print(result_b.values.get('final_adr'))

if __name__ == "__main__":
    asyncio.run(main())
