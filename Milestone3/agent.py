import os
import google.generativeai as genai
from typing import TypedDict, List, Any, Optional, Literal
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# --- IMPORT CUSTOM TOOLS ---
# These imports rely on the files being in the same directory
from Tools.database import Neo4jConnection
from Tools.cypher_tool import classification_chain, extraction_chain, generate_cypher_query
from Tools.prompt_engineer_tool import optimize_query
from Tools.rag_tool import search_knowledge_base
from Tools.llm_factory import get_llm

# Setup Gemini for the final synthesizer
load_dotenv()

# ---------------------------------------------------------
# 1. DEFINE HYBRID STATE
# ---------------------------------------------------------
class HybridState(TypedDict):
    # Input
    user_query: str
    retrieval_mode: Literal["baseline", "embeddings", "hybrid"]
    selected_model: str 
    
    # Shared NLU (Intent & Entities)
    intent: str
    entities: dict
    
    # Branch A: Cypher (Structured)
    cypher_sql: Optional[str]
    cypher_params: Optional[dict]
    cypher_results: List[dict]
    
    # Branch B: Vector (Unstructured)
    optimized_query: str
    vector_docs: List[str]
    
    # Output
    final_answer: str

# ---------------------------------------------------------
# 2. DEFINE SHARED NODES (NLU)
# ---------------------------------------------------------

def classify_node(state: HybridState):
    print(f"\n--- NODE: INTENT CLASSIFIER ({state['retrieval_mode']}) ---")
    response = classification_chain.invoke({"query": state["user_query"]})
    print(f"Intent: {response.intent}")
    return {"intent": response.intent}

def extract_node(state: HybridState):
    print("\n--- NODE: ENTITY EXTRACTOR ---")
    response = extraction_chain.invoke({"query": state["user_query"]})
    entities = response.model_dump()
    # Filter None values for cleaner logs
    clean_entities = {k: v for k, v in entities.items() if v is not None}
    print(f"Entities: {clean_entities}")
    return {"entities": entities}

# ---------------------------------------------------------
# 3. DEFINE BRANCH A NODES (CYPHER)
# ---------------------------------------------------------

def cypher_gen_node(state: HybridState):
    print("\n--- NODE: CYPHER GENERATOR ---")
    query, params = generate_cypher_query(state["intent"], state["entities"])
    if query:
        print(f"Generated Cypher (Partial): {query.strip()[:50]}...")
    else:
        print("No Cypher template matched.")
    return {"cypher_sql": query, "cypher_params": params}

def cypher_exec_node(state: HybridState):
    print("\n--- NODE: NEO4J EXECUTION ---")
    if not state.get("cypher_sql"):
        return {"cypher_results": []}
    
    db = Neo4jConnection()
    try:
        data = db.query(state["cypher_sql"], state["cypher_params"])
        print(f"Neo4j returned {len(data)} records.")
        return {"cypher_results": data}
    finally:
        db.close()

# ---------------------------------------------------------
# 4. DEFINE BRANCH B NODES (VECTOR RAG)
# ---------------------------------------------------------

def prompt_eng_node(state: HybridState):
    print("\n--- NODE: PROMPT ENGINEER ---")
    optimized = optimize_query(state["user_query"])
    print(f"Optimized for Vector: '{optimized}'")
    return {"optimized_query": optimized}

def rag_search_node(state: HybridState):
    print("\n--- NODE: VECTOR SEARCH ---")
    docs = search_knowledge_base(state["optimized_query"])
    print(f"FAISS retrieved {len(docs)} text chunks.")
    return {"vector_docs": docs}

# ---------------------------------------------------------
# 5. DEFINE SYNTHESIS NODE (MERGE)
# ---------------------------------------------------------

def synthesizer_node(state: HybridState):
    print("\n--- NODE: FINAL SYNTHESIS ---")
    
    # Prepare Structured Data Context
    structured_data = "Skipped (Mode: Embeddings Only)"
    if state.get("cypher_results") is not None:
        if state["cypher_results"]:
            structured_data = str(state["cypher_results"])
        else:
            structured_data = "No structured data found."

    # Prepare Unstructured Data Context
    unstructured_data = "Skipped (Mode: Baseline Only)"
    if state.get("vector_docs") is not None:
        if state["vector_docs"]:
            unstructured_data = "\n\n".join(state["vector_docs"])
        else:
            unstructured_data = "No text context found."

    # Master Prompt
    prompt_text = f"""
    You are an advanced Airline Operations Assistant.
    Answer the user's question based on the active retrieval methods.
    
    MODE: {state['retrieval_mode'].upper()}

    1. STRUCTURED DATABASE (Facts, numbers, flight IDs):
    {structured_data}

    2. UNSTRUCTURED TEXT (Reviews, feedback, descriptions):
    {unstructured_data}

    ### USER QUESTION
    {state['user_query']}

    ### INSTRUCTIONS
    - Use ONLY the provided data.
    - Return answer in a user friendly manner.
    - If a source says "Skipped", do not hallucinate data for it.
    - Prioritize Structured Data for stats/delays, and Text for sentiment.
    """

    try:
        model_name = state.get("selected_model", "Gemini Flash")
        print(f"   [Synthesis] Using Model: {model_name}")
        
        llm = get_llm(model_name)
        
        # Invoke LLM (LangChain interface)
        # Wrap prompt in HumanMessage for Chat Models
        messages = [HumanMessage(content=prompt_text)]
        response = llm.invoke(messages)
        
        # Handle different return types
        if hasattr(response, "content"):
            answer = response.content.strip()
        else:
            answer = str(response).strip()
            
    except Exception as e:
        answer = f"Error during synthesis: {e}"

    return {"final_answer": answer}

# ---------------------------------------------------------
# 6. BUILD THE GRAPH WITH ROUTING
# ---------------------------------------------------------
workflow = StateGraph(HybridState)

# Add Nodes
workflow.add_node("classifier", classify_node)
workflow.add_node("extractor", extract_node)
workflow.add_node("cypher_gen", cypher_gen_node)
workflow.add_node("cypher_exec", cypher_exec_node)
workflow.add_node("prompt_eng", prompt_eng_node)
workflow.add_node("rag_search", rag_search_node)
workflow.add_node("synthesizer", synthesizer_node)

# --- STANDARD FLOW ---
workflow.add_edge(START, "classifier")
workflow.add_edge("classifier", "extractor")

# --- CONDITIONAL ROUTING ---
def route_mode(state: HybridState):
    """Decides which path(s) to take based on user selection."""
    mode = state["retrieval_mode"]
    
    if mode == "baseline":
        return ["cypher_gen"]     # Run Only Cypher
    elif mode == "embeddings":
        return ["prompt_eng"]     # Run Only Embeddings
    else: 
        return ["cypher_gen", "prompt_eng"] # Run Both (Hybrid)

# Configure the split after extraction
workflow.add_conditional_edges(
    "extractor",
    route_mode,
    {
        "cypher_gen": "cypher_gen",
        "prompt_eng": "prompt_eng"
    }
)

# --- PATH CONNECTIONS ---
# Path A: Cypher Execution
workflow.add_edge("cypher_gen", "cypher_exec")
workflow.add_edge("cypher_exec", "synthesizer")

# Path B: Vector Search
workflow.add_edge("prompt_eng", "rag_search")
workflow.add_edge("rag_search", "synthesizer")

# Final step
workflow.add_edge("synthesizer", END)

app = workflow.compile()

# ---------------------------------------------------------
# 7. EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n=== SELECTABLE RETRIEVAL AGENT INITIALIZED ===")
    
    while True:
        print("\n-------------------------------------------")
        print("1. Select Retrieval Mode")
        print("2. Select LLM Model")
        print("3. Run Query")
        print("q. Quit")
        
        # Default settings if not changed
        if 'current_mode' not in locals(): current_mode = "hybrid"
        if 'current_model' not in locals(): current_model = "Gemini Flash"
        
        print(f"\nCurrent Settings: Mode=[{current_mode}], Model=[{current_model}]")
        
        choice = input("Enter choice: ")
        
        if choice == '1':
            print("\nRetrieval Modes:")
            print("1. Baseline (Cypher Only)")
            print("2. Embeddings (Vector Only)")
            print("3. Hybrid (Both)")
            m = input("Choice (1-3): ")
            mode_map = {"1": "baseline", "2": "embeddings", "3": "hybrid"}
            current_mode = mode_map.get(m, "hybrid")
            
        elif choice == '2':
            print("\nLLM Models:")
            print("1. Gemini Flash")
            print("2. Mistral-7B")
            print("3. Zephyr-7B")
            m = input("Choice (1-3): ")
            model_map = {"1": "Gemini Flash", "2": "Mistral-7B", "3": "Zephyr-7B"}
            current_model = model_map.get(m, "Gemini Flash")
            
        elif choice == '3':
            q = input(f"Enter query: ")
            if q.lower() == 'q': break
            
            inputs = {
                "user_query": q,
                "retrieval_mode": current_mode,
                "selected_model": current_model
            }
            
            try:
                result = app.invoke(inputs)
                print("\n>> FINAL ANSWER:")
                print(result["final_answer"])
            except Exception as e:
                print(f"\n[Error] Agent execution failed: {e}")
                
        elif choice.lower() == 'q':
            break
