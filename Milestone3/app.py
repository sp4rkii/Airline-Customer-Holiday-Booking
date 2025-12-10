import streamlit as st
import time
from agent import app as graph_app  # Import your LangGraph application

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Airline Graph-RAG Assistant",
    page_icon="✈️",
    layout="wide"
)

# --- CSS FOR STYLING ---
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/airplane-take-off.png", width=60)
    st.title("Control Panel")
    
    st.markdown("### ⚙️ Configuration")
    
    # 1. Retrieval Method Selection (Page 6 Req)
    retrieval_mode = st.radio(
        "Retrieval Method:",
        ("hybrid", "baseline", "embeddings"),
        format_func=lambda x: x.capitalize(),
        help="Baseline = Cypher Only\nEmbeddings = Vector Search\nHybrid = Both"
    )
    
    # 2. Model Selection (Page 6 Req)
    selected_model = st.selectbox(
        "Select LLM Model:",
        ("Gemini Flash", "Mistral-7B", "Zephyr-7B"),
        index=0
    )
    
    st.divider()
    st.info(f"**Mode:** {retrieval_mode.capitalize()}\n**Model:** {selected_model}")
    
    if st.button("Clear Chat History", type="primary"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("✈️ Airline Operations Assistant")
st.markdown("*Ask about flights, delays, passenger satisfaction, or route analysis.*")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If there's debug info (context), display it in an expander
        if "details" in msg:
            with st.expander("🔍 View Reasoning & Context"):
                st.json(msg["details"])

# Handle User Input
if user_input := st.chat_input("Ex: 'Find flights from IAX with high delays'"):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Process with Agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_container = st.status("🧠 Processing...", expanded=True)
        
        try:
            # Inputs for the Graph
            inputs = {
                "user_query": user_input,
                "retrieval_mode": retrieval_mode,
                "selected_model": selected_model
            }
            
            # --- EXECUTE AGENT ---
            # using invoke to run the graph synchronously
            final_state = graph_app.invoke(inputs)
            
            # Extract Results
            final_answer = final_state.get("final_answer", "No answer generated.")
            intent = final_state.get("intent", "Unknown")
            entities = final_state.get("entities", {})
            cypher_query = final_state.get("cypher_sql", "")
            cypher_results = final_state.get("cypher_results", [])
            vector_docs = final_state.get("vector_docs", [])
            
            # --- UPDATE STATUS STEPS ---
            status_container.write(f"✅ **Intent:** `{intent}`")
            
            if retrieval_mode in ["baseline", "hybrid"]:
                if cypher_query:
                    status_container.write("✅ **Cypher:** Generated & Executed")
                else:
                    status_container.write("⚠️ **Cypher:** No template matched")
            
            if retrieval_mode in ["embeddings", "hybrid"]:
                count = len(vector_docs) if vector_docs else 0
                status_container.write(f"✅ **Vector Search:** Found {count} documents")
                
            status_container.update(label="Response Ready!", state="complete", expanded=False)
            
            # --- DISPLAY FINAL ANSWER ---
            message_placeholder.markdown(final_answer)
            
            # --- PREPARE DEBUG DETAILS FOR TRANSPARENCY (Page 5 Req) ---
            debug_info = {
                "1_Intent": intent,
                "2_Entities": {k: v for k, v in entities.items() if v is not None},
            }
            
            if retrieval_mode in ["baseline", "hybrid"]:
                debug_info["3_Cypher_Query"] = cypher_query
                debug_info["4_Graph_Context_Records"] = cypher_results
                
            if retrieval_mode in ["embeddings", "hybrid"]:
                debug_info["5_Vector_Context_Chunks"] = vector_docs

            # Show context immediately for this response
            with st.expander("🔍 View Retrieved Context (Transparency Layer)"):
                
                tab1, tab2, tab3 = st.tabs(["Structured (Graph)", "Unstructured (Vector)", "Prompt Logic"])
                
                with tab1:
                    if cypher_query:
                        st.code(cypher_query, language="cypher")
                        if cypher_results:
                            st.dataframe(cypher_results)
                        else:
                            st.warning("Query returned no data.")
                    else:
                        st.info("No Cypher query ran for this request.")
                        
                with tab2:
                    if vector_docs:
                        for i, doc in enumerate(vector_docs):
                            st.text_area(f"Chunk {i+1}", doc, height=100)
                    else:
                        st.info("No vector embeddings used.")
                        
                with tab3:
                    st.json(debug_info)

            # Save to History
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_answer,
                "details": debug_info
            })
            
        except Exception as e:
            status_container.update(label="Error Occurred", state="error")
            st.error(f"An error occurred: {str(e)}")