import os
import faiss
import numpy as np
import pickle
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# 1. Load Credentials from config.txt
# ---------------------------------------------------------
def load_config(file_path="../../config.txt"):
    config = {}
    try:
        with open(file_path, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    config[key.strip().lower()] = value.strip() 
        return config
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None

# ---------------------------------------------------------
# 2. Extract Data from Neo4j
# ---------------------------------------------------------
def fetch_graph_data(uri, username, password):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    query = """
    MATCH (p:Passenger)-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
    MATCH (f)-[:DEPARTS_FROM]->(origin:Airport)
    MATCH (f)-[:ARRIVES_AT]->(dest:Airport)
    RETURN 
        p.record_locator AS passenger_id,
        p.loyalty_program_level AS loyalty,
        p.generation AS gen,
        j.passenger_class AS class,
        j.food_satisfaction_score AS food_score,
        j.arrival_delay_minutes AS delay,
        j.actual_flown_miles AS miles,
        j.number_of_legs AS legs,
        j.passenger_class AS p_class,
        f.flight_number AS flight_num,
        f.fleet_type_description AS fleet,
        origin.station_code AS origin_code,
        dest.station_code AS dest_code
    """
    
    records = []
    try:
        with driver.session() as session:
            result = session.run(query)
            records = [record.data() for record in result]
            print(f"Successfully extracted {len(records)} records from Neo4j.")
    except Exception as e:
        print(f"Failed to connect or query Neo4j: {e}")
    finally:
        driver.close()
        
    return records

# ---------------------------------------------------------
# 3. Main Workflow
# ---------------------------------------------------------
config = load_config()

if config:
    graph_data = fetch_graph_data(config.get("uri"), config.get("username"), config.get("password"))

    if graph_data:
        # B. Text Serialization
        feature_texts = []
        for r in graph_data:
            if r['food_score'] >= 8:
                food_desc = "delicious and excellent"
            elif r['food_score'] <= 3:
                food_desc = "terrible and poor"
            else:
                food_desc = "average"
            
            text_representation = (
                f"Passenger {r['passenger_id']} ({r['gen']}, {r['loyalty']} status) "
                f"booked {r['p_class']} class on Flight {r['flight_num']} "
                f"(operated by {r['fleet']}). "
                f"The journey from {r['origin_code']} to {r['dest_code']} "
                f"covered {r['miles']} miles across {r['legs']} leg(s). "
                f"Feedback: The food was {food_desc} (rated {r['food_score']}/10). "
                f"The flight had an arrival delay of {r['delay']} minutes."
            )
            feature_texts.append(text_representation)

        # C. Generate Embeddings
        print("Loading embedding model...")
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        print("Encoding graph data into vectors...")
        embeddings = embedder.encode(feature_texts, convert_to_numpy=True)
        print(f"Embedding Shape: {embeddings.shape}")

        # D. Build FAISS Index
        d = embeddings.shape[1] 
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        print(f"FAISS index built successfully! Total vectors: {index.ntotal}")

        # E. Save Index and Texts one directory up
        save_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
        faiss_index_path = os.path.join(save_dir, "airline_db.index")
        texts_path = os.path.join(save_dir, "airline_texts.pkl")

        faiss.write_index(index, faiss_index_path)
        print(f"Saved FAISS index to '{faiss_index_path}'")

        with open(texts_path, "wb") as f:
            pickle.dump(feature_texts, f)
        print(f"Saved text chunks to '{texts_path}'")

        # Optional: Test Retrieval immediately
        print("\n--- Sample Serialized Text ---")
        print(feature_texts[0])

        test_query = "High delay and low food satisfaction"
        query_emb = embedder.encode([test_query], convert_to_numpy=True)
        distances, indices = index.search(query_emb, k=1)
        
        print("\n--- Test Retrieval ---")
        print(f"Query: {test_query}")
        best_match_idx = indices[0][0]
        print(f"Best Match:\n{feature_texts[best_match_idx]}")
