import faiss
import pickle
import numpy as np
import os
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

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

# 1. Neo4j Configuration
config = load_config()
URI = config.get("uri")
USER = config.get("username")
PASSWORD = config.get("password")

def fetch_graph_data():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    query = """
    MATCH (p:Passenger)-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
    MATCH (f)-[:DEPARTS_FROM]->(origin:Airport)
    MATCH (f)-[:ARRIVES_AT]->(dest:Airport)
    RETURN 
        p.record_locator AS passenger_id,
        p.loyalty_program_level AS loyalty,
        p.generation AS gen,
        j.passenger_class AS p_class,
        j.food_satisfaction_score AS food_score,
        j.arrival_delay_minutes AS delay,
        j.actual_flown_miles AS miles,
        j.number_of_legs AS legs,
        f.flight_number AS flight_num,
        f.fleet_type_description AS fleet,
        origin.station_code AS origin_code,
        dest.station_code AS dest_code
    """
    try:
        with driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]
    finally:
        driver.close()

def serialize_data(data):
    texts = []
    for r in data:
        # Semantic Logic
        score = r['food_score']
        if score >= 8: food_desc = "delicious and excellent"
        elif score <= 3: food_desc = "terrible and poor"
        else: food_desc = "average"

        delay = r['delay']
        if delay > 30: delay_desc = f"significantly delayed by {delay} minutes"
        elif delay > 0: delay_desc = f"slightly delayed by {delay} minutes"
        else: delay_desc = "on time"

        text = (
            f"Passenger {r['passenger_id']} ({r['gen']}, {r['loyalty']} status) "
            f"booked {r['p_class']} class on Flight {r['flight_num']} "
            f"(operated by {r['fleet']}). "
            f"The journey from {r['origin_code']} to {r['dest_code']} "
            f"covered {r['miles']} miles. "
            f"Feedback: The food was {food_desc} (rated {score}/10). "
            f"The flight was {delay_desc}."
        )
        texts.append(text)
    return texts

def build_index(model_name, filename, texts):
    print(f"\n--- Processing Model: {model_name} ---")
    embedder = SentenceTransformer(model_name)
    print("Encoding...")
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    
    # FAISS Dimension depends on the model (384 vs 768)
    d = embeddings.shape[1]
    print(f"Vector Dimensions: {d}")
    
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    
    faiss.write_index(index, filename)
    print(f"Saved index to {filename}")

if __name__ == "__main__":
    # 1. Fetch Data
    print("Fetching data from Neo4j...")
    data = fetch_graph_data()
    print(f"Fetched {len(data)} records.")
    
    # 2. Serialize
    texts = serialize_data(data)
    
    # 3. Save Text Chunks (Shared by both models)
    with open("airline_texts.pkl", "wb") as f:
        pickle.dump(texts, f)
    
    # 4. Build Index A (MiniLM - Baseline)
    build_index("sentence-transformers/all-MiniLM-L6-v2", "airline_db_mini.index", texts)
    
    # 5. Build Index B (MPNet - High Performance)
    build_index("sentence-transformers/all-mpnet-base-v2", "airline_db_mpnet.index", texts)
    
    print("\nDONE! Ready for comparison.")

    
# Load Texts
with open("airline_texts.pkl", "rb") as f:
    texts = pickle.load(f)

# Load Model A
print("Loading Model A (MiniLM)...")
model_a = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index_a = faiss.read_index("airline_db_mini.index")

# Load Model B
print("Loading Model B (MPNet)...")
model_b = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
index_b = faiss.read_index("airline_db_mpnet.index")

def get_results(query, model, index, k=1):
    vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(vec, k)
    results = []
    for idx in indices[0]:
        if idx < len(texts):
            results.append(texts[idx])
    return results

if __name__ == "__main__":
    print("\n=== EMBEDDING MODEL SHOWDOWN ===")
    print("Model A: all-MiniLM-L6-v2 (Fast, 384 dim)")
    print("Model B: all-mpnet-base-v2 (Precise, 768 dim)")
    
    while True:
        q = input("\nEnter test query (or 'q'): ")
        if q.lower() == 'q': break
        
        # Get results
        res_a = get_results(q, model_a, index_a, k=1)
        res_b = get_results(q, model_b, index_b, k=1)
        
        print(f"\nQUERY: '{q}'")
        print("-" * 60)
        print(f"MODEL A (MiniLM) FOUND:\n{res_a[0]}")
        print("-" * 60)
        print(f"MODEL B (MPNet) FOUND: \n{res_b[0]}")
        print("=" * 60)