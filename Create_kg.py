import pandas as pd
from neo4j import GraphDatabase
import sys

# --- Configuration ---
CONFIG_FILE = 'config.txt'
CSV_FILE = 'Airline_surveys_sample.csv'

def read_config(file_path):
    """Reads the config.txt file to get database credentials."""
    config = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[key] = value
        return config
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        sys.exit(1)

def create_constraints(driver):
    """Creates uniqueness constraints and indexes based on the schema."""
    queries = [
        "CREATE CONSTRAINT passenger_id IF NOT EXISTS FOR (p:Passenger) REQUIRE p.record_locator IS UNIQUE",
        "CREATE CONSTRAINT journey_id IF NOT EXISTS FOR (j:Journey) REQUIRE j.feedback_ID IS UNIQUE",
        "CREATE CONSTRAINT airport_id IF NOT EXISTS FOR (a:Airport) REQUIRE a.station_code IS UNIQUE",
        "CREATE INDEX flight_composite_index IF NOT EXISTS FOR (f:Flight) ON (f.flight_number, f.fleet_type_description)"
    ]
    with driver.session() as session:
        for q in queries:
            session.run(q)
        print("Constraints and indexes verified.")

def print_statistics(driver):
    """Prints statistics about the created Knowledge Graph."""
    print("\nKnowledge Graph Statistics:")
    
    queries = {
        "Passengers": "MATCH (p:Passenger) RETURN count(p) as count",
        "Journeys": "MATCH (j:Journey) RETURN count(j) as count",
        "Flights": "MATCH (f:Flight) RETURN count(f) as count",
        "Airports": "MATCH (a:Airport) RETURN count(a) as count",
        "TOOK relationships": "MATCH ()-[r:TOOK]->() RETURN count(r) as count",
        "ON relationships": "MATCH ()-[r:ON]->() RETURN count(r) as count",
        "DEPARTS_FROM relationships": "MATCH ()-[r:DEPARTS_FROM]->() RETURN count(r) as count",
        "ARRIVES_AT relationships": "MATCH ()-[r:ARRIVES_AT]->() RETURN count(r) as count"
    }
    
    with driver.session() as session:
        for label, query in queries.items():
            result = session.run(query)
            count = result.single()['count']
            print(f"  - {label}: {count}")

def run_interactive_queries(driver):
    """Allows the user to input Cypher queries and see the results."""
    print("\n" + "="*50)
    print("INTERACTIVE QUERY MODE")
    print("Enter a Cypher query to run against the database.")
    print("Type 'exit' or simply press Enter to quit.")
    print("="*50)

    while True:
        user_input = input("\nEnter Cypher Query> ").strip()
        
        if not user_input or user_input.lower() == 'exit':
            print("Exiting interactive mode.")
            break
            
        try:
            with driver.session() as session:
                result = session.run(user_input)
                records = list(result)
                
                if not records:
                    print("No records returned.")
                else:
                    print(f"Found {len(records)} records:")
                    for i, record in enumerate(records, 1):
                        # record.data() returns a clean dictionary of the result
                        print(f"[{i}] {record.data()}")
                        
        except Exception as e:
            print(f"Query Error: {e}")

def load_data(driver, csv_path):
    """Reads CSV and ingests data into Neo4j using batch processing."""
    print(f"Reading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Make sure it is in the same folder.")
        sys.exit(1)

    # --- Data Pre-processing ---
    # Clean column names (remove accidental spaces)
    df.columns = df.columns.str.strip()
    
    # Verify 'passenger_class' exists
    if 'passenger_class' not in df.columns:
        # Fallback check if it's named 'class' like in older versions or Pandas defaults
        if 'class' in df.columns:
             df.rename(columns={'class': 'passenger_class'}, inplace=True)
        else:
             print("Error: Could not find 'passenger_class' column in CSV.")
             print(f"Columns found: {list(df.columns)}")
             sys.exit(1)

    # Handle potential missing values (NaN)
    df['food_satisfaction_score'] = pd.to_numeric(df['food_satisfaction_score'], errors='coerce').fillna(0).astype(int)
    df['arrival_delay_minutes'] = pd.to_numeric(df['arrival_delay_minutes'], errors='coerce').fillna(0).astype(int)
    df['actual_flown_miles'] = pd.to_numeric(df['actual_flown_miles'], errors='coerce').fillna(0).astype(int)
    df['number_of_legs'] = pd.to_numeric(df['number_of_legs'], errors='coerce').fillna(0).astype(int)
    
    # Ensure strings
    df['passenger_class'] = df['passenger_class'].fillna("Unknown").astype(str)
    df['feedback_ID'] = df['feedback_ID'].astype(str)
    
    # --- Ingestion Query ---
    cypher_query = """
    UNWIND $rows AS row
    
    MERGE (p:Passenger {record_locator: row.record_locator})
    SET p.loyalty_program_level = row.loyalty_program_level,
        p.generation = row.generation

    MERGE (a_origin:Airport {station_code: row.origin_station_code})
    MERGE (a_dest:Airport {station_code: row.destination_station_code})

    MERGE (f:Flight {
        flight_number: row.flight_number, 
        fleet_type_description: row.fleet_type_description
    })

    MERGE (j:Journey {feedback_ID: row.feedback_ID})
    SET j.food_satisfaction_score = row.food_satisfaction_score,
        j.arrival_delay_minutes = row.arrival_delay_minutes,
        j.actual_flown_miles = row.actual_flown_miles,
        j.number_of_legs = row.number_of_legs,
        j.passenger_class = row.passenger_class

    MERGE (p)-[:TOOK]->(j)
    MERGE (j)-[:ON]->(f)
    MERGE (f)-[:DEPARTS_FROM]->(a_origin)
    MERGE (f)-[:ARRIVES_AT]->(a_dest)
    """

    batch_size = 1000
    total_rows = len(df)
    print(f"Starting ingestion of {total_rows} rows...")

    with driver.session() as session:
        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:i+batch_size].to_dict('records')
            try:
                session.run(cypher_query, rows=batch)
                print(f"Processed rows {i} to {min(i+batch_size, total_rows)}")
            except Exception as e:
                print(f"Error processing batch starting at {i}: {e}")

    print("Data ingestion complete.")

def main():
    config = read_config(CONFIG_FILE)
    uri = config.get('URI')
    username = config.get('USERNAME')
    password = config.get('PASSWORD')

    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        print("Connected to Neo4j.")
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        sys.exit(1)

    try:
        create_constraints(driver)
        load_data(driver, CSV_FILE)
        # Call the statistics function here
        print_statistics(driver)
        # Start the interactive query loop
        run_interactive_queries(driver)

    finally:
        driver.close()

if __name__ == "__main__":
    main()