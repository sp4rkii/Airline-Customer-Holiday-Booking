import pandas as pd
from neo4j import GraphDatabase
import sys

# =========================
# CONFIGURATION
# =========================
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


# =========================
# CONSTRAINTS
# =========================
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


# =========================
# STATISTICS
# =========================
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
            print(f"  - {label}: {result.single()['count']}")


# =========================
# VERIFICATION QUERIES 1–5
# =========================
def run_verification_queries(driver):
    print("\n==============================")
    print("RUNNING VERIFICATION QUERIES")
    print("==============================\n")

    # ---------- QUERY 1 ----------
    q1 = """
    MATCH (f:Flight)-[:DEPARTS_FROM]->(o:Airport),
          (f)-[:ARRIVES_AT]->(d:Airport)
    RETURN o.station_code AS origin,
           d.station_code AS destination,
           count(f) AS flight_count
    ORDER BY flight_count DESC
    LIMIT 5;
    """

    # ---------- QUERY 2 ----------
    q2 = """
    MATCH (j:Journey)-[:ON]->(f:Flight)
    WITH f.flight_number AS flight_number,
        count(j) AS passenger_feedback_count
    RETURN flight_number, passenger_feedback_count
    ORDER BY passenger_feedback_count DESC
    LIMIT 10;
    """

    # ---------- QUERY 3 ----------
    q3 = """
    MATCH (p:Passenger)-[:TOOK]->(j:Journey)
    WHERE j.number_of_legs > 1
    WITH p.generation AS generation,
         count(j) AS multi_leg_count,
         avg(j.food_satisfaction_score) AS avg_score
    RETURN generation, multi_leg_count, avg_score
    ORDER BY multi_leg_count DESC;
    """

    # ---------- QUERY 4 ----------
    q4 = """
    MATCH (j:Journey)-[:ON]->(f:Flight)
    WITH f.flight_number AS flight_id,
         avg(j.arrival_delay_minutes) AS avg_arrival_delay
    RETURN flight_id, avg_arrival_delay
    ORDER BY avg_arrival_delay ASC
    LIMIT 10;
    """

    # ---------- QUERY 5 ----------
    q5 = """
    MATCH (p:Passenger)-[:TOOK]->(j:Journey)
    WITH p.loyalty_program_level AS loyalty_level,
         avg(j.actual_flown_miles) AS avg_actual_flown_miles
    RETURN loyalty_level, avg_actual_flown_miles
    ORDER BY avg_actual_flown_miles DESC;
    """

    with driver.session() as session:
        # Query 1
        print("Query 1 — Top 5 Routes by Number of Flights:\n")
        for r in session.run(q1):
            print(f"  {r['origin']} → {r['destination']} | flights: {r['flight_count']}")

        # Query 2
        print("\nQuery 2 — Top 10 Flights by Passenger Feedback:\n")
        for r in session.run(q2):
            print(f"  Flight {r['flight_number']} | feedbacks: {r['passenger_feedback_count']}")

        # Query 3
        print("\nQuery 3 — Avg Food Satisfaction for Multi-Leg Journeys:\n")
        for r in session.run(q3):
            print(f"  {r['generation']} | journeys: {r['multi_leg_count']} | avg score: {r['avg_score']}")

        # Query 4
        print("\nQuery 4 — Flights with Shortest Arrival Delay:\n")
        for r in session.run(q4):
            print(f"  Flight {r['flight_id']} | avg delay: {r['avg_arrival_delay']} minutes")

        # Query 5
        print("\nQuery 5 — Avg Flown Miles by Loyalty Level:\n")
        for r in session.run(q5):
            print(f"  {r['loyalty_level']} | avg miles: {r['avg_actual_flown_miles']}")

    print("\nVerification queries complete.\n")


# =========================
# INTERACTIVE QUERY MODE
# =========================
def run_interactive_queries(driver):
    print("\n" + "=" * 50)
    print("INTERACTIVE QUERY MODE")
    print("Enter a Cypher query to run against the database.")
    print("Type 'exit' or press Enter to quit.")
    print("=" * 50)

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
                        print(f"[{i}] {record.data()}")

        except Exception as e:
            print(f"Query Error: {e}")


# =========================
# LOAD CSV INTO NEO4J
# =========================
def load_data(driver, csv_path):
    print(f"Reading {csv_path}...")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    df.columns = df.columns.str.strip()

    if 'passenger_class' not in df.columns:
        if 'class' in df.columns:
            df.rename(columns={'class': 'passenger_class'}, inplace=True)
        else:
            print("Error: Could not find passenger_class column.")
            print("Columns:", list(df.columns))
            sys.exit(1)

    # Normalize numeric columns
    df['food_satisfaction_score'] = pd.to_numeric(df['food_satisfaction_score'], errors='coerce').fillna(0).astype(int)
    df['arrival_delay_minutes'] = pd.to_numeric(df['arrival_delay_minutes'], errors='coerce').fillna(0).astype(int)
    df['actual_flown_miles'] = pd.to_numeric(df['actual_flown_miles'], errors='coerce').fillna(0).astype(int)
    df['number_of_legs'] = pd.to_numeric(df['number_of_legs'], errors='coerce').fillna(0).astype(int)

    df['passenger_class'] = df['passenger_class'].fillna("Unknown")
    df['feedback_ID'] = df['feedback_ID'].astype(str)

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
    print(f"Starting ingestion of {total_rows} records...")

    with driver.session() as session:
        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:i+batch_size].to_dict('records')
            try:
                session.run(cypher_query, rows=batch)
                print(f"Processed rows {i} → {min(i+batch_size, total_rows)}")
            except Exception as e:
                print(f"Error in batch starting at row {i}: {e}")

    print("Data ingestion complete.")


# =========================
# MAIN
# =========================
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
        print_statistics(driver)
        run_verification_queries(driver)
        run_interactive_queries(driver)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
