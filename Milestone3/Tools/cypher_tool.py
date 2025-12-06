import os
from dotenv import load_dotenv
from typing import Optional, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from .database import Neo4jConnection

# Load environment variables from .env file immediately
load_dotenv()

# 1. SETUP: Configure your Gemini API Key
if "GOOGLE_API_KEY" not in os.environ:
    print("Warning: GOOGLE_API_KEY not found in environment. Please ensure it is set in your .env file.")

# Initialize the Gemini Model (Flash is faster/cheaper for this task)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0  # Keep it 0 for deterministic results
)

# --- PART A: INTENT CLASSIFICATION ---

class IntentClassification(BaseModel):
    """Classify the user's operational airline query."""
    intent: Literal[
        "flight_search", 
        "analyze_delays", 
        "satisfaction_analysis", 
        "passenger_profiling"
    ] = Field(..., description="The specific operational goal of the user query.")

classifier_llm = llm.with_structured_output(IntentClassification)

classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an operational assistant for an Airline Company.
    Classify the user query into strictly one of these intents:

    1. flight_search: Find flights, routes, or specific flight details.
    2. analyze_delays: Analyze delays for flights, routes, or airports.
    3. satisfaction_analysis: Analyze food satisfaction or customer complaints.
    4. passenger_profiling: Analyze passenger data, loyalty levels, or history.
    """),
    ("human", "{query}"),
])

classification_chain = classifier_prompt | classifier_llm

# --- PART B: ENTITY EXTRACTION ---

class AirlineEntities(BaseModel):
    """Extract relevant entities for the Knowledge Graph."""
    origin: Optional[str] = Field(None, description="Origin Airport Code (e.g., ORD).")
    destination: Optional[str] = Field(None, description="Destination Airport Code (e.g., LAX).")
    station_code: Optional[str] = Field(None, description="Airport Code for station-specific queries.")
    flight_number: Optional[str] = Field(None, description="Flight ID (e.g., AA123).")
    fleet_desc: Optional[str] = Field(None, description="Aircraft model (e.g., Boeing 737). only use the aircraft model number")
    record_locator: Optional[str] = Field(None, description="Passenger Record Locator (PNR).")
    feedback_id: Optional[str] = Field(None, description="Specific Feedback ID.")
    level: Optional[str] = Field(None, description="Loyalty Level (Gold, Silver, etc.).")
    p_class: Optional[str] = Field(None, description="Passenger Class (Business, Economy, etc.).")
    gen: Optional[str] = Field(None, description="Generation (Gen Z, Millennials, etc.).")
    min_delay: Optional[int] = Field(None, description="Minimum delay in minutes.")
    max_score: Optional[float] = Field(None, description="Maximum food score.")
    min_miles: Optional[int] = Field(None, description="Minimum flown miles.")
    max_miles: Optional[int] = Field(None, description="Maximum flown miles.")
    min_legs: Optional[int] = Field(None, description="Minimum number of legs.")

extractor_llm = llm.with_structured_output(AirlineEntities)

extractor_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Extract entities from the airline query to map to the database schema.
    - Convert city names to airport codes (e.g., "Chicago" -> "ORD").
    - Extract numerical values for min/max filters.
    - If an entity is missing, return null.
    """),
    ("human", "{query}"),
])

extraction_chain = extractor_prompt | extractor_llm

# --- MAIN PIPELINE ---

# Initialize DB Connection
db = Neo4jConnection()

def generate_cypher_query(intent, entities):
    """
    Returns a tuple: (Cypher Query String, Parameters Dictionary)
    Matches the schema: Passenger, Journey, Flight, Airport
    """
    
    # --- DATA TYPE CORRECTION ---
    # Flight numbers in Neo4j are Integers, but extraction might return Strings.
    if entities.get('flight_number') and isinstance(entities['flight_number'], str) and entities['flight_number'].isdigit():
        entities['flight_number'] = int(entities['flight_number'])

    # --- GROUP 1: FLIGHT LOOKUP & ROUTES (3 Queries) ---
    
    # Q1: Find Flights by Origin AND Destination
    if intent == "flight_search" and entities.get('origin') and entities.get('destination'):
        query = """
        MATCH (f:Flight)-[:DEPARTS_FROM]->(o:Airport {station_code: $origin})
        MATCH (f)-[:ARRIVES_AT]->(d:Airport {station_code: $destination})
        RETURN f.flight_number, f.fleet_type_description, o.station_code as Origin, d.station_code as Dest
        LIMIT 10;
        """
        return query, entities

    # Q2: Find Flights Arriving at a Specific Airport (Your current working query)
    if intent == "flight_search" and entities.get('destination'):
        query = """
        MATCH (f:Flight)-[:ARRIVES_AT]->(a:Airport {station_code: $destination})
        RETURN f.flight_number, f.fleet_type_description, a.station_code as Destination
        LIMIT 15;
        """
        return query, entities

    # Q3: Lookup Specific Flight Details (by Flight Number)
    if intent == "flight_search" and entities.get('flight_number'):
        query = """
        MATCH (f:Flight {flight_number: $flight_number})
        MATCH (f)-[:DEPARTS_FROM]->(o:Airport)
        MATCH (f)-[:ARRIVES_AT]->(d:Airport)
        RETURN f.flight_number, f.fleet_type_description, o.station_code as Origin, d.station_code as Dest;
        """
        return query, entities


    # --- GROUP 2: DELAY ANALYSIS (3 Queries) ---

    # Q4: Analyze Average Delays for a Specific Route (Origin -> Dest)
    if intent == "analyze_delays" and entities.get('origin') and entities.get('destination'):
        query = """
        MATCH (f:Flight)-[:DEPARTS_FROM]->(o:Airport {station_code: $origin})
        MATCH (f)-[:ARRIVES_AT]->(d:Airport {station_code: $destination})
        MATCH (j:Journey)-[:ON]->(f)
        RETURN f.flight_number, avg(j.arrival_delay_minutes) as Avg_Delay, max(j.arrival_delay_minutes) as Max_Delay
        ORDER BY Avg_Delay DESC;
        """
        return query, entities

    # Q5: Identify "Problem Airports" (High Delays departing from X)
    if intent == "analyze_delays" and entities.get('origin'):
        query = """
        MATCH (f:Flight)-[:DEPARTS_FROM]->(a:Airport {station_code: $origin})
        MATCH (j:Journey)-[:ON]->(f)
        WITH f, avg(j.arrival_delay_minutes) as flight_avg_delay
        WHERE flight_avg_delay > 15
        RETURN f.flight_number, f.fleet_type_description, flight_avg_delay
        ORDER BY flight_avg_delay DESC LIMIT 5;
        """
        return query, entities

    # Q6: Analyze Delays by Fleet Type (e.g., "Are Boeing 737s usually late?")
    if intent == "analyze_delays" and entities.get('fleet_desc'):
        query = """
        MATCH (j:Journey)-[:ON]->(f:Flight)
        WHERE f.fleet_type_description CONTAINS $fleet_desc
        RETURN f.fleet_type_description as Fleet, avg(j.arrival_delay_minutes) as Avg_Delay
        LIMIT 5;
        """
        return query, entities


    # --- GROUP 3: PASSENGER SATISFACTION (2 Queries) ---

    # Q7: Food Satisfaction by Passenger Class (Business vs Economy)
    if intent == "satisfaction_analysis" and entities.get('p_class'):
        query = """
        MATCH (j:Journey {passenger_class: $p_class})-[:ON]->(f:Flight)
        RETURN j.passenger_class, avg(j.food_satisfaction_score) as Avg_Food_Score, count(j) as Total_Pax
        """
        return query, entities

    # Q8: Customer Complaints Check (Find flights with low satisfaction)
    if intent == "satisfaction_analysis": 
        # Default query if no specific entity is provided, finds worst flights
        query = """
        MATCH (j:Journey)-[:ON]->(f:Flight)
        WITH f, avg(j.food_satisfaction_score) as score
        WHERE score < 3
        RETURN f.flight_number, f.fleet_type_description, score
        ORDER BY score ASC LIMIT 5;
        """
        return query, entities


    # --- GROUP 4: PASSENGER PROFILING (2 Queries) ---

    # Q9: Loyalty Program Analysis (e.g., "Do Gold members complain more?")
    if intent == "passenger_profiling" and entities.get('level'):
        query = """
        MATCH (p:Passenger {loyalty_program_level: $level})-[:TOOK]->(j:Journey)
        RETURN p.loyalty_program_level, avg(j.food_satisfaction_score) as Avg_Food_Rating, avg(j.arrival_delay_minutes) as Avg_Delay_Exp
        """
        return query, entities

    # Q10: Passenger History Lookup (by Record Locator)
    if intent == "passenger_profiling" and entities.get('record_locator'):
        query = """
        MATCH (p:Passenger {record_locator: $record_locator})-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
        RETURN p.record_locator, f.flight_number, j.passenger_class, j.arrival_delay_minutes
        """
        return query, entities

    # Fallback if no specific template matches
    return None, None


def process_user_query(user_input: str):
    print(f"--- Processing: '{user_input}' ---")
    
    # Step 1: Classify Intent
    intent_result = classification_chain.invoke({"query": user_input})
    print(f"✅ Intent: {intent_result.intent}")
    
    # Step 2: Extract Entities
    entity_result = extraction_chain.invoke({"query": user_input})
    print(f"✅ Entities: {entity_result.model_dump()}")
    
    # Step 3: Generate Cypher Query (Router)
    cypher_query, params = generate_cypher_query(intent_result.intent, entity_result.model_dump())
    
    results = None
    if cypher_query:
        print(f"✅ Cypher Query:\n{cypher_query}")
        print(f"✅ Parameters: {params}")
        
        # Step 4: Execute Query
        print("--- Executing in Neo4j ---")
        results = db.query(cypher_query, params)
        if results:
            print(f"✅ Results ({len(results)} records):")
            for r in results:
                print(r)
        else:
            print("⚠️ No results found or query failed.")
            
    else:
        print("❌ No Cypher query generated.")
    
    return intent_result.intent, entity_result.model_dump(), cypher_query, results

# --- TEST EXAMPLES ---
if __name__ == "__main__":
    print("--- Airline Assistant (Type 'quit' to exit) ---")
    try:
        while True:
            user_query = input("\nEnter your query: ")
            if user_query.lower() in ["quit", "exit"]:
                break
            
            try:
                process_user_query(user_query)
            except Exception as e:
                print(f"Error processing query: {e}")
    finally:
        db.close()
