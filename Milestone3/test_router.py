from preprocessing import process_user_query
import time

test_cases = [
    # GROUP 1: FLIGHT LOOKUP & ROUTES
    {
        "id": "Q1",
        "query": "Find flights from IAX to LAX",
        "expected_intent": "flight_search",
        "desc": "Origin + Destination"
    },
    {
        "id": "Q2",
        "query": "Find flights to LAX",
        "expected_intent": "flight_search",
        "desc": "Destination Only"
    },
    {
        "id": "Q3",
        "query": "Show me details for flight 1878",
        "expected_intent": "flight_search",
        "desc": "Flight Number"
    },
    
    # GROUP 2: DELAY ANALYSIS
    {
        "id": "Q4",
        "query": "How are the delays for flights from IAX to LAX?",
        "expected_intent": "analyze_delays",
        "desc": "Avg Delay (Origin + Dest)"
    },
    {
        "id": "Q5",
        "query": "Which flights from IAX have high delays?",
        "expected_intent": "analyze_delays",
        "desc": "Problem Airports (Origin)"
    },
    {
        "id": "Q6",
        "query": "Are Boeing 737s usually late?",
        "expected_intent": "analyze_delays",
        "desc": "Fleet Type Delay"
    },
    
    # GROUP 3: PASSENGER SATISFACTION
    {
        "id": "Q7",
        "query": "Are Business class passengers happy with the food?",
        "expected_intent": "satisfaction_analysis",
        "desc": "Food Satisfaction by Class"
    },
    {
        "id": "Q8",
        "query": "Show me flights with bad food ratings",
        "expected_intent": "satisfaction_analysis",
        "desc": "Low Satisfaction Flights"
    },
    
    # GROUP 4: PASSENGER PROFILING
    {
        "id": "Q9",
        "query": "Do Gold members complain more?",
        "expected_intent": "passenger_profiling",
        "desc": "Loyalty Program Analysis"
    },
    {
        "id": "Q10",
        "query": "Show history for passenger with record locator ABC123",
        "expected_intent": "passenger_profiling",
        "desc": "Passenger History"
    }
]

print("=== STARTING ROUTER TEMPLATE TESTS (With Delays) ===")

passed_count = 0
failed_count = 0
failed_tests = []
results_found_count = 0

for case in test_cases:
    print(f"\n--- TEST {case['id']}: {case['desc']} ---")
    print(f"Query: {case['query']}")
    
    try:
        intent, entities, cypher, results = process_user_query(case['query'])
        
        # Validation
        error_reasons = []
        if intent != case['expected_intent']:
            error_reasons.append(f"Intent Mismatch (Expected {case['expected_intent']}, got {intent})")
        
        if not cypher:
            error_reasons.append("No Cypher Generated")
            
        if error_reasons:
            print(f"❌ FAILED: {', '.join(error_reasons)}")
            failed_count += 1
            failed_tests.append(f"{case['id']} ({case['desc']}): {', '.join(error_reasons)}")
        else:
            print("✅ PASSED (Intent & Cypher)")
            passed_count += 1
            if results:
                results_found_count += 1
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        failed_count += 1
        failed_tests.append(f"{case['id']} ({case['desc']}): Exception - {str(e)}")
    
    print("Waiting 5 seconds to respect API limits...")
    time.sleep(5)

print("\n" + "="*40)
print("       TEST EXECUTION SUMMARY")
print("="*40)
print(f"Total Tests Run: {len(test_cases)}")
print(f"Successful Executions: {passed_count}")
print(f"Tests with Data Returned: {results_found_count}")
print(f"Failed Tests: {failed_count}")

if failed_tests:
    print("\n❌ List of Failed Tests:")
    for failure in failed_tests:
        print(f"  - {failure}")
else:
    print("\n✅ All tests executed successfully!")
print("="*40)
