import os
from neo4j import GraphDatabase

def load_config(file_path="../config.txt"):
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
    
config = load_config()
URI = config.get("uri")
USER = config.get("username")
PASSWORD = config.get("password")    
    
class Neo4jConnection:
    def __init__(self):
        # Placeholder credentials - PLEASE UPDATE THESE
        config = load_config()
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
            self.driver.verify_connectivity()
            print(f"✅ Connected to Neo4j at {URI}")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")

    def close(self):
        if self.driver:
            self.driver.close()

    def query(self, cypher_query, parameters=None):
        if not self.driver:
            return None
        
        with self.driver.session() as session:
            try:
                result = session.run(cypher_query, parameters)
                return [record.data() for record in result]
            except Exception as e:
                print(f"❌ Query failed: {e}")
                return None

# Singleton instance
# db = Neo4jConnection() 
# We will initialize this in the main script to avoid connection on import if credentials aren't set
