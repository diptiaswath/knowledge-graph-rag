"""
    Loads healthcare CSV data into Neo4j to: 
    - create a knowledge graph with providers, patients, and treatment relationships 
    - setting up the foundation for vector embedding and semantic search functionality.

    This is the data preparation step that transforms a flat CSV into a connected graph database 
    structure before Graph-RAG capabilities.
"""
from dotenv import load_dotenv
import os
from langchain_neo4j import Neo4jGraph
import pandas as pd

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

# Load CSV 
print("=== Loading CSV Data ===")
df = pd.read_csv('../healthcare/healthcare.csv') 
print(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# Clear existing data 
print("\n=== Clearing Existing Data ===")
kg.query("MATCH (n) DETACH DELETE n")
print("Database cleared")

print("\n=== Create Knowledge Graph ===")
# Setup knowledge graph: 
# Healthcare Providers -  with name, bio, and location properties
# Patients - with name, age, and gender properties
# TREATS Relationships - connect providers to patients with specialization and condition properties

# Create unique providers first
print("\n=== Creating Healthcare Providers ===")
unique_providers = df[['Provider', 'Bio', 'Location', 'Specialization']].drop_duplicates(subset=['Provider'])

for _, row in unique_providers.iterrows():
    kg.query("""
        MERGE (hp:HealthcareProvider {name: $name})
        SET hp.bio = $bio,
            hp.location = $location,
            hp.specialization= $specialization
        """,
        params={
            "name": row['Provider'],
            "bio": row['Bio'],
            "location": row['Location'],
            "specialization":row['Specialization']
        }
    )

print(f"Created {len(unique_providers)} unique healthcare providers")

# Create unique patients
print("\n=== Creating Patients ===")
unique_patients = df[['Patient', 'Patient_Age', 'Patient_Gender']].drop_duplicates(subset=['Patient'])

for _, row in unique_patients.iterrows():
    kg.query("""
        MERGE (p:Patient {name: $name})
        SET p.age = $age,
            p.gender = $gender
        """,
        params={
            "name": row['Patient'],
            "age": int(row['Patient_Age']),
            "gender": row['Patient_Gender']
        }
    )

print(f"Created {len(unique_patients)} unique patients")

# Create TREATS relationships with properties: specialization and condition
# Why Relationships Store Data
# The specialization and condition are stored on the relationship rather than the nodes because:
# Same provider can treat different patients with different specializations
# Same patient can be treated for different conditions
# This captures the specific context of each treatment interaction
print("\n=== Creating TREATS Relationships ===")
for _, row in df.iterrows():
    kg.query("""
        MATCH (hp:HealthcareProvider {name: $provider_name})
        MATCH (p:Patient {name: $patient_name})
        MERGE (hp)-[r:TREATS]->(p)
        SET r.specialization = $specialization,
            r.condition = $condition
        """,
        params={
            "provider_name": row['Provider'],
            "patient_name": row['Patient'],
            "specialization": row['Specialization'],
            "condition": row['Patient_Condition']
        }
    )

print(f"Created {len(df)} TREATS relationships")

# Verification
print("\n=== Database Verification ===")

# Count nodes
provider_count = kg.query("MATCH (hp:HealthcareProvider) RETURN COUNT(hp) as count")[0]['count']
patient_count = kg.query("MATCH (p:Patient) RETURN COUNT(p) as count")[0]['count']
relationship_count = kg.query("MATCH ()-[r:TREATS]->() RETURN COUNT(r) as count")[0]['count']

print(f"Healthcare Providers: {provider_count}")
print(f"Patients: {patient_count}")
print(f"TREATS relationships: {relationship_count}")

# Sample data verification
print("\n=== Sample Healthcare Providers ===")
sample_providers = kg.query("""
    MATCH (hp:HealthcareProvider)
    RETURN hp.name, hp.bio, hp.location
    LIMIT 3
""")

for provider in sample_providers:
    print(f"Name: {provider['hp.name']}")
    print(f"Bio: {provider['hp.bio']}")
    print(f"Location: {provider['hp.location']}")
    print("---")

# Sample relationships
print("\n=== Sample Relationships ===")
sample_relationships = kg.query("""
    MATCH (hp:HealthcareProvider)-[r:TREATS]->(p:Patient)
    RETURN hp.name, r.specialization, r.condition, p.name, p.age
    LIMIT 3
""")

for rel in sample_relationships:
    print(f"{rel['hp.name']} treats {rel['p.name']} (age {rel['p.age']}) for {rel['r.condition']} in {rel['r.specialization']}")

print("\n=== Data Loading Complete! ===")
print("Neo4j datasbse is now ready for vector embedding generation!")


print("\nNext steps:")
print("1. Create the vector index")
print("2. Generate embeddings for provider bios")
print("3. Perform semantic search")