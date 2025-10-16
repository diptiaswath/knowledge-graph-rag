"""
    Neo4j knowledge graph tutorial that demonstrates fundamental graph database operations using Albert Einstein as an example.

    Core Functionality
    1. Knowledge Graph Construction

    Creates nodes representing entities: Person (Einstein), Subject (Physics), NobelPrize, Countries (Germany, USA)
    Establishes relationships: STUDIED, WON, BORN_IN, DIED_IN
    Uses MERGE instead of CREATE to prevent duplicates

    2. Database Operations

    Connection management: Establishes and closes Neo4j database connections
    Transaction handling: Uses sessions and transactions for data integrity
    Error handling: Includes try-catch blocks for robust operation

    3. Query Patterns

    Simple queries: Returns node properties (like names)
    Path queries: Returns complete relationship paths
    Flexible query functions: Two different query methods for different use cases

    4. Learning Examples
    For various Cypher query patterns:

    Finding all nodes and relationships
    Directional relationship queries
    Data deletion operations
    Troubleshooting failed queries
"""
from dotenv import load_dotenv
import os
from neo4j import GraphDatabase

load_dotenv()

# Neo4j offers a free tier in its Aura cloud service 
# https://neo4j.com/docs/aura/classic/auradb/getting-started/create-database/
AURA_INSTANCENAME = os.environ["AURA_INSTANCENAME"] # Cloud-based instance
NEO4J_URI = os.environ["NEO4J_URI"]                 # Neo4j instance URI
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]       # Neo4j account username
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]       # Neo4j account password
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)


driver = GraphDatabase.driver(NEO4J_URI, auth=AUTH)              # Connect to Neo4j


def connect_and_query():
    """
        Connect and execute queries on the Neo4j instance
    """
    try:
        with driver.session() as session:                         # Opens a session with Neo4j
            result = session.run("MATCH (n) RETURN count(n)")     # Executes this query 
            count = result.single().value()
            print(f"Number of nodes: {count}")                    # Return the query result
    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.close()

def create_entities(tx):
    """
        Create Albert Einstein node along with other nodes using MERGE V.S. CREATE.

        Person node: (a:Person {name: 'Albert Einstein'})

            Label: Person
            Property: name = 'Albert Einstein'


        Subject node: (p:Subject {name: 'Physics'})

            Label: Subject
            Property: name = 'Physics'


        NobelPrize node: (n:NobelPrize {name: 'Nobel Prize in Physics'})

            Label: NobelPrize
            Property: name = 'Nobel Prize in Physics'


        Country nodes:

            (g:Country {name: 'Germany'})
            (u:Country {name: 'USA'})
    """
    # MERGE: Checks if the nodes or relationships already exist in the graph. 
    # If they do, it reuses them; if not, it creates them. 
    # This makes MERGE more suitable for ensuring no duplication of entities or relationships.
    tx.run("MERGE (a:Person {name: 'Albert Einstein'})")

    # Create other nodes
    tx.run("MERGE (p:Subject {name: 'Physics'})")
    tx.run("MERGE (n:NobelPrize {name: 'Nobel Prize in Physics'})")
    tx.run("MERGE (g:Country {name: 'Germany'})")
    tx.run("MERGE (u:Country {name: 'USA'})")


def create_relationships(tx):
    """
        Creates relationships between the nodes.

        The Four Relationships Being Created

        1. STUDIED: `Einstein → Physics`
        - Albert Einstein studied Physics

        2. WON: `Einstein → Nobel Prize in Physics`
        - Albert Einstein won the Nobel Prize in Physics

        3. BORN_IN: `Einstein → Germany`
        - Albert Einstein was born in Germany

        4. DIED_IN: `Einstein → USA`
        - Albert Einstein died in the USA
    
    """
    # Create studied relationship
    tx.run(
        """
        MATCH (a:Person {name: 'Albert Einstein'}), (p:Subject {name: 'Physics'})
        MERGE (a)-[:STUDIED]->(p)
        """
    )

    # Create won relationship
    tx.run(
        """
        MATCH (a:Person {name: 'Albert Einstein'}), (n:NobelPrize {name: 'Nobel Prize in Physics'})
        MERGE (a)-[:WON]->(n)
        """
    )

    # Create born in relationship
    tx.run(
        """
        MATCH (a:Person {name: 'Albert Einstein'}), (g:Country {name: 'Germany'})
        MERGE (a)-[:BORN_IN]->(g)
    """
    )

    # Create died in relationship
    tx.run(
        """
        MATCH (a:Person {name: 'Albert Einstein'}), (u:Country {name: 'USA'})
        MERGE (a)-[:DIED_IN]->(u)
        """
    )


# Function to connect and run a simple Cypher query
def query_graph_simple(cypher_query):
    """
        Connects to Neo4J and runs a simple Cypher query.

        This query is expected to return a column or field called "name".
        Use for queries with simple node properties. 
            Ex: MATCH (p:Person) RETURN p.name as name
            Return names like: Albert Einstein
        
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=AUTH)
    print(f"\nExecution of query: {cypher_query}")
    try:
        with driver.session() as session:             # database=NEO4J_DATABASE
            result = session.run(cypher_query)
            for record in result:
                print(record["name"])
    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.close()


def query_graph(cypher_query):
    """
         Connects to Neo4J and runs a Cypher query.

         This query is expected to return a column or field called "path".
         Use for queries that return relationship paths between nodes.
            Ex: MATCH path = (p:Person)-[:STUDIED]->(s:Subject) 
                RETURN path
            Returns complete paths showing relationships: (Einstein)-[:STUDIED]->(Physics)
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=AUTH)
    print(f"\nExecution of query: {cypher_query}")
    try:
        with driver.session() as session:              # database=NEO4J_DATABASE
            result = session.run(cypher_query)

            for record in result:
                print(record["path"])
    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.close()


def build_knowledge_graph():
    """
        Builds a Knowledge graph by opening a session with the Neo4j database instance
        and creates entities and relationships 
    """
    # Open a session with the Neo4j database
    try:
        with driver.session() as session:              # database=NEO4J_DATABASE
            session.execute_write(create_entities)     # create entities
            session.execute_write(create_relationships)# create relationships
    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.close()


if __name__ == "__main__":
    # Build the knowledge graph
    build_knowledge_graph()

    # Simple Cypher query to find all node names
    simple_query = """           
        MATCH (n)
        RETURN n.name AS name 
    """
    query_graph_simple(simple_query)

    # Cypher query to find all paths related to Albert Einstein
    einstein_query = """
        MATCH path=(a:Person {name: 'Albert Einstein'})-[:STUDIED]->(s:Subject)
        RETURN path
        UNION
        MATCH path=(a:Person {name: 'Albert Einstein'})-[:WON]->(n:NobelPrize)
        RETURN path
        UNION
        MATCH path=(a:Person {name: 'Albert Einstein'})-[:BORN_IN]->(g:Country)
        RETURN path
        UNION
        MATCH path=(a:Person {name: 'Albert Einstein'})-[:DIED_IN]->(u:Country)
        RETURN path
    """
    query_graph(einstein_query)


# # Run this to see the entire graph in the neo4j browser/console
# MATCH (n)-[r]->(m)
# RETURN n, r, m;

# Find specific entity and its relationship with other entities
# MATCH (n {name: 'Albert Einstein'})-[r]->(m)
# RETURN n, r, m

# No records returned for this - Find a Subject node with name 'Physics' (labeled as p)
# Looks for any relationship r going outward from Physics (->)
# Any node m that Physics points to. 
# However, we only have these relationships. 
# MERGE (a)-[:STUDIED]->(p)    # Einstein → Physics
# MERGE (a)-[:WON]->(n)        # Einstein → Nobel Prize
# MERGE (a)-[:BORN_IN]->(g)    # Einstein → Germany  
# MERGE (a)-[:DIED_IN]->(u)    # Einstein → USA
# MATCH (p:Subject {name: 'Physics'})-[r]->(m) RETURN p, r, m

# Look in both directions
# MATCH (p:Subject {name: 'Physics'})-[r]-(m) RETURN p, r, m 

# Look for incoming relationships
# MATCH (p:Subject {name: 'Physics'})<-[r]-(m) RETURN p, r, m


# Deleting all nodes and relationships
# This command ensures that all relationships connected to the nodes are also removed:
# MATCH (n)
# DETACH DELETE n

# Delete specific node and its associated relationships
# MATCH (n:Person {name: 'Albert Einstein'})
# DETACH DELETE n

