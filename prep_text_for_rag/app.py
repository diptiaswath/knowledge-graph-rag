"""
    This sets up the complete GraphRAG retrieval pipeline:  
        - creates vector index, 
        - generates embeddings from healthcare provider bios stored in a knowledge graph, and 
        - performs semantic search to find relevant providers based on natural language queries.

    This is the retrieval component of GraphRAG that combines:

        - Knowledge graph structure (providers, patients, relationships)
        - Vector embeddings for semantic understanding
        - Intelligent search that finds providers by meaning, not just keywords

    SO: 
        - This retriever understands CONCEPTUAL RELATIONSHIPS that humans intuitively know that 
        traditional search can't handle.

        - Code also stops at retrieval - to complete GraphRAG, we'd add an LLM generation step that uses the 
        retrieved provider information plus graph context to generate comprehensive answers.
"""

from dotenv import load_dotenv
import os
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI

load_dotenv()

AURA_INSTANCENAME = os.environ["AURA_INSTANCENAME"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

# Set up to use OpenAI API
chat = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")


# Connecting to Neo4j database instance
kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)   

# STEP1 - Create Vector Index named "health_providers_embeddings" for storing vector embeddings
# What it indexes: Indexes nodes with the label HealthCareProvider, specifically indexes the comprehensiveEmbeddings property
# Index Config:  Vectors are 1536 - Open AI embedding size, Uses cosine similarity for comparing vectors
kg.query(
     """
     CREATE VECTOR INDEX health_providers_embeddings IF NOT EXISTS
     FOR (hp:HealthcareProvider) ON (hp.comprehensiveEmbedding)
     OPTIONS {
       indexConfig: {
         `vector.dimensions`: 1536,
         `vector.similarity_function`: 'cosine'
       }
     }
     """
)

# STEP2 - Test to see if the index was created
res = kg.query(
     """
     SHOW VECTOR INDEXES
     """
 )
print("\n=== Vector Index named: health_providers_embeddings status ===")
print(res)

# STEP3 - Generate and store embeddings (populates the index automatically)
# Query to populate vector index by generating and storing embeddings for healthcare providers
# 1. Finds Healthcare Providers who treat patients and only includes providers that have a bio
# 2. Generate vector (semantic meaning) embeddings of each provider's bio, with OpenAI's embedding API
# 3. Filter valid vercors where the embeddding was successfully generated
# 4. Adds the vector as a property named "comprehensiveEmbedding" on each healthcare provider node
kg.query(
     """
     MATCH (hp:HealthcareProvider)-[:TREATS]->(p:Patient)
     WHERE hp.bio IS NOT NULL
     WITH hp, genai.vector.encode(
         hp.bio,
         "OpenAI",
         {
           token: $openAiApiKey,
           endpoint: $openAiEndpoint
         }) AS vector
     WITH hp, vector
     WHERE vector IS NOT NULL
     CALL db.create.setNodeVectorProperty(hp, "comprehensiveEmbedding", vector)
     """,
     params={
         "openAiApiKey": OPENAI_API_KEY,
         "openAiEndpoint": OPENAI_ENDPOINT,
     },
)

# STEP 4: Verify the embeddings were created
# Verify to see if the embedding generation worked by finding healthcare providers with bios 
# and return 3 pieces of data for each provider - hp.bio, hp.name, hp.comprehensiveEmbeddng - the vector embedding if it exists
# limits to 5 results
result = kg.query(
     """
     MATCH (hp:HealthcareProvider)
     WHERE hp.bio IS NOT NULL
     RETURN hp.bio, hp.name, hp.comprehensiveEmbedding
     LIMIT 5
     """
)

print("\n=== Embeddings verification: ===")
for record in result:
     print(f"Name: {record['hp.name']}")
     print(f"Bio: {record['hp.bio']}")
     print(f"Has embedding: {record['hp.comprehensiveEmbedding'] is not None}")
     print("---")



# == Querying the graph for a healthcare provider ==
# Matches: Any provider bio with similar meaning vectors to "dermatology" - vector embedding search
# Finds: Providers with "skin specialist", "acne treatment", "eczema expert", "cosmetic dermatology" in their bios
# V.S. 
# Match only by: Exact word "dermatology" in provider bio
# Keyword search Misses: Providers who say "skin specialist", "cosmetic proceedures" etc
question = "Give me a list of healthcare providers in the area of dermatology"

# Execute the query
result = kg.query(
    """
    WITH genai.vector.encode(
        $question,
        "OpenAI",
        {
          token: $openAiApiKey,
          endpoint: $openAiEndpoint
        }) AS question_embedding

    // Semantic Search: Perform vector similarity search against the specified vector index 
    // aka, finds the healthcare providers whose bio embeddings are most semantically similar to your question embedding
    // to return healthcare_provider nodes that are most similar with similaity score 1=identical, 0=different
    CALL db.index.vector.queryNodes(
        'health_providers_embeddings',
        $top_k,
        question_embedding
        ) YIELD node AS healthcare_provider, score
    
    
    // Graph Based Filtering: Enhance with relationship context to filter results to only include 
    // actual dermatology providers, eliminating false matches like pediatricians or neurologists
    MATCH (healthcare_provider)-[r:TREATS]->(p:Patient)
    WHERE r.specialization = 'Dermatology' 
       OR healthcare_provider.bio CONTAINS 'dermatolog'
       OR healthcare_provider.bio CONTAINS 'skin'
        
    // Relevance Boosting: Calculate specialty relevance boost
    // Enhance scores for providers with stronger dermatology connections - direct specialization gets highest boost
    WITH healthcare_provider, score,
        CASE 
            WHEN r.specialization CONTAINS 'Dermatology' THEN score + 0.1
            WHEN healthcare_provider.bio CONTAINS 'dermatolog' THEN score + 0.05
            WHEN healthcare_provider.bio CONTAINS 'skin' THEN score + 0.03
            ELSE score
        END AS boosted_score,
        // Context enrichment: that provides comprehensive context about each provider's practice areas
        collect(DISTINCT r.specialization) as specializations,
        collect(DISTINCT r.condition) as conditions_treated
        
        RETURN healthcare_provider.name, 
               healthcare_provider.bio,
               healthcare_provider.location,
               round(boosted_score * 1000) / 1000 as final_score,
               specializations,
               conditions_treated
        ORDER BY final_score DESC
        LIMIT 3
    """,
    params={
        "openAiApiKey": OPENAI_API_KEY,
        "openAiEndpoint": OPENAI_ENDPOINT,
        "question": question,
        "top_k": 3,
    },
)


print(f"\n=== Search results for: '{question}' ===")
for record in result:
    print(f"Name: {record['healthcare_provider.name']}")
    print(f"Bio: {record['healthcare_provider.bio']}")
    print(f"Location: {record['healthcare_provider.location']}")
    print(f"Specializations: {record['specializations']}")
    print(f"Conditions treated: {record['conditions_treated']}")
    print(f"Score: {record['final_score']}")
    print("---")
