"""
Neo4j Hybrid RAG System for Roman Empire Q&A
A system that combines graph-based and vector-based retrieval for intelligent question answering.
"""

import os
from typing import List, Tuple
from dotenv import load_dotenv

# LangChain imports
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Neo4j imports
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars

# Pydantic for structured output
from pydantic import BaseModel, Field


class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that appear in the text",
    )


class Neo4jRAGSystem:
    """
    Neo4j-based hybrid RAG system combining graph and vector retrieval.
    """
    
    def __init__(self):
        """Initialize the RAG system with Neo4j and OpenAI connections."""
        load_dotenv()
        self._setup_connections()
        self._setup_models()
        self._setup_retrievers()
        self._setup_chain()
    
    def _setup_connections(self):
        """Setup Neo4j and OpenAI connections."""
        # Neo4j configuration
        self.neo4j_uri = os.environ["NEO4J_URI"]
        self.neo4j_username = os.environ["NEO4J_USERNAME"]
        self.neo4j_password = os.environ["NEO4J_PASSWORD"]
        
        # OpenAI configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize Neo4j graph
        self.kg = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
        )
    
    def _setup_models(self):
        """Setup LLM and embedding models."""
        self.chat = ChatOpenAI(
            api_key=self.openai_api_key, 
            temperature=0, 
            model="gpt-4o-mini"
        )
        
        self.embeddings = OpenAIEmbeddings()
    
    def _setup_retrievers(self):
        """
            Setup hybrid retrieval system combining vector and structured search.
    
            Creates two complementary retrieval mechanisms:
            1. Vector index for semantic similarity search on document content
            2. Entity extraction chain for structured graph traversal
            3. Fulltext index for fuzzy entity matching with typo tolerance
            
            Example workflow:
            - User: "What did Augustus accomplish?"
            - Entity extraction → ["Augustus"] 
            - Graph search → Augustus relationships
            - Vector search → semantic content about accomplishments
            - Result → precise relationships + contextual information   
        """
        
        # Vector index for hybrid search
        # Creates a vector index from existing Neo4j graph data
        # Enables Hybrid search = vector similarity + traditional keyword search
        # Looks for Document nodes that have both text content and embedding vectors
        self.vector_index = Neo4jVector.from_existing_graph(
            self.embeddings,                    # OpenAI embeddings model
            search_type="hybrid",               # Combines vector similarity with keyword search
            node_label="Document",              # Look for nodes labeled "Document"
            text_node_properties=["text"],      # Uses "text" property for content
            embedding_node_property="embedding", # Stores vectors in "embedding" property
        )
        
        # Entity extraction chain
        # Creates an LLM chain that extracts entities from user questions
        # Takes input like: "When did Augustus become emperor?"
        # Returns structured output: Entities(names=["Augustus"])
        # The | operator chains: prompt → LLM → structured parsing
        entity_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are extracting organization and person entities from the text."),
            ("human", "Use the given format to extract information from the following input: {question}"),
        ])
        self.entity_chain = entity_prompt | self.chat.with_structured_output(Entities)
        

        # Fulltext search index creation on entity nodes
        # Creates a fulltext index named "entity" that enabled text-based search capabilities on nodes with __Entity__label using its id property
        # Enables fuzzy matching: "Augusus" → "Augustus" (handles typos)
        # IF NOT EXISTS prevents errors if index already exists
        # Searches across all nodes with __Entity__ label 
        self.kg.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

        
    
    def _setup_chain(self):
        """
            Setup the main conversational RAG pipeline that handles both standalone questions 
            and followup questions with chat history

            Example Flow:
            Input: {"question": "When did he become emperor?", "chat_history": [...]}
                ↓
            RunnableParallel splits into:
                ├─ "context": search_query → retriever → "Augustus became emperor in 27 BC..."
                ├─ "question": "When did he become emperor?" (passed through)
                ↓
            answer_prompt: Combines context + question into final prompt
                ↓
            self.chat: LLM generates answer
                ↓
            StrOutputParser: "Augustus became the first Roman emperor in 27 BC."
        """
        
        # Template for condensing chat history
        # Converts context-dependent questions into self-contained ones
        # Example: 
        # Chat History: "Who was the first emperor?" → "Augustus was the first emperor."
        # Follow-up: "When did he become emperor?"
        # Output standalone question: "When did Augustus become the first Roman emperor?"
        condense_template = """Given the following conversation and a follow up question, 
                rephrase the follow up question to be a standalone question, in its original language.
                
                Chat History: {chat_history}
                Follow Up Input: {question}
                Standalone question:"""
        
        self.condense_question_prompt = PromptTemplate.from_template(condense_template)
        
        # Search query branch for handling chat history
        # Checks: Does input have chat history?
        # If YES: Format history → condense with current question → return standalone question
        # If NO: Just pass through the original question
        self.search_query = RunnableBranch(
            # Branch1: If chat history exists
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
                # Process: format history → condense question → get standalone question
                RunnablePassthrough.assign(chat_history=lambda x: self._format_chat_history(x["chat_history"]))
                | self.condense_question_prompt
                | ChatOpenAI(temperature=0)
                | StrOutputParser(),
            ),
            # Branch2: No chat history, use question as-is
            RunnableLambda(lambda x: x["question"]),
        )
        
        # Final answer template
        answer_template = """Answer the question based only on the following context:
        {context}
        
        Question: {question}
        Use natural language and be concise.
        Answer:"""
        
        answer_prompt = ChatPromptTemplate.from_template(answer_template)
        
        # Complete RAG chain
        self.chain = (
            RunnableParallel({
                "context": self.search_query | self.retriever,   # Get Context
                "question": RunnablePassthrough(),               # Pass question through
            }) 
            | answer_prompt                                      # Format context, question with template
            | self.chat                                          # Send to LLM gpt-4o-mini
            | StrOutputParser()                                  # Extract text response
        )

    def _format_chat_history(self, chat_history: List[Tuple[str, str]]) -> List:
        """Format chat history for conversation chain."""
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer


class DataIngestion:
    """
        Handles data loading and graph creation.
        It does so by transforming unstructured text into structured knowledge graph.  
    """
    
    def __init__(self, rag_system: Neo4jRAGSystem):
        self.rag_system = rag_system
    
    def load_and_process_wikipedia(self, query: str = "The Roman empire", num_docs: int = 3):
        """
           Load Wikipedia articles and convert to graph format.
        """
        print(f"Loading Wikipedia articles for: {query}")
        
        # Load documents by downloading wiki articles for the query 
        raw_documents = WikipediaLoader(query=query).load()
        
        # Split into chunks - only first 3 num_docs processed
        text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
        documents = text_splitter.split_documents(raw_documents[:num_docs])
        
        # Convert to graph documents
        # LLM reads each text chunk and extracts entities and relationships to get output graph
        # Input text: "Augustus defeated Mark Antony at the Battle of Actium in 31 BC."
        # Output graph:
        # - Entity: Augustus
        # - Entity: Mark Antony  
        # - Entity: Battle of Actium
        # - Relationship: (Augustus)-[DEFEATED]->(Mark Antony)
        # - Relationship: (Battle of Actium)-[OCCURRED_IN]->(31 BC) 
        llm_transformer = LLMGraphTransformer(llm=self.rag_system.chat)
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        
        # Store in Neo4j
        # Saves both the graph structure AND original text
        # Creates nodes and relationships in Neo4j
        # Enables both graph queries and vector search
        print("Storing documents in Neo4j...")
        self.rag_system.kg.add_graph_documents(
            graph_documents,
            include_source=True,  # Keep original text for vector search
            baseEntityLabel=True, # Add __Entity__ label for indexing
        )
        print("Data ingestion complete!")
    
    def clear_database(self):
        """
            Clear all data from Neo4j database. 
            Match(n): Finds all nodes
            Detach Delete: Removes nodes and their relationships
        """
        print("Clearing Neo4j database...")
        self.rag_system.kg.query("MATCH (n) DETACH DELETE n")
        print("Database cleared!")

# Extending the class to add retrieval methods
class Neo4jRAGSystem(Neo4jRAGSystem):  
    
    def generate_full_text_query(self, input_text: str) -> str:
        """
           Generate a full-text search query with fuzzy matching.
           Fuzzy Search Builder that returns Neo4j fulltext query syntax.
        """
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input_text).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()
    
    def structured_retriever(self, question: str) -> str:
        """
            Retrieve structured data based on entities in the question.
            Graph Relationship Finder: 
            1. Extract entities from question: "What did Augustus accomplish?" → ["Augustus"]
            2. Find matching nodes in graph using fuzzy search
            3. Get relationships for each entity (excluding MENTIONS relationships)
            4. Return relationship strings : "Augustus - RULED -> Roman Empire"
        """
        result = ""
        entities = self.entity_chain.invoke({"question": question})
        
        for entity in entities.names:
            print(f"Getting entity relationships for: {entity}")
            response = self.kg.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL (node) {
                  WITH node
                  MATCH (node)-[r:!MENTIONS]->(neighbor)
                  RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                  UNION ALL
                  WITH node
                  MATCH (node)<-[r:!MENTIONS]-(neighbor)
                  RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": self.generate_full_text_query(entity)},
            )
            result += "\n".join([el["output"] for el in response])
        return result
    
    def retriever(self, question: str) -> str:
        """
            Combined retriever using both structured and unstructured data.
           
            Structured: Specific relationships from knowledge graph
            Unstructured: Semantically similar document chunks

            Example output:
            Structured data:
            Augustus - BECAME -> Roman Emperor
            Augustus - DEFEATED -> Mark Antony

            Unstructured data:
            #Document Augustus was the founder of the Roman Empire...
            #Document The transition from Republic to Empire occurred...
        """
        print(f"Processing question: {question}")
        
        # Get structured graph data
        structured_data = self.structured_retriever(question)
        
        # Get unstructured vector data
        unstructured_data = [
            el.page_content for el in self.vector_index.similarity_search(question)
        ]
        
        # Combine both data sources
        final_data = f"""Structured data:
                {structured_data}

                Unstructured data:
                {"#Document ".join(unstructured_data)}
                """
        
        # print(f'Retrived context: {final_data}')
        print(f"Retrieved context length: {len(final_data)} characters")
        return final_data
    
    def ask(self, question: str, chat_history: List[Tuple[str, str]] = None) -> str:
        """Ask a question to the RAG system."""
        query_input = {"question": question}
        if chat_history:
            query_input["chat_history"] = chat_history
        
        return self.chain.invoke(query_input)


def main():
    """Main function to demonstrate the RAG system."""
    print("Initializing Neo4j RAG System...")
    rag_system = Neo4jRAGSystem()
    
    # Load new data
    data_ingestion = DataIngestion(rag_system)
    # Clear existing data
    data_ingestion.clear_database()  
    # Load and process wikipedia 
    data_ingestion.load_and_process_wikipedia("The Roman empire")
    
    print("\n" + "="*100)
    print("Neo4j RAG System Ready!")
    print("="*100)
    
    # 1: Simple question
    print("\n1. Demo Simple Question:")
    simple_question = "How did the Roman empire fall?"
    simple_answer = rag_system.ask(simple_question)
    print(f"Q: {simple_question}")
    print(f"A: {simple_answer}")
    
    # 2: Conversational question with history
    print("\n2. Demo Conversational Question:")
    chat_history = [("Who was the first emperor?", "Augustus was the first emperor.")]
    follow_up_question = "When did he become the first emperor?"
    conversational_answer = rag_system.ask(follow_up_question, chat_history)
    print(f"Previous context: {chat_history[0]}")
    print(f"Q: {follow_up_question}")
    print(f"A: {conversational_answer}")
    
    # Interactive mode
    print("\n3. Interactive Mode:")
    print("Ask questions about the Roman Empire (type 'quit' to exit):")
    
    conversation_history = []
    while True:
        user_question = input("\nYour question: ").strip()
        if user_question.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_question:
            answer = rag_system.ask(user_question, conversation_history)
            print(f"Answer: {answer}")
            
            # Add to conversation history
            conversation_history.append((user_question, answer))
            
            # Keep only last 3 exchanges to manage context length
            if len(conversation_history) > 3:
                conversation_history = conversation_history[-3:]


if __name__ == "__main__":
    main()