import streamlit as st
from streamlit_chat import message
from timeit import default_timer as timer
import traceback

from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import AzureChatOpenAI  # Updated import

import dotenv
import os

dotenv.load_dotenv()

# Test connection function
def test_connections():
    """Test Neo4j and OpenAI connections"""
    errors = []
    
    # Test Neo4j connection
    try:
        neo4j_url = os.getenv("NEO4J_CONNECTION_URL")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if not all([neo4j_url, neo4j_user, neo4j_password]):
            errors.append("Neo4j credentials missing in .env file")
        else:
            graph = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_password)
            # Test query
            result = graph.query("RETURN 'Connection successful' as message")
            st.success("✅ Neo4j connection successful")
    except Exception as e:
        errors.append(f"Neo4j connection failed: {str(e)}")
    
    # Test OpenAI connection
    try:
        openai_base = os.getenv("OPENAI_API_BASE")
        openai_key = os.getenv("OPENAI_API_KEY")
        openai_version = os.getenv("OPENAI_API_VERSION")
        
        if not all([openai_base, openai_key, openai_version]):
            errors.append("OpenAI credentials missing in .env file")
        else:
            llm = AzureChatOpenAI(
                azure_deployment="chat-gpt4",  # Updated parameter name
                azure_endpoint=openai_base,
                api_version=openai_version,
                api_key=openai_key,
                temperature=0
            )
            # Test simple call
            response = llm.predict("Hello")
            st.success("✅ OpenAI connection successful")
    except Exception as e:
        errors.append(f"OpenAI connection failed: {str(e)}")
    
    return errors

# OpenAI API configuration - Alternative approach
try:
    import os
    from langchain_openai import AzureChatOpenAI
    
    llm = AzureChatOpenAI(
        azure_deployment="chat-gpt4",  # Updated parameter name
        azure_endpoint=os.getenv("OPENAI_API_BASE"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        api_key=os.getenv("OPENAI_API_KEY"),  # Direct API key
        temperature=0
    )
    
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {e}")
    st.stop()

# Neo4j configuration
neo4j_url = os.getenv("NEO4J_CONNECTION_URL")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# Cypher generation prompt - Updated with proper escaping
cypher_generation_template = """
You are an expert Neo4j Cypher translator who converts Vietnamese to Cypher based on the Neo4j Schema provided, following the instructions below:
1. Generate Cypher query compatible ONLY for Neo4j Version 5
2. Do not use EXISTS, SIZE, HAVING keywords in the cypher. Use alias when using the WITH keyword
3. Use only Nodes and relationships mentioned in the schema
4. Use exact matching with node labels and property names as shown in examples
5. Available node types: Disease, SubDisease, Symptom, Complication, Treatment, Description, TimeStage, Cause, Population, Duration, Type, Action, Goal, Topic, Advice, Mechanism, ConditionNote, Note, RiskFactor, Definition
6. Available relationships: HAS_SYMPTOM, HAS_CAUSE, HAS_TREATMENT, HAS_DESCRIPTION, OCCURS_AT, HAS_SUBTYPE, AFFECTS, HAS_COMPLICATION, HAS_TYPE, OCCURS_IN, LASTS_FOR, EXPECTED_BY_AGE, HAS_DEFINITION, RECOMMENDED_FOR
7. When searching for diseases or conditions, use exact name matching with property syntax

schema: {schema}

Examples:
Question: Các triệu chứng của Vô kinh thứ phát?
Answer: ```MATCH (sd:SubDisease {{name: "Vô kinh thứ phát"}})-[:HAS_SYMPTOM]->(s:Symptom)
RETURN s.name ```
Question: Các triệu chứng của Chứng kinh nguyệt ẩn?
Answer: ```MATCH (sd:SubDisease {{name: "Chứng kinh nguyệt ẩn"}})-[:HAS_SYMPTOM]->(s:Symptom)
RETURN s.name ```
Question: Các triệu chứng của Mãn kinh?
Answer: ```MATCH (sd:Disease {{name: "Mãn kinh"}})-[:HAS_SYMPTOM]->(s:Symptom)
RETURN s.name  ```
Question: Cách điều trị U xơ tử cung? 
Answer: ```MATCH (d:Disease {{name: "U xơ tử cung"}})-[:HAS_TREATMENT]->(t:Treatment)
RETURN t.name```
Question: Cách điều trị Viêm âm đạo?
Answer: ```MATCH (d:Disease {{name: "Viêm âm đạo"}})-[:HAS_TREATMENT]->(t:Treatment)
RETURN t.name```
Question: Cách điều trị Bệnh viêm vùng chậu?
Answer: ```MATCH (d:Disease {{name: "Bệnh viêm vùng chậu"}})-[:HAS_TREATMENT]->(t:Treatment)
RETURN t.name```
Question: Mãn kinh diễn ra ở người nào?
Answer: ```MATCH (d:Disease {{name: "Mãn kinh"}})-[:AFFECTS]->(p:Population)
RETURN p.name```

Question: {question}
"""

cypher_prompt = PromptTemplate(
    template=cypher_generation_template,
    input_variables=["schema", "question"]
)

# Updated QA Template to handle database results properly
CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
If the provided information is empty, say that you don't know the answer.
Final answer should be easily readable and structured.
The final answer should be in Vietnamese.

CRITICAL: If the context contains database results in list format, extract ONLY the values and present them as a clear list. Do NOT add your own knowledge.

For example:
- If context shows results about types or categories, list them clearly
- If context shows symptoms, treatments, or causes, present them in numbered format
- Always use the exact values from the database

Information:
{context}

Question: {question}
Helpful Answer:"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

def query_graph(user_input):
    try:
        graph = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_password)
        
        # Test graph connection
        test_query = graph.query("RETURN 1 as test")
        print(f"Graph connection test: {test_query}")
        
        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            return_intermediate_steps=True,
            cypher_prompt=cypher_prompt,
            qa_prompt=qa_prompt,
            allow_dangerous_requests=True
        )
        
        print(f"Processing question: {user_input}")
        result = chain(user_input)
        print(f"Chain result: {result}")
        
        return result
    except Exception as e:
        print(f"Error in query_graph: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise e

# Streamlit UI
st.set_page_config(layout="wide")

if "user_msgs" not in st.session_state:
    st.session_state.user_msgs = []
if "system_msgs" not in st.session_state:
    st.session_state.system_msgs = []

title_col, empty_col, img_col = st.columns([2, 1, 2])    

with title_col:
    st.title("Conversational Neo4J Assistant")
with img_col:
    st.image("https://dist.neo4j.com/wp-content/uploads/20210423062553/neo4j-social-share-21.png", width=200)

# Add connection test button
if st.button("Test Connections"):
    with st.spinner("Testing connections..."):
        errors = test_connections()
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
        else:
            st.success("✅ All connections working!")

# Add debug info
with st.expander("Debug Info"):
    st.write("Environment Variables:")
    st.write(f"- NEO4J_CONNECTION_URL: {'✅' if os.getenv('NEO4J_CONNECTION_URL') else '❌'}")
    st.write(f"- NEO4J_USER: {'✅' if os.getenv('NEO4J_USER') else '❌'}")
    st.write(f"- NEO4J_PASSWORD: {'✅' if os.getenv('NEO4J_PASSWORD') else '❌'}")
    st.write(f"- OPENAI_API_BASE: {'✅' if os.getenv('OPENAI_API_BASE') else '❌'}")
    st.write(f"- OPENAI_API_KEY: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
    st.write(f"- OPENAI_API_VERSION: {'✅' if os.getenv('OPENAI_API_VERSION') else '❌'}")

user_input = st.text_input("Enter your question", key="input")

if user_input:
    cypher_query = None
    database_results = None
    
    with st.spinner("Processing your question..."):
        st.session_state.user_msgs.append(user_input)
        start = timer()

        try:
            result = query_graph(user_input)
            print(f"Full result: {result}")
            
            if "intermediate_steps" in result and result["intermediate_steps"]:
                intermediate_steps = result["intermediate_steps"]
                if len(intermediate_steps) > 0:
                    cypher_query = intermediate_steps[0].get("query", "No query generated")
                if len(intermediate_steps) > 1:
                    database_results = str(intermediate_steps[1].get("context", "No results"))
            
            answer = result.get("result", "No answer generated")
            st.session_state.system_msgs.append(answer)
            
            # Show success message
            st.success("✅ Question processed successfully!")
            
        except Exception as e:
            error_msg = f"Failed to process question: {str(e)}"
            st.error(error_msg)
            
            # Show detailed error in expander
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            
            print(f"Error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")

    st.write(f"Time taken: {timer() - start:.2f}s")

    col1, col2, col3 = st.columns([1, 1, 1])

    # Display the chat history
    with col1:
        st.subheader("Chat History")
        if st.session_state["system_msgs"]:
            for i in range(len(st.session_state["system_msgs"]) - 1, -1, -1):
                message(st.session_state["system_msgs"][i], key=str(i) + "_assistant")
                message(st.session_state["user_msgs"][i], is_user=True, key=str(i) + "_user")

    with col2:
        st.subheader("Generated Cypher Query")
        if cypher_query:
            st.text_area("Last Cypher Query", cypher_query, key="_cypher", height=240)
        else:
            st.info("No Cypher query generated yet")
        
    with col3:
        st.subheader("Database Results")
        if database_results:
            st.text_area("Last Database Results", database_results, key="_database", height=240)
        else:
            st.info("No database results yet")