import streamlit as st
from streamlit_chat import message
from timeit import default_timer as timer

from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import AzureChatOpenAI

import dotenv
import os

dotenv.load_dotenv()

# OpenAI API configuration
llm = AzureChatOpenAI(
    deployment_name = "chat-gpt4",
    openai_api_base = os.getenv("OPENAI_API_BASE"),
    openai_api_version = os.getenv("OPENAI_API_VERSION"),
    openai_api_key = os.getenv("OPENAI_API_KEY"),
    openai_api_type = "azure",
    temperature = 0
)

#Neo4j configuration
neo4j_url = os.getenv("NEO4J_CONNECTION_URL")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")


# Cypher generation prompt
cypher_generation_template = """
You are an expert Neo4j Cypher translator who converts Vietnamese to Cypher based on the Neo4j Schema provided, following the instructions below:
1. Generate Cypher query compatible ONLY for Neo4j Version 5
2. Do not use EXISTS, SIZE, HAVING keywords in the cypher. Use alias when using the WITH keyword
3. Use only Nodes and relationships mentioned in the schema
4. All property-based searches must be case-insensitive and fuzzy (partial match) using the toLower(...) CONTAINS '...' pattern.
Examples:
To search for a disease:
toLower(disease.name) CONTAINS 'viêm âm đạo'
To search for a symptom:
toLower(symptom.name) CONTAINS 'đau bụng'
To search for treatment description:
toLower(treatment.description) CONTAINS 'kháng sinh'
5. Never use relationships that are not mentioned in the given schema
6. When asked about a Disease, always match properties using case-insensitive and fuzzy search, optionally with the OR-operator if multiple fields are defined.
Example — To find the disease “Viêm âm đạo”, use: toLower(disease.name) CONTAINS 'viêm âm đạo' OR toLower(disease.description) CONTAINS 'viêm âm đạo'
When asked about a disease, always use this pattern:
toLower(disease.name) CONTAINS '<disease name>'


schema: {schema}

Examples:
Question: Các triệu chứng của Vô kinh thứ phát?
Answer: ```MATCH (sd:SubDisease {name: "Vô kinh thứ phát"})-[:HAS_SYMPTOM]->(s:Symptom)
RETURN s.name ```
Question: Các triệu chứng của Chứng kinh nguyệt ẩn?
Answer: ```MATCH (sd:SubDisease {name: "Chứng kinh nguyệt ẩn"})-[:HAS_SYMPTOM]->(s:Symptom)
RETURN s.name ```
Question: Các triệu chứng của Mãn kinh?
Answer: ```MATCH (sd:Disease {name: "Mãn kinh"})-[:HAS_SYMPTOM]->(s:Symptom)
RETURN s.name  ```
Question: Cách điều trị U xơ tử cung? 
Answer: ```MATCH (d:Disease {name: "U xơ tử cung"})-[:HAS_TREATMENT]->(t:Treatment)
RETURN t.name```
Question: Cách điều trị Viêm âm đạo?
Answer: ```MATCH (d:Disease {name: "Viêm âm đạo"})-[:HAS_TREATMENT]->(t:Treatment)
RETURN t.name```
Question: Cách điều trị Bệnh viêm vùng chậu?
Answer: ```MATCH (d:Disease {name: "Bệnh viêm vùng chậu"})-[:HAS_TREATMENT]->(t:Treatment)
RETURN t.name```
Question: Mãn kinh diễn ra ở người nào?
Answer: ```MATCH (d:Disease {name: "Mãn kinh"})-[:AFFECTS]->(p:Population)
RETURN p.name```

Question: {question}
"""

cypher_prompt = PromptTemplate(
    template = cypher_generation_template,
    input_variables = ["schema", "question"]
)

CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
If the provided information is empty, say that you don't know the answer.
Final answer should be easily readable and structured.
The final answer should be in Vietnamese
Information:
{context}

Question: {question}
Helpful Answer:"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

def query_graph(user_input):
    graph = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_password)
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        return_intermediate_steps=True,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
        allow_dangerous_requests=True
        )
    result = chain(user_input)
    return result


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

user_input = st.text_input("Enter your question", key="input")
if user_input:
    cypher_query = None  # 👈 Thêm dòng này
    database_results = None  # 👈 Thêm dòng này
    with st.spinner("Processing your question..."):
        st.session_state.user_msgs.append(user_input)
        start = timer()

        try:
            result = query_graph(user_input)
            print(result) 
            
            intermediate_steps = result["intermediate_steps"]
            cypher_query = intermediate_steps[0]["query"]
            database_results = intermediate_steps[1]["context"]

            answer = result["result"]
            st.session_state.system_msgs.append(answer)
        except Exception as e:
            st.write("Failed to process question. Please try again.")
            print(e)

    st.write(f"Time taken: {timer() - start:.2f}s")

    col1, col2, col3 = st.columns([1, 1, 1])

    # Display the chat history
    with col1:
        if st.session_state["system_msgs"]:
            for i in range(len(st.session_state["system_msgs"]) - 1, -1, -1):
                message(st.session_state["system_msgs"][i], key = str(i) + "_assistant")
                message(st.session_state["user_msgs"][i], is_user=True, key=str(i) + "_user")

    with col2:
        if cypher_query:
            st.text_area("Last Cypher Query", cypher_query, key="_cypher", height=240)
        
    with col3:
        if database_results:
            st.text_area("Last Database Results", database_results, key="_database", height=240)
    