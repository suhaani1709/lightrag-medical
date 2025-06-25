import getpass
from langchain_openai import ChatOpenAI
from typing import Optional,TypedDict,Sequence,Annotated
from langchain_core.messages import BaseMessage
import operator
import os
import logging
openrouter_api_key=""
#  = getpass.getpass("Enter your OpenRouter API key: ")
# AIzaSyCiERMpXF57euE_QSmvrvlRkZ3BTtZLdQc

os.environ["GOOGLE_API_KEY"]= ""
os.environ["OPENAI_API_KEY"] = openrouter_api_key
os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
OPENAI_API_KEY = "" 

model = ChatOpenAI(temperature=0,streaming=True)
import os
from langchain_google_genai import ChatGoogleGenerativeAI


from typing import Optional,Literal

from dotenv import load_dotenv
from langchain_core.utils.utils import secret_from_env
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel,Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph,START,END
from langchain.schema import Document

import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.llama_index_impl import (
    llama_index_complete_if_cache,
    llama_index_embed,
)
from lightrag.utils import TokenTracker
from lightrag.utils import EmbeddingFunc,setup_logger
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from lightrag.namespace import NameSpace
import asyncio
import nest_asyncio

nest_asyncio.apply()

from lightrag.kg.shared_storage import initialize_pipeline_status
from huggingface_hub import login
login(token="your_hf_token_here")

WORKING_DIR = "./test2"
print(f"WORKING_DIR: {WORKING_DIR}")

os.environ["NEO4J_URI"] = "your_neo4j_uri_here"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "your_neo4j_password_here"
os.environ["MONGO_URI"] = "mongodb+srv://your_mongo_connection_here/?retryWrites=true&w=majority&appName=Cluster0"
os.environ["MONGO_DATABASE"] = "LightRAG"
os.environ["TAVILY_API_KEY"] = "your_tavily_key_here"

# Model configuration
LLM_MODEL = os.environ.get("LLM_MODEL", "cohere/command-a")
print(f"LLM_MODEL: {LLM_MODEL}")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))
print(f"EMBEDDING_MAX_TOKEN_SIZE: {EMBEDDING_MAX_TOKEN_SIZE}")
# AIzaSyCiERMpXF57euE_QSmvrvlRkZ3BTtZLdQc

from neo4j import AsyncGraphDatabase
async def test_connection():
    uri = "your_neo4j_uri_here"
    auth = ("neo4j", "your_neo4j_password_here")

    driver = AsyncGraphDatabase.driver(uri, auth=auth)
    try:
        async with driver.session() as session:
            result = await session.run("RETURN 1")
            value = await result.single()
            print(" Connected:", value)
    except Exception as e:
        print(" Connection failed:", e)
    finally:
        await driver.close()

asyncio.run(test_connection())

if not os.path.exists(WORKING_DIR):
    print(f"Creating working directory: {WORKING_DIR}")
    os.mkdir(WORKING_DIR)
token_tracker=TokenTracker()
setup_logger("lightrag", level="DEBUG")

NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION="neo4j"

import asyncio


tools = TavilySearchResults(max_results=1,include_answer=True)

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],operator.add]
    
class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("OPENROUTER_API_KEY", default=None),
    )
    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def _init_(self,
                 openai_api_key: Optional[str] = None,
                 **kwargs):
        openai_api_key = (
            openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        )
        super()._init_(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=openai_api_key,
            **kwargs
        )


llm = ChatOpenRouter(model_name="openai/gpt-4.1-nano",temperature=0,streaming= True)
# messages = [
#     (
#         "system",
#         "you are helpful assistant that gives me 10 words poem on my prompt",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = openrouter_model.invoke(messages)
# print(ai_msg.content)
#  llm = ChatOpenAI(model="", temperature=0)
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )
    
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to medical guidance for infants, kids and also pregnant mothers.  
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
print(
    question_router.invoke(
        {"question": "Who will the Bears draft first in the NFL draft?"}
    )
)
print(question_router.invoke({"question": "What are the types of agent memory?"}))

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )



structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question even not perfectly but still related to question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)
# include the generation variable that is the answer from both rag and web
answer_grader = answer_prompt | structured_llm_grader

system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    try:

        if "llm_instance" not in kwargs:

            llm_instance = OpenRouter(
                model="google/gemma-3-12b-it:free",  
                api_key=OPENAI_API_KEY,
                temperature=0.7,
            )
            kwargs["llm_instance"] = llm_instance
            
        print(f"Calling LLM with prompt length: {len(prompt)}")

        try:
            response = await llama_index_complete_if_cache(
                kwargs["llm_instance"],
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )

            if response is None:
                print("Warning: Received None response from LLM")
                return "Error: LLM returned None"
                
            return response
            
        except Exception as e:
            print(f"Error in LLM call: {str(e)}")
            return "Error in LLM processing. Please check OpenRouter configuration."
            
    except Exception as e:
        print(f"LLM setup failed: {str(e)}")
        return "Error in LLM setup"



async def embedding_func(texts):
    try:
        embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,   
            max_length=512,
            trust_remote_code=True  
        )
        return await llama_index_embed(texts, embed_model=embed_model)
    except Exception as e:
        print(f"Embedding failed: {str(e)}")
        raise


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    print(f"embedding_dim={embedding_dim}")
    return embedding_dim


async def initialize_rag():

    embedding_dimension = await get_embedding_dim()

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        graph_storage="Neo4JStorage",
        vector_storage="MongoVectorDBStorage",
        kv_storage="MongoKVStorage",
        log_level="INFO",
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        vector_db_storage_cls_kwargs={"cosine_better_than_threshold": 0.7},
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
            func=embedding_func,
        ),
    )
    print(rag)
    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

def main(question):
    rag = asyncio.run(initialize_rag())
    
    token_tracker.reset()
    # Insert Einstein data and verify
    # try:
    #     print("Inserting Einstein data...")
    #     # insertion_result = rag.insert(strings_list)
    #     # print(f"Einstein data insertion completed: {insertion_result}")
    # except Exception as e:
    #     print(f"Error during Einstein data insertion: {e}")
    print("Token usage:", token_tracker.get_usage())
    return{"answer":rag.query(question, param=QueryParam(mode="hybrid"))}

    




from langchain_core.messages import HumanMessage,AIMessage

def get_latest_question(agent_state:AgentState):
    for message in reversed(agent_state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content
    return None

def get_latest_answer(agent_state:AgentState):
    for message in reversed(agent_state["messages"]):
        if isinstance(message, AIMessage):
            return message.content
    return None
            
def create_web_search_rag_prompt():
    """
    Creates a RAG prompt template specifically designed for web search responses.
    
    This prompt helps the model:
    1. Synthesize information from retrieved web content
    2. Stay focused on the user's question
    3. Acknowledge the source (web search)
    4. Structure information in a clear, helpful way
    
    Returns:
        ChatPromptTemplate: A formatted prompt template for web search RAG
    """
    system_template = """You are a helpful assistant that provides accurate information based on web search results.
    
    Your task is to answer the user's question using ONLY the information provided in the web search results below.
    
    Guidelines:
    - Synthesize the information from the search results to provide a comprehensive answer
    - Stay strictly focused on the user's question
    - If the search results don't contain enough information to answer the question fully, acknowledge this limitation
    - Do not make up information that isn't present in the search results
    - Use a clear structure with paragraphs and bullet points where appropriate
    - If search results contain conflicting information, acknowledge the different perspectives
    - If information is time-sensitive, note when the search was conducted
    - Cite specific facts or statistics from the search results where possible
    
    Remember: Quality over quantity. A concise, accurate answer is better than a lengthy, speculative one.
    
    Web search results:
    {context}
    """
    
    human_template = """Question: {question}"""
    
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])


def web_search(state):
    """
    Web search based on the user's question, retrieves information and generates
    a comprehensive answer using a RAG (Retrieval-Augmented Generation) approach.
    
    This function:
    1. Retrieves the user's question from the state
    2. Performs a web search using the Tavily search tool
    3. Formats the search results for use in a RAG prompt
    4. Generates a response using the LLM and the RAG prompt
    5. Updates the state with the generated response
    
    Args:
        state (AgentState): The current state of the agent conversation
        
    Returns:
        AgentState: Updated state with the generated response added to messages
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting web search process")
    
    # Get the user's question from the state
    question = get_latest_question(state)
    print(question)
    if not question:
        logger.warning("No question found in state")
        return {"messages": state["messages"] + [AIMessage(content="I couldn't find a question to answer.")]}
    
    logger.info(f"Processing question: {question}")
    
    try:
        # Create the RAG prompt
        prompt = create_web_search_rag_prompt()
        
        # Initialize the Tavily search tool if not already initialized globally
        from langchain_community.tools.tavily_search import TavilySearchResults
        tavily_search = TavilySearchResults(max_results=3, include_answer=True)
        
        # Perform the web search
        logger.info(f"Performing web search for: {question}")
        search_results = tavily_search.invoke({"query": question})
        
        # Check if we got any results
        if not search_results or len(search_results) == 0:
            logger.warning("No search results found")
            return {"messages": state["messages"] + [AIMessage(content="I searched the web but couldn't find relevant information on this topic.")]}
        
        # Format the search results for the RAG prompt
        web_results_text = "\n\n".join([f"Source {i+1}:\n{result['content']}" for i, result in enumerate(search_results)])
        web_results_doc = Document(page_content=web_results_text)
        
        logger.info("Creating RAG chain")
        # Create the RAG chain
        rag_chain = prompt | llm | StrOutputParser()
        
        # Generate the response
        logger.info("Generating response")
        generation = rag_chain.invoke({
            "context": web_results_doc.page_content,
            "question": question
        })
        
        # Add citation if available from Tavily
        citation_text = ""
        for result in search_results:
            if "metadata" in result and "source" in result["metadata"]:
                citation_text += f"\n\nSource: {result['metadata']['source']}"
        
        if citation_text:
            generation += "\n\n---" + citation_text
        
        logger.info("Web search process completed successfully")
        state["messages"] = state["messages"] + [AIMessage(content=generation)]
        return state 
        
    except Exception as e:
        logger.error(f"Error in web search: {str(e)}", exc_info=True)
        error_message = "I encountered an error while searching the web. Please try asking your question differently."
        return {"messages": state["messages"] + [AIMessage(content=error_message)]}
    
    
def generate(state:AgentState):
    question = get_latest_question(state)
    answer = main(question)
    state["messages"] = state["messages"] + [AIMessage(content=answer["answer"])]
    return state
    
from langchain.schema import HumanMessage

def transform_query(state: AgentState):
    print("---TRANSFORM QUERY---")
    question = get_latest_question(state)
    
    # Re-write the question
    better_question = question_rewriter.invoke({"question": question})

    # Wrap the rewritten question in a HumanMessage (or SystemMessage, depending on your design)
    rewritten_msg = HumanMessage(content=better_question)

    # Return only the updated messages (LangGraph will auto-merge due to ⁠ operator.add ⁠)
    return {"messages": [rewritten_msg]}


def grade_answer(state: AgentState):
    question = get_latest_question(state)
    generation_dict = get_latest_answer(state)
    
    # Safely extract the actual answer string
    generation = generation_dict.get("answer", "") if isinstance(generation_dict, dict) else generation_dict
    
    print("Grading Answer:", generation)
    
    ans = answer_grader.invoke({"question": question, "generation": generation})
    print(ans.binary_score)
    
    return ans

def route_question(state:AgentState):
    question = get_latest_question(state)
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    
def decide_to_generate(state:AgentState):
    question = get_latest_question(state)
    answer = get_latest_answer(state)
    if answer is None:
        return "transform_query"
    
    grade_result = answer_grader.invoke({"question": question, "generation": answer})
    print(grade_result.binary_score)
    if grade_result.binary_score == "yes":
        return "useful"
    else:
        return "web_search"


workflow = StateGraph(AgentState)

workflow.add_node("web_search",web_search)
workflow.add_node("generate",generate)
workflow.add_node("transform_query",transform_query)
workflow.add_node("grade_answer",grade_answer)
workflow.add_conditional_edges(START,route_question,{
    "web_search": "web_search",
    "vectorstore":"generate"
},
                               )
workflow.add_edge("transform_query","generate")
workflow.add_edge("generate","grade_answer")
workflow.add_edge("web_search",END)
workflow.add_conditional_edges("grade_answer",
                               decide_to_generate,{
                                   "useful":END,
                                   "web_search":"web_search"
                               },)
app=workflow.compile()

from pprint import pprint

state = {
    "messages": [HumanMessage(content="my baby is doing vomit from past 3 days what should I do?")]
}

for output in app.stream(state):
    for key, value in output.items():
        pprint(f"Node '{key}':⁠ {value} ⁠")
    pprint("\n---\n") 