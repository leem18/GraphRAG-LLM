import os
import sys
from requests import HTTPError
from langgraph.graph import END, StateGraph, START
from typing import List
from typing_extensions import TypedDict
import warnings
import datetime
import json
from lib.web_search_tool import WebSearchTool
from lib.llm_setup import LLMSetup

# Create Langsmith client to interact with API
from langsmith import Client
client = Client()


warnings.filterwarnings('ignore')
sys.setrecursionlimit(100000)

now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Hyper parameter setting
log_file_name = os.getenv("LOG_FILE_NAME")
original_question = os.getenv("QUESTION")
search_tool_name = os.getenv("SEARCH_TOOL_NAME")
target_period = os.getenv("TARGET_PERIOD")
target_area = os.getenv("TARGET_AREA")
target_topic = os.getenv("TARGET_TOPIC")

retrieval_grader_model_name = os.getenv("RETRIEVAL_GRADER_MODEL_NAME")
retrieval_grader_temperature = int(os.getenv("RETRIEVAL_GRADER_TEMPERATURE"))
retrieval_grader_prompt_name = os.getenv("RETRIEVAL_GRADER_PROMPT_NAME")

hypothetical_generator_model_name = os.getenv("HYPOTHETICAL_GENERATOR_MODEL_NAME")
hypothetical_generator_temperature = int(os.getenv("HYPOTHETICAL_GENERATOR_TEMPERATURE"))
hypothetical_generator_prompt_name = os.getenv("HYPOTHETICAL_GENERATOR_PROMPT_NAME")


generator_model_name = os.getenv("GENERATOR_MODEL_NAME")
generator_temperature = int(os.getenv("GENERATOR_TEMPERATURE"))
generator_prompt_name = os.getenv("GENERATOR_PROMPT_NAME")

question_rewriter_model_name = os.getenv("QUESTION_REWRITER_MODEL_NAME")
question_rewriter_temperature = float(os.getenv("QUESTION_REWRITER_TEMPERATURE"))
question_rewriter_prompt_name = os.getenv("QUESTION_REWRITER_PROMPT_NAME")

search_max_depth = int(os.getenv("MAX_DEPTH"))
max_loop_cnt = int(os.getenv("MAX_LOOP_CNT"))

HYPO = ""

# info logging
def log_info():
    log_dict = {}
    retrieval_grader = {
        'model_name' : retrieval_grader_model_name,
        'temperature' : retrieval_grader_temperature
    }

    generator = {
        'model_name' : generator_model_name,
        'temperature' : generator_temperature
    }
    question_rewriter = {
        'model_name' : question_rewriter_model_name,
        'temperature' : question_rewriter_temperature
    }
    log_dict['retrieval_grader'] = retrieval_grader
    log_dict['generator'] = generator
    log_dict['question_rewriter'] = question_rewriter
    log_dict['time'] = now_time
    log_dict['search_max_depth'] = search_max_depth
    log_dict['retrieval_urls'] = LLMSetup().retrieval_urls
    log_dict['iteration'] = []
    log_file = os.path.join('./log', f'log_{now_time}.txt')
    return  log_dict, log_file

log_dict, log_file = log_info()

llm_setup = LLMSetup()

retriever = llm_setup.set_retriever(urls=llm_setup.retrieval_urls)

# Retrieval Grader
retrieval_grader = llm_setup.set_retrieval_grader(
    model_name=retrieval_grader_model_name,
    temperature=retrieval_grader_temperature,
    prompt_template=retrieval_grader_prompt_name
)
# Hypothetical Generator
hypothetical_generator = llm_setup.set_hypothetical_generator(
    model_name=hypothetical_generator_model_name,
     temperature=hypothetical_generator_temperature,
     prompt_template=hypothetical_generator_prompt_name
)

# generate rag chain
rag_chain = llm_setup.set_generate_rag_chain(
    model_name=generator_model_name,
    temperature=generator_temperature,
    prompt_template=generator_prompt_name
)

# question rewriter
question_rewriter = llm_setup.set_question_rewriter(
    model_name=question_rewriter_model_name,
    temperature=question_rewriter_temperature,
    prompt_template=question_rewriter_prompt_name
)

web_search_tool = WebSearchTool(engine=search_tool_name, max_results=search_max_depth)

### Setting graph state
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        hypothetical: hypothetical doc to original question
        reason: reasons for document relevancies.
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]
    hypothetical: str
    reason: str


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """    
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.get_relevant_documents(question)
    print(documents)
    print("type", type)
    filtered_docs = []
    
    return {"documents": filtered_docs, "question": question}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    print("----------------------------")
    print("question", question)
    print("documents", documents)
        

    # RAG generation
    generation = rag_chain.invoke({"documents": documents, "question": question, "TARGET_PERIOD": target_period, "TARGET_AREA": target_area, "TARGET_TOPIC": target_topic})
    global log_dict
    # record the log for the result
    log_dict['result'] = {
        'question': question,
        'answer' : generation,
        'documents' : [doc['content'] for doc in documents]  # Document 객체를 문자열로 변환
    }

    return {"documents": documents, "question": question, "generation": generation}

def generate_hypothetical(state):
    """
    Generates a hypothetical document that provides context to the
    original query (SCQ)

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates hypothetical key with the hypothetical document
    """
    print("---ORIGINAL QUERY---")
    question = state["question"]
    print(question)
    print('')

    print("---GENERATE HYPOTHETICAL---")
    hypothetical_document = hypothetical_generator.invoke(
        {"question": question}
    )
    print('HYPOTHETICAL DOCUMENT:', hypothetical_document)
    return {"hypothetical": hypothetical_document}

def grade_documents(state):
    global original_question
    """
    Determines whether the retrieved documents are relevant to the 
    hypothetical document

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
                      and reasons for document relevancy
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    hypothetical_document = state["hypothetical"]

    # Score each doc
    filtered_docs = []
    web_search = "Yes"
    why_str = ""

    for d in documents:
        result = retrieval_grader.invoke(
            {"hypothetical_document": hypothetical_document, "document": d['content']}
        )
        grade = result.binary_score
        # score = result.cosine_similarity
        why = result.reason
        # print("---COSINE SIMILARITY SCORE---")
        # print("cos similarity score:", score)
        if grade == "Yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
            web_search = "No"
            break
        else:
            why_str += why
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    return {"documents": filtered_docs, "question": question, "web_search": web_search, "reason": why_str}

def transform_query(state):
    """
    Transform the query to produce a better question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    reason = state["reason"]

    print("prev Question: " + question)

    # Re-write question
    better_question = question_rewriter.invoke(
        {"reason": reason, "question": question}
        )
    print("better Question: " + better_question)
    return {"documents": documents, "question": better_question}

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    global loop_cnt
    loop_cnt += 1

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    try:
        docs = web_search_tool.run(question)
        # print(docs)
        # print("type(docs)", type(docs))
        # print("type(docs)", type(docs[0x]))
        
        for d in docs:
            # documents.append(Document(page_content=d['content']))
            documents.append(d)

    except HTTPError as http_err:
        print("#############################")
        print(f"HTTP error occurred: {http_err}")
        print("#############################")
        return {"documents": documents, "question": question}
    except Exception as err:
        print("#############################")
        print(f"An error occurred: {err}")
        print("#############################")
        return {"documents": documents, "question": question}

    global log_dict
    log_dict['iteration'].append({
        'iteration_num' : loop_cnt,
        'question': question,
        'documents' : docs,  # Document 객체를 문자열로 변환
    })
    return {"documents": documents, "question": question}

# conditional edges Edges
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

### Setting the workflow and nodes
workflow = StateGraph(GraphState)

workflow.add_node("web_search_node", web_search)  # web search
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate'
#workflow.add_node("reason_summary", reason_summary)  
workflow.add_node("hypothetical_generator", generate_hypothetical) # generate hypothetical
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("retrieve", retrieve)  # retrieve

### Building the graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "hypothetical_generator")
workflow.add_edge("hypothetical_generator", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
#workflow.add_edge("reason_summary", "transform_query")
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "grade_documents")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

loop_cnt = 0

# Run
inputs = {"question": original_question}
for output in app.stream(inputs, {"recursioƒn_limit": 10000}):
    for key, value in output.items():
        if loop_cnt >= max_loop_cnt:
            print(f"Loop ended. Failed to get proper data after {max_loop_cnt} times of web search")
            with open("./log/" + log_file_name + "_"+ now_time + '.json', 'w') as json_file:
                json.dump(log_dict, json_file, indent=4, ensure_ascii=False)
            exit(0)

# Final generation
with open("./log/" + log_file_name + "_"+ now_time + '.json', 'w') as json_file:
    json.dump(log_dict, json_file, indent=4, ensure_ascii=False)