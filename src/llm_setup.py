# lib/llm_setup.py

import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
import os


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


class LLMSetup:
    def __init__(self):
        self.retrieval_urls = load_json("data/urls/retrieval_urls.json").get('urls', [])

    def set_retriever(self, urls):
        docs = [WebBaseLoader(url, verify_ssl=False).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Add to vectorDB
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=OpenAIEmbeddings(),
        )
        retriever = vectorstore.as_retriever()
        return retriever

    def set_retrieval_grader(self, model_name='gpt-4o', temperature=0, prompt_template=''):
        # Data model
        class GradeDocuments(BaseModel):
            """Binary score for relevance check on retrieved documents."""
            binary_score: str = Field(
                description="Documents are relevant to the question, 'yes' or 'no'"
            )
            reason: str = Field(
                description="Why documents are relevant/not relevant to the question"
            )
        # LLM with function call
        grader_llm = ChatOpenAI(model=model_name, temperature=temperature)
        structured_llm_grader = grader_llm.with_structured_output(GradeDocuments)

        grade_prompt = hub.pull(prompt_template)
        retrieval_grader = grade_prompt | structured_llm_grader
        return retrieval_grader

    def set_generate_rag_chain(self, model_name='gpt-4o', temperature=0, prompt_template=''):
        # Prompt
        # LLM
        generate_llm = ChatOpenAI(model_name=model_name, temperature=temperature)

        generate_prompt = hub.pull(prompt_template)
        # Chain
        rag_chain = generate_prompt | generate_llm | StrOutputParser()
        return rag_chain

    def set_question_rewriter(self, model_name='gpt-4o', temperature=0, prompt_template=''):
        ### Question Re-writer
        # LLM
        question_rewriter_llm = ChatOpenAI(model=model_name, temperature=temperature)

        re_write_prompt = hub.pull(prompt_template)
        question_rewriter = re_write_prompt | question_rewriter_llm | StrOutputParser()
        return question_rewriter

    def set_hypothetical_generator(self, model_name='gpt-4o', temperature=0, prompt_template=''):
        # (based off of HyDE)
        hypothetical_generator_llm = ChatOpenAI(model=model_name, temperature=temperature)

        hypo_prompt = hub.pull(prompt_template)
        hypothetical_document = hypo_prompt | hypothetical_generator_llm | StrOutputParser()
        return hypothetical_document
