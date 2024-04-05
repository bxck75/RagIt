import os
import getpass

import faiss
import numpy as np
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_community.document_loaders import GithubLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from huggingface_hub import InferenceClient


class GithubFAISSRetriever:
    def __init__(self, repo_url, branch="main", commit_ish=None):
        self.repo_url = repo_url
        self.branch = branch
        self.commit_ish = commit_ish

        self.loader = GithubLoader(repo_url, branch, commit_ish)
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, add_start_index=True)


        self.vectorstore = None
        self.retriever = None
        self.retrieval_chain = None
        self.llm = None

    def create_embedder(self, model_name):
        self.embeddings = OpenAIEmbeddings()
        
    def create_llm(self, model_name):
        self.llm = HF_LLM(model_name)

    def create_vectorstore(self, output_path):
        documents = self.loader.load()
        documents = self.text_splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.vectorstore.save(output_path)

    def load_vectorstore(self, input_path):
        self.vectorstore = FAISS.load(input_path)

    def create_retriever(self):
        self.retriever = self.vectorstore.as_retriever()

    def create_retrieval_chain(self, retrieval_qa_chat_prompt):
        combine_docs_chain = create_stuff_documents_chain(
            self.llm, retrieval_qa_chat_prompt
        )
        self.retrieval_chain = create_retrieval_chain(self.retriever, combine_docs_chain)

    def chat(self, input):
        return self.retrieval_chain.invoke({"input": input})


class HF_LLM:
    def __init__(self, model_name: str):
        self.client = InferenceClient(model_name)

    def __call__(self, text: str) -> str:
        return self.chat_completion(messages=[{"role": "user", "content": text}])

    def chat_completion(self, messages: list, max_tokens: int = 200) -> str:
        return self.client.chat_completion(messages, max_tokens=max_tokens)["choices"][0]["message"]["content"]


if __name__ == "__main__":
    repo_url = "https://github.com/huggingface/transformers"
    branch = "main"
    commit_ish = None

    output_path = "transformers_vectorstore.faiss"

    github_faiss_retriever = GithubFAISSRetriever(repo_url, branch, commit_ish)
    github_faiss_retriever.create_vectorstore(output_path)
    github_faiss_retriever.load_vectorstore(output_path)
    github_faiss_retriever.create_retriever()

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    github_faiss_retriever.create_llm("HuggingFaceH4/zephyr-7b-beta")
    github_faiss_retriever.create_retrieval_chain(retrieval_qa_chat_prompt)

    input = "What is the difference between a transformer and a recurrent neural network?"
    response = github_faiss_retriever.chat(input)
    print(response)