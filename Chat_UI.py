import streamlit as st
from github_faiss_retriever import GithubFAISSRetriever

# Instantiate GithubFAISSRetriever
repo_url = st.text_area("github repo url", placeholder="https://github.com/huggingface/transformers")

st.write("Using Hugging Face's transformers library.\n Example:({}) \n".format(repo_url))
branch = "main"
commit_ish = None
github_faiss_retriever = GithubFAISSRetriever(repo_url, branch, commit_ish)

# Load the pre-trained vectorstore
output_path = "transformers_vectorstore.faiss"
github_faiss_retriever.load_vectorstore(output_path)

# Set up Streamlit app
st.title("GitHub FAQ Chat")

# User input
user_input = st.text_input("You:", "")

if user_input:
    # Chat with the retriever
    response = github_faiss_retriever.chat(user_input)
    st.text_area("Response:", response)