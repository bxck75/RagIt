import os ,re
script_dir = os.path.dirname(os.path.abspath(__file__))
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from typing import Union
import warnings
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from typing import Any, Iterator, List, Optional
from huggingface_hub import login
from langchain.llms import HuggingFaceHub
from elevenlabs import set_api_key
from tempfile import TemporaryDirectory
from langchain_community.tools.eleven_labs.text2speech import ElevenLabsText2SpeechTool
from langchain.utilities.serpapi import SerpAPIWrapper
from langchain.embeddings import HuggingFaceEmbeddings

warnings.filterwarnings('ignore')

from credits import (
    HUGGINGFACE_TOKEN,
    HUGGINGFACE_TOKEN as HUGGINGFACEHUB_API_TOKEN,
    HUGGINGFACE_EMAIL,
    HUGGINGFACE_PASS,
    OPENAI_API_KEY,
    ELEVENLABS_API_KEY,
    SERPAPI_API_KEY)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
os.environ["HUGGINGFACE_TOKEN"] = HUGGINGFACE_TOKEN
os.environ["HUGGINGFACE_EMAIL"] = HUGGINGFACE_EMAIL
os.environ["HUGGINGFACE_PASS"] = HUGGINGFACE_PASS
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ELEVEN_API_KEY"] = ELEVENLABS_API_KEY


from huggingface_hub import login
from langchain.llms import HuggingFaceHub

class HuggingHelper:
    def __init__(self,debug=False):
        self.debug = debug
    
    
        set_api_key(ELEVENLABS_API_KEY)
        self.tts = ElevenLabsText2SpeechTool()
        self.serp_search = SerpAPIWrapper()

        self.repo_list = [
            {"user": "tiiuae", "model": "falcon-7b-instruct"},
            {"user": "mistralai", "model": "Mistral-7B-v0.1"},
            {"user": "openchat", "model": "openchat_3.5"},
            {"user": "01-ai", "model": "Yi-34B"},
            {"user": "codellama", "model": "CodeLlama-7b-Python-hf"},
        ]
        self.repo_ids = []  # Initialize an empty list to store repo IDs
        for i, repo in enumerate(self.repo_list):
            repo_id = f"{repo['user']}/{repo['model']}"
            self.repo_ids.append(repo_id)

        self.embeddings= HuggingFaceEmbeddings(
                                    model_name="all-MiniLM-L6-v2",
                                    model_kwargs = {'device': 'cpu'},
                                    encode_kwargs = {'normalize_embeddings': True}
                                )

        self.llms = self.create_llms(self.repo_ids, self.embeddings)
        self.openllm = self.create_openllm("openchat/openchat_3.5", self.embeddings)
        self.best_llm = self.create_best_llm("tiiuae/falcon-7b-instruct", self.embeddings)
        self.working_directory = TemporaryDirectory()
        self.data_dir=os.path.join(script_dir,"data")
        self.img_dir=os.path.join(script_dir,"images")
        
    def login_hub(self,api_key, token_dir="/home/codemonkeyxl/.cache/huggingface/token"):
        if os.path.exists(token_dir):
            self.newsession_bool = False
            self.write_permission_bool = False
        else:
            self.newsession_bool = True
            self.write_permission_bool = False
        try:
            login(api_key, new_session= self.newsession_bool, write_permission= self.write_permission_bool )


    def list_hub_tasks(self):
        ''' get list of possible tasks for models  on the hugging face model hub'''
        from langchain.llms import HuggingFaceHub

        hugging_face_hub = HuggingFaceHub()
        tasks = hugging_face_hub.list_tasks()
        if self.debug:
            for task in tasks:
                print(task["id"])

        self.model_tasks = tasks

    def llm_fetch(self, prompt):
        # Track messages separately
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": prompt}
        ]
        # Update streamlit messages manually
        st.session_state.messages.extend(messages)
        for msg in messages:
            if msg["role"] == "system":
                st.chat_message("system").write(msg["content"])
            elif msg["role"] == "user":
                st.chat_message("user").write(msg["content"])

        response = self.best_llm(prompt=" ".join([f"{msg['role']}: {msg['content']}" for msg in messages]))
        return f"{response['text']}"


    def create_llms(self, repo_ids: List[str]) -> List[HuggingFaceHub]:
        llms = []
        for repo_id in repo_ids:
            llm = HuggingFaceHub(repo_id=repo_id, task="text-generation", model_kwargs={"min_length": 32, "max_length": 1000, "temperature": 0.1, "max_new_tokens": 1024, "num_return_sequences": 1})
            llms.append(llm)
        return llms


    def create_openllm(self, repo_id: str, embeddings: HuggingFaceEmbeddings) -> HuggingFaceHub:
        openllm = HuggingFaceHub(repo_id=repo_id, task="text-generation", model_kwargs={"min_length": 16, "max_length": 1000, "temperature": 0.1, "max_new_tokens": 512, "num_return_sequences": 1})
        return openllm


    def create_best_llm(self, repo_id: str, embeddings: HuggingFaceEmbeddings) -> HuggingFaceHub:
        best_llm = HuggingFaceHub(repo_id=repo_id, task="text-generation", model_kwargs={"min_length": 200, "max_length": 1000, "temperature": 0.1, "max_new_tokens": 512, "num_return_sequences": 1})
        return best_llm