import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
from src.common_responses import check_common_question
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

embeddings = download_hugging_face_embeddings()

# Initializing Pinecone
index_name = "medical-chatbotn"
# Loading the index
docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(model="model/llama-2-7b-chat.Q4_0.gguf", 
                    model_type="llama",
                    config={"max_new_tokens": 1024, 
                            "temperature": 0.8,
                            "context_length": 2048},
                    local_files_only=True)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input_msg = msg.strip()
    
    # Check for common questions first
    common_response = check_common_question(input_msg)
    if common_response:
        print("Common Response: ", common_response)
        return str(common_response)
    
    # If not a common question, use RAG + LLM
    result = qa.invoke({"query": input_msg})
    print("Response: ", result['result'])
    return str(result['result'])

if __name__ == "__main__":
    app.run(debug=True)