from flask import Flask, requests, jsonify, render_template
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import GooglePalmEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
import os
from datetime import datetime


app = Flask(__name__, template_folder='templates', static_folder='static')

google = GooglePalm(google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)

path = "faiss_store_openai.index"
instructor_embeddings = GooglePalmEmbeddings()


def create_vector_data():
    if os.path.exists('faiss_store_openai.index'):
        pass
    else:
        data = CSVLoader(file_path='company.csv', source_column="prompt")
        new_data = data.load()

        # Create a FAISS instance for vector database from 'data' and embeddings
        vectordb = FAISS.from_documents(documents=new_data,
                                        embedding=instructor_embeddings)

        # Save the FAISS instance locally
        vectordb.save_local(path)


def get_chain(query):
    # Load the FAISS instance with embeddings for consistency
    data = FAISS.load_local(path, embeddings=instructor_embeddings)
    retriever = data.as_retriever(score_threshold=0.7)

    prompt_template =  """*Instruction:* Answer the following question based on the provided context only. Use the information from the "response" section in the context whenever possible, but avoid making significant changes.

*Context:* {context}

*Question:* {question}

*Answer:* I don't know."  # Pre-fill the answer with "I don't know"
"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}
    chain = RetrievalQA.from_chain_type(llm=google,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs=chain_type_kwargs)

    # Call the chain with the query and return the result
    return chain(query)["result"]
    

API_KEY = 'f0bd38ac5bb3c8'

def get_location(ip):
    url = f'http://ipinfo.io/{ip}?token={API_KEY}'
    response = requests.get(url)
    return response.json()

@app.route('/')
def index():
    
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    if client_ip:
        client_ip = client_ip.split(',')[0]
    
   
    location_data = get_location(client_ip)
    
    city = location_data.get('city', 'Unknown')
    country = location_data.get('country', 'Unknown')
    ip = location_data.get('ip', 'Unknown')
    org = location_data.get('org', 'Unknown')
    
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    response = get_chain(user_input)
    return jsonify({"response": response})


if __name__ == '_main_':
    create_vector_data()
    app.run(debug=True)
