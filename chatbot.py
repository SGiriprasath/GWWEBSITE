from flask import Flask, request, jsonify, render_template
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import GooglePalmEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA



import os

app = Flask(__name__, template_folder='templates', static_folder='static')

google = GooglePalm(google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.0)
embeddings_new = GooglePalmEmbeddings(google_api_key=os.getenv("GOOGLE_API_KEY"))
file_path = "faiss_store_openai.index"


def create_vector_data():
    if os.path.exists('faiss_store_openai'):
        pass
    else:
        data = CSVLoader(file_path='company.csv', source_column="prompt")
        new_data = data.load()

        text_splitter = RecursiveCharacterTextSplitter()
        docs = text_splitter.split_documents(new_data)
        # create embeddings and save it to FAISS index
        vector_data = FAISS.from_documents(docs, embeddings_new)
        vector_data.save_local(file_path)


def get_chain(query):
    data = FAISS.load_local(file_path, embeddings=embeddings_new)
    retriever = data.as_retriever(score_threshold=0.1)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly give the message "I don't know." rather than making a answer this is important.
    CONTEXT: {context}
    QUESTION: {question}"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}
    chain = RetrievalQA.from_chain_type(llm=google,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs=chain_type_kwargs)
    print(chain({"query": query})["result"])
    if len(query) == 1:
        return "I don't know"
    elif "I don't know" in chain({"query":query})["result"]:
        return "I don't know"
    elif "your response:" in chain({"query": query})["result"] or "response:" in chain({"query": query})["result"]:
        return chain({"query": query})["result"].split("response")[-1]
    else:
        return chain({"query": query})["result"]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    response = get_chain(user_input)
    return jsonify({"response": response})


if __name__ == '__main__':
    create_vector_data()
    app.run(debug=True)
