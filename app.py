from flask import Flask, render_template, request, jsonify
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

app = Flask(__name__)

# Initialize the data directory
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Settings control global defaults
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Groq(model="llama3-70b-8192", api_key="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")

try:
    # Create index when server starts
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    print("✅ Vector index created successfully")
except Exception as e:
    print(f"⚠️ Error creating index: {str(e)}")
    # Create empty index if no documents
    index = VectorStoreIndex([])
    query_engine = index.as_query_engine()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def process_query():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message.strip():
        return jsonify({'response': 'Please ask a question.'})
    
    try:
        # Get response from the query engine
        response = query_engine.query(user_message)
        return jsonify({'response': str(response)})
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return jsonify({'response': f"Sorry, I encountered an error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)