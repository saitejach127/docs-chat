from flask import Flask, request, render_template
app = Flask(__name__, template_folder="templates")
from DocChat import DocChat
import os

docChat = DocChat("embeddings/budget_embeddings.csv", 'Use the below Budget 2023-24 speech document to answer the subsequent question. If the answer cannot be found in the article, write "I could not find an answer."', "You answer questions about Budget Speech 2023-24., ",'Budget Doc :', APIKEY=os.environ["OPENAI_API_KEY"])
 

@app.route("/", methods=["GET"])
def show_ui():
    return render_template("index.html")

@app.route("/api", methods=["POST"])
def get_answer():
    question = request.json.get("question")
    return {
        "success": True,
        "message": docChat.ask(query=question)
    }