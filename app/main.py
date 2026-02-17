from flask import Flask, request, jsonify
from db.get_db import get_database
from routes.regression import reg
from routes.predict_route import predict_bp

db = get_database()
db.connect()

app = Flask(__name__)

@app.route("/insert", methods=["POST"])
def insert():
    data = request.json
    doc_id = db.insert(data)
    return jsonify({"inserted_id": str(doc_id)})

@app.route("/find")
def find_all():
    results = db.find({})
    return jsonify({"results": [str(r) for r in results]})

@app.route("/")
def home():
    return f"Currently using: {type(db).__name__}"


app.register_blueprint(reg, url_prefix="/regression")
app.register_blueprint(predict_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)