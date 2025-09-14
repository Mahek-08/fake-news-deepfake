from flask import Flask, request, render_template
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/check_text", methods=["POST"])
def check_text():
    text = request.form["news_text"]
    # TODO: load your fake news model and predict
    result = "FAKE"  # placeholder
    return f"<h2>Prediction: {result}</h2><br><a href='/'>Go Back</a>"

@app.route("/check_media", methods=["POST"])
def check_media():
    file = request.files["file"]
    if not file:
        return "No file uploaded!"
    
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    # TODO: Add your deepfake/image model inference here
    result = "REAL"  # placeholder
    return f"<h2>Prediction: {result}</h2><br><a href='/'>Go Back</a>"

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
