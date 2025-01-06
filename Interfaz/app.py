# app.py
from flask import Flask, render_template, request
from model import EmotionModel

app = Flask(__name__)

# Cargar el modelo (ajusta la ruta al archivo de tu modelo)
model = EmotionModel('path/to/your/model.pth')

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        text = request.form["text"]
        if len(text.split()) <= 300:  # Validación de las palabras
            prediction = model.predict(text)
        else:
            prediction = "Texto demasiado largo. Ingresa 300 palabras o menos."
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)