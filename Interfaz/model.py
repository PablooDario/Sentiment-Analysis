# model.py
import torch

class EmotionModel:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()  # Establecer el modelo en modo evaluación

    def predict(self, text):
        # Aquí deberías procesar el texto, hacer la tokenización y convertirlo en un tensor
        # Este es solo un ejemplo general. Usa el preprocesamiento que corresponda a tu modelo.
        input_tensor = self.text_to_tensor(text)
        with torch.no_grad():
            output = self.model(input_tensor)
        predicted_class = output.argmax(dim=1).item()  # Suponiendo que el modelo devuelve logits
        return predicted_class

    def text_to_tensor(self, text):
        # Este método debe convertir el texto en un tensor
        # Dependerá de cómo esté entrenado tu modelo, por ejemplo, usando un Tokenizer de HuggingFace, etc.
        pass