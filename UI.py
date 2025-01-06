import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F 

class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
    super(LSTM, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    # LSTM
    self.lstm = nn.LSTM(
        input_size,
        hidden_size,
        num_layers,
        dropout=dropout,
        batch_first=True
    )

    self.classifier = nn.Sequential(
        nn.Linear(hidden_size, hidden_size // 2),  
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size // 2, hidden_size//4),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size//4, output_size)
    )

  def forward(self, x):
    if len(x.shape) == 2:
      x = x.unsqueeze(1)  # Agrega dimensión de secuencia (batch_size, 1, embedding_dim)

    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

    # LSTM output: output of all the hidden_state, hidde_state, cell_state
    lstm_out, (hidden, _) = self.lstm(x, (h0,c0))
    # out: batch_size, seq_length, hidden_state

    # Classification layers
    output = self.classifier(lstm_out[:, -1, :])
    return output


# Función para cargar el modelo guardado
def cargar_modelo():
    embedding_dim = 768  # Embedding size
    hidden_dim = 256
    output_dim = 6  # Number of emotion or classes
    num_layers = 2
    dropout = 0.45

    # Model
    model = LSTM(embedding_dim, hidden_dim, num_layers, output_dim, dropout)
    model.load_state_dict(torch.load('model/modelo_emociones.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocesamiento y vectorización del texto
def procesar_texto(text):
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings

# Función para predecir la emoción del texto
def predecir_emocion(modelo, label_mapping):
    text = entry_texto.get("1.0", tk.END).strip()  # Obtener el texto del widget Text
    
    if not text:
        messagebox.showwarning("Entrada vacía", "Por favor, ingrese un texto para predecir la emoción.")
        return

    embedding = procesar_texto(text)
    
    with torch.no_grad():
        outputs = modelo(embedding)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_label = label_mapping[predicted_class]
        
        # Mostrar el resultado
        label_resultado.config(text=f"Emoción Predicha: {predicted_label}")

# Configuración principal de la interfaz gráfica
def main():
    global entry_texto, label_resultado

    # Mapeo de etiquetas
    label_mapping = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }

    # Cargar el modelo
    modelo = cargar_modelo()

    # Crear ventana principal
    ventana = tk.Tk()
    ventana.title("Predicción de Emoción en Texto")

    # Crear widgets
    label_instrucciones = tk.Label(ventana, text="Ingresa un texto para predecir la emoción:")
    label_instrucciones.pack(pady=10)

    entry_texto = tk.Text(ventana, height=10, width=50)
    entry_texto.pack(pady=10)

    boton_predecir = tk.Button(
        ventana,
        text="Predecir Emoción",
        command=lambda: predecir_emocion(modelo, label_mapping)
    )
    boton_predecir.pack(pady=10)

    label_resultado = tk.Label(ventana, text="Emoción Predicha: ", font=("Arial", 14))
    label_resultado.pack(pady=10)

    # Iniciar la interfaz gráfica
    ventana.mainloop()

if __name__ == "__main__":
    main()
