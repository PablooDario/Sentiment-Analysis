import tkinter as tk
from tkinter import messagebox
import torch
from torch import nn
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Cargar el modelo de PyTorch guardado
def cargar_modelo():
    # Cargar el modelo guardado
    modelo = torch.load('modelo_emociones.pth')  # Cambia el nombre de archivo a tu archivo .pth
    modelo.eval()  # Poner el modelo en modo de evaluación
    return modelo

# Preprocesamiento y vectorización del texto
def procesar_texto(texto):
    # Asegúrate de usar el mismo vectorizador que usaste para entrenar el modelo
    vectorizador = CountVectorizer()  # Este debe coincidir con el usado al entrenar el modelo
    texto_vectorizado = vectorizador.transform([texto])
    return torch.tensor(texto_vectorizado.toarray(), dtype=torch.float32)

# Función para predecir la emoción del texto
def predecir_emocion():
    texto = entry_texto.get("1.0", tk.END).strip()  # Obtener el texto del widget Text
    if len(texto) == 0:
        messagebox.showwarning("Entrada vacía", "Por favor, ingrese un texto para predecir la emoción.")
        return

    # Procesar el texto para convertirlo en características que el modelo pueda entender
    texto_tensor = procesar_texto(texto)

    # Cargar el modelo
    modelo = cargar_modelo()

    # Realizar la predicción
    with torch.no_grad():  # Desactivar el cálculo de gradientes
        prediccion = modelo(texto_tensor)  # Pasar el tensor al modelo
        emocion = torch.argmax(prediccion).item()  # Obtener la clase con mayor puntuación

    # Mostrar el resultado de la predicción
    label_resultado.config(text=f"Emoción Predicha: {emocion}")

# Crear la interfaz gráfica
ventana = tk.Tk()
ventana.title("Predicción de Emoción en Texto")

# Crear widgets de la interfaz
label_instrucciones = tk.Label(ventana, text="Ingresa un texto para predecir la emoción:")
label_instrucciones.pack(pady=10)

entry_texto = tk.Text(ventana, height=10, width=50)
entry_texto.pack(pady=10)

boton_predecir = tk.Button(ventana, text="Predecir Emoción", command=predecir_emocion)
boton_predecir.pack(pady=10)

label_resultado = tk.Label(ventana, text="Emoción Predicha: ", font=("Arial", 14))
label_resultado.pack(pady=10)

# Iniciar la interfaz gráfica
ventana.mainloop()