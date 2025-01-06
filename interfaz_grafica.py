import gradio as gr
from joblib import dump
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel


def get_embedding(text):
    """
    Obtiene el embedding de un único texto usando un modelo y un tokenizer.

    Args:
        text (str): El texto a procesar.
        tokenizer: El tokenizer compatible con el modelo.
        model: El modelo que genera los embeddings.

    Returns:
        numpy.ndarray: El embedding del texto.
    """

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment") # obtiene un ID para cada palabra
    model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment") # "mapea los ID a embeddings"

    model.eval()  # Poner el modelo en modo evaluación

    # Tokenizar el texto
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # Obtener embeddings
    with torch.no_grad():
        outputs = model(**encoded)

    # Obtener el embedding del token [CLS]
    embedding = outputs.last_hidden_state[:, 0, :].numpy()

    return embedding

def predecir_emocion(texto):
    if len(texto) > 300:
        raise gr.Error("El texto ingresado es demasiado largo. Por favor, ingresa un texto de máximo 300 caracteres.")
    elif len(texto) == 0:
        raise gr.Error("No ha ingresado ningun texto. Por favor, ingresa un texto de máximo 300 caracteres.")
    elif len(texto) > 15:
        embedding = get_embedding(texto)
        return "Feliz"
    else:
        embedding = get_embedding(texto)
        return "Triste"



# Crear la interfaz de usuario
iface = gr.Interface(
    fn=predecir_emocion,
    inputs="text",
    outputs="text",
    title="Análisis de emociones",
    description="Introduce un texto menor a 300 caracteres."
)

iface.launch()