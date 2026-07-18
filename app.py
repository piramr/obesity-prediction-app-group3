# =====================================
# APP WEB - PREDICCIÓN DE OBESIDAD
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

from tensorflow.keras.models import load_model

# ==============================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================

st.set_page_config(
    page_title="Predicción de Obesidad",
    page_icon="⚕️",
    layout="centered"
)

# ==============================
# CARGAR MODELO Y OBJETOS
# ==============================

@st.cache_resource
def cargar_modelo():
    modelo = load_model("modelos/rna_obesity_model.keras")
    return modelo

@st.cache_resource
def cargar_objetos():
    encoders = joblib.load("modelos/obesity_encoders.pkl")
    target_encoder = joblib.load("modelos/obesity_target_encoder.pkl")
    scaler = joblib.load("modelos/obesity_scaler.pkl")
    columnas = joblib.load("modelos/obesity_columnas.pkl")
    return encoders, target_encoder, scaler, columnas

modelo = cargar_modelo()
encoders, target_encoder, scaler, columnas = cargar_objetos()

# ==============================
# TRADUCCIÓN DE RESULTADOS
# ==============================

traduccion_resultados = {
    "Insufficient_Weight": "Peso insuficiente",
    "Normal_Weight": "Peso normal",
    "Overweight_Level_I": "Sobrepeso nivel I",
    "Overweight_Level_II": "Sobrepeso nivel II",
    "Obesity_Type_I": "Obesidad tipo I",
    "Obesity_Type_II": "Obesidad tipo II",
    "Obesity_Type_III": "Obesidad tipo III"
}

# ==============================
# TÍTULO
# ==============================

st.title("Predicción del Nivel de Obesidad")
st.write(
    "Esta aplicación utiliza una Red Neuronal Artificial entrenada con el dataset "
    "Obesity para predecir el nivel de obesidad de una persona a partir de sus datos "
    "físicos y hábitos de vida."
)

st.info("Modelo utilizado: RNA - Red Neuronal Artificial")

# ==============================
# DATOS DE EJEMPLO
# ==============================

ejemplo = {
    "Gender": "Male",
    "Age": 21.0,
    "Height": 1.72,
    "Weight": 80.0,
    "family_history_with_overweight": "yes",
    "FAVC": "yes",
    "FCVC": 2.0,
    "NCP": 3.0,
    "CAEC": "Sometimes",
    "SMOKE": "no",
    "CH2O": 2.0,
    "SCC": "no",
    "FAF": 1.0,
    "TUE": 1.0,
    "CALC": "Sometimes",
    "MTRANS": "Public_Transportation"
}

st.info("Los campos ya vienen cargados con datos de prueba. Puedes modificarlos y presionar Predecir.")

# ==============================
# FORMULARIO
# ==============================

st.subheader("Ingrese los datos de la persona")

with st.form("formulario_obesidad"):

    genero = st.selectbox(
        "Género",
        options=["Male", "Female"],
        format_func=lambda x: "Masculino" if x == "Male" else "Femenino",
        index=["Male", "Female"].index(ejemplo["Gender"])
    )

    edad = st.number_input(
        "Edad",
        min_value=1.0,
        max_value=100.0,
        value=float(ejemplo["Age"]),
        step=1.0
    )

    altura = st.number_input(
        "Altura en metros",
        min_value=1.0,
        max_value=2.5,
        value=float(ejemplo["Height"]),
        step=0.01
    )

    peso = st.number_input(
        "Peso en kilogramos",
        min_value=20.0,
        max_value=250.0,
        value=float(ejemplo["Weight"]),
        step=1.0
    )

    antecedentes = st.selectbox(
        "¿Tiene antecedentes familiares de sobrepeso?",
        options=["yes", "no"],
        format_func=lambda x: "Sí" if x == "yes" else "No",
        index=["yes", "no"].index(ejemplo["family_history_with_overweight"])
    )

    favc = st.selectbox(
        "¿Consume alimentos altos en calorías frecuentemente?",
        options=["yes", "no"],
        format_func=lambda x: "Sí" if x == "yes" else "No",
        index=["yes", "no"].index(ejemplo["FAVC"])
    )

    fcvc = st.slider(
        "Frecuencia de consumo de vegetales",
        min_value=1.0,
        max_value=3.0,
        value=float(ejemplo["FCVC"]),
        step=0.1
    )

    ncp = st.slider(
        "Número de comidas principales al día",
        min_value=1.0,
        max_value=4.0,
        value=float(ejemplo["NCP"]),
        step=0.1
    )

    caec = st.selectbox(
        "Consumo de comida entre comidas",
        options=["no", "Sometimes", "Frequently", "Always"],
        format_func=lambda x: {
            "no": "No",
            "Sometimes": "A veces",
            "Frequently": "Frecuentemente",
            "Always": "Siempre"
        }[x],
        index=["no", "Sometimes", "Frequently", "Always"].index(ejemplo["CAEC"])
    )

    smoke = st.selectbox(
        "¿Fuma?",
        options=["yes", "no"],
        format_func=lambda x: "Sí" if x == "yes" else "No",
        index=["yes", "no"].index(ejemplo["SMOKE"])
    )

    ch2o = st.slider(
        "Consumo diario de agua",
        min_value=1.0,
        max_value=3.0,
        value=float(ejemplo["CH2O"]),
        step=0.1
    )

    scc = st.selectbox(
        "¿Monitorea las calorías que consume?",
        options=["yes", "no"],
        format_func=lambda x: "Sí" if x == "yes" else "No",
        index=["yes", "no"].index(ejemplo["SCC"])
    )

    faf = st.slider(
        "Frecuencia de actividad física",
        min_value=0.0,
        max_value=3.0,
        value=float(ejemplo["FAF"]),
        step=0.1
    )

    tue = st.slider(
        "Tiempo usando dispositivos tecnológicos",
        min_value=0.0,
        max_value=2.0,
        value=float(ejemplo["TUE"]),
        step=0.1
    )

    calc = st.selectbox(
        "Consumo de alcohol",
        options=["no", "Sometimes", "Frequently", "Always"],
        format_func=lambda x: {
            "no": "No",
            "Sometimes": "A veces",
            "Frequently": "Frecuentemente",
            "Always": "Siempre"
        }[x],
        index=["no", "Sometimes", "Frequently", "Always"].index(ejemplo["CALC"])
    )

    mtrans = st.selectbox(
        "Medio de transporte utilizado",
        options=[
            "Public_Transportation",
            "Walking",
            "Automobile",
            "Motorbike",
            "Bike"
        ],
        format_func=lambda x: {
            "Public_Transportation": "Transporte público",
            "Walking": "Caminar",
            "Automobile": "Automóvil",
            "Motorbike": "Motocicleta",
            "Bike": "Bicicleta"
        }[x],
        index=[
            "Public_Transportation",
            "Walking",
            "Automobile",
            "Motorbike",
            "Bike"
        ].index(ejemplo["MTRANS"])
    )

    boton = st.form_submit_button("Predecir nivel de obesidad")

# ==============================
# PREDICCIÓN
# ==============================

if boton:
    datos = {
        "Gender": genero,
        "Age": edad,
        "Height": altura,
        "Weight": peso,
        "family_history_with_overweight": antecedentes,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans
    }

    df_input = pd.DataFrame([datos])

    # Asegurar el mismo orden de columnas usado en el entrenamiento
    df_input = df_input[columnas]

    # Codificar variables categóricas
    for columna, encoder in encoders.items():
        df_input[columna] = encoder.transform(df_input[columna])

    # Normalizar datos
    df_input_scaled = scaler.transform(df_input)

    # Predecir
    probabilidades = modelo.predict(df_input_scaled)
    clase_predicha = np.argmax(probabilidades, axis=1)[0]

    resultado_original = target_encoder.inverse_transform([clase_predicha])[0]
    resultado_espanol = traduccion_resultados.get(resultado_original, resultado_original)

    confianza = np.max(probabilidades) * 100

    st.success(f"Nivel de obesidad predicho: {resultado_espanol}")
    st.write(f"Confianza del modelo: {confianza:.2f}%")

    st.subheader("Probabilidades por clase")

    clases_originales = target_encoder.classes_
    clases_espanol = [
        traduccion_resultados.get(clase, clase)
        for clase in clases_originales
    ]

    df_probabilidades = pd.DataFrame({
        "Clase": clases_espanol,
        "Probabilidad": probabilidades[0]
    })

    df_probabilidades["Probabilidad"] = df_probabilidades["Probabilidad"].round(4)

    st.dataframe(df_probabilidades)