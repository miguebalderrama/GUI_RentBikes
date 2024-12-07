import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import requests
from decouple import config

import os

#API_TOKEN = config("MY_API_TOKEN")
API_TOKEN = os.getenv('MY_API_TOKEN')

# Cargar el modelo guardado
@st.cache_resource  # Para almacenar en caché el modelo
def cargar_modelo():
    try:
        with open("xgboost_model.pkl", "rb") as file:
            modelo = pickle.load(file)
        return modelo
    except FileNotFoundError:
        st.error("El archivo del modelo no se encuentra en la ruta especificada.")
    except Exception as e:
        st.error(f"Ocurrió un error al cargar el modelo: {e}")

xgboost_model = cargar_modelo()

# Cargar el scaler entrenado
@st.cache_resource  # Para almacenar en caché el scaler
def cargar_scaler():
    try:
        with open("scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        return scaler
    except FileNotFoundError:
        st.error("El archivo del scaler no se encuentra en la ruta especificada.")
    except Exception as e:
        st.error(f"Ocurrió un error al cargar el scaler: {e}")

scaler = cargar_scaler()

# Agregar contenido a la barra lateral
st.sidebar.title("Información del sistema")
st.sidebar.image("images.jpg", caption="Sistema de predicción de bicicletas",  use_container_width=True)
st.sidebar.markdown(
    """
    <p style="line-height:0.9;">
        Este sistema utiliza un modelo de aprendizaje automático para predecir la cantidad de bicicletas alquiladas basado en diferentes factores climáticos y temporales.
    </p>
    """,
    unsafe_allow_html=True
)

# Crear interfaz de usuario
st.title("Sistema de Predicción de Bicicletas Compartidas")
st.write("Ingrese los valores para hacer una predicción:")

# Función para obtener datos en tiempo real (ejemplo con clima)
def get_seoul_weather(api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q=Seoul&appid={api_key}&units=metric"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        
        # Extraer información que coincida con las columnas de tu modelo
        weather = {
            "Temperatura (°C)": data["main"]["temp"],  # Temperature
            "Humedad (%)": data["main"]["humidity"],  # Humidity
            "Velocidad del viento (m/s)": data["wind"]["speed"],  # Wind speed
            "Visibilidad (10m)": data.get("visibility", 0) / 10,  # Visibility in decameters
            "Temperatura de rocío (°C)": None,  # Dew point (requiere cálculo)
            "Radiación solar (MJ/m2)": None,  # No disponible directamente
            "Precipitación (mm)": data.get("rain", {}).get("1h", 0),  # Rainfall in last hour
            "Nieve (cm)": data.get("snow", {}).get("1h", 0) * 10,  # Snowfall in last hour (convert to cm)
            "Descripción": data["weather"][0]["description"].capitalize(),
        }

        # Calcular el punto de rocío (Dew Point) si se proporcionan datos suficientes
        if "temp" in data["main"] and "humidity" in data["main"]:
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            weather["Temperatura de rocío (°C)"] = temp - ((100 - humidity) / 5)
        
        # Radiación solar no está disponible directamente en la API gratuita.
        # Se puede estimar usando librerías especializadas o servicios adicionales.

        return weather
    else:
        return None

# Configuración de la barra lateral
#st.sidebar.title("Datos en tiempo real - Seúl")

# Obtener y mostrar datos en tiempo real
api_key = API_TOKEN  # Reemplaza con tu clave de API
weather_data = get_seoul_weather(api_key)

if weather_data: 
    # Insertar un estilo CSS para reducir el interlineado
    st.sidebar.markdown(
        """
        <style>
        .sidebar-text {
            line-height: 0.9;  /* Ajusta este valor para cambiar el interlineado */
            margin-bottom: 2px; /* Espaciado inferior entre elementos */
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    st.sidebar.subheader("Clima actual en Seul")
    for key, value in weather_data.items():
        st.sidebar.markdown(f"<p class='sidebar-text'>{key}: {value}</p>", unsafe_allow_html=True)
else:
    st.sidebar.error("No se pudieron cargar los datos en tiempo real.")

# División de columnas para los campos de entrada
col1, col2 = st.columns(2)

# Campos de entrada en la primera columna
with col1:
    month = st.number_input("Mes (1-12)", min_value=1, max_value=12, step=1)
    day = st.number_input("Día (1-31)", min_value=1, max_value=31, step=1)
    hour = st.number_input("Hora (0-23)", min_value=0, max_value=23, step=1)
    temperature = st.number_input("Temperatura (°C)")
    humidity = st.number_input("Humedad (%)")
    visibility = st.number_input("Visibilidad (10 m)", min_value=0.0)
    wind_speed = st.number_input("Velocidad del viento (m/s)")

# Campos de entrada en la segunda columna
with col2:
    dew_point_temperature = st.number_input("Temperatura de rocío (°C)")
    solar_radiation = st.number_input("Radiación solar (MJ/m2)")
    rainfall = st.number_input("Precipitación (mm)")
    snowfall = st.number_input("Nieve (cm)")    
    holiday = st.selectbox("¿Es día festivo?", options=["No Holiday", "Holiday"])
    season = st.selectbox("Estación del año", options=["Spring", "Summer", "Autumn", "Winter"])
    functioning_day = st.selectbox("¿Es un día de funcionamiento?", options=["Yes", "No"])

# Generar columnas binarias para el DataFrame
season_columns = {"Spring": [0, 1, 0, 0], "Summer": [0, 0, 1, 0], 
                  "Autumn": [1, 0, 0, 0], "Winter": [0, 0, 0, 1]}
holiday_columns = {"Holiday": [1, 0], "No Holiday": [0, 1]}
functioning_day_columns = {"Yes": [0, 1], "No": [1, 0]}

# Mapear los valores seleccionados
seasons_autumn, seasons_spring, seasons_summer, seasons_winter = season_columns[season]
holiday_holiday, holiday_no_holiday = holiday_columns[holiday]
functioning_day_no, functioning_day_yes = functioning_day_columns[functioning_day]

# Crear el DataFrame
data = pd.DataFrame({
    'Hour': [hour],
    'Temperature(°C)': [temperature],
    'Humidity(%)': [humidity],
    'Wind speed (m/s)': [wind_speed],
    'Visibility (10m)': [visibility],
    'Dew point temperature(°C)': [dew_point_temperature],
    'Solar Radiation (MJ/m2)': [solar_radiation],
    'Rainfall(mm)': [rainfall],
    'Snowfall (cm)': [snowfall],
    'Month': [month],
    'Day': [day],
    'Seasons_Autumn': [seasons_autumn],
    'Seasons_Spring': [seasons_spring],
    'Seasons_Summer': [seasons_summer],
    'Seasons_Winter': [seasons_winter],
    'Holiday_Holiday': [holiday_holiday],
    'Holiday_No Holiday': [holiday_no_holiday],
    'Functioning Day_No': [functioning_day_no],
    'Functioning Day_Yes': [functioning_day_yes]
})


input_df = pd.DataFrame(data)

# Aplicar el escalado utilizando el scaler cargado
input_df_scaled = scaler.transform(input_df)
print(input_df)
# Realizar la predicción
if st.button("Predecir"):
    try:
        prediction = xgboost_model.predict(input_df_scaled)
        st.success(f"El número estimado de bicicletas alquiladas es: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Ocurrió un error al realizar la predicción: {e}")
