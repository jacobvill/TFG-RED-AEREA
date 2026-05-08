import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.neighbors import BallTree

# 1. CONFIGURACIÓN
st.set_page_config(page_title="TFG - Control de Mapa", layout="wide")


@st.cache_data
def cargar_y_procesar():
    df = pd.read_csv("airports.csv")
    tipos_validos = ['small_airport', 'medium_airport', 'large_airport']
    df = df[df['type'].isin(tipos_validos)].copy()
    df = df.dropna(subset=['latitude_deg', 'longitude_deg'])
    df['continent'] = df['continent'].fillna('NA')

    # Cálculo de aislamiento
    coords = np.radians(df[['latitude_deg', 'longitude_deg']].values)
    tree = BallTree(coords, metric='haversine')
    distancias, _ = tree.query(coords, k=2)
    df['distancia_vecino_km'] = distancias[:, 1] * 6371
    return df


df = cargar_y_procesar()

# 2. BARRA LATERAL
st.sidebar.header("Configuración")

# --- EL TRUCO DEL ZOOM ---
st.sidebar.subheader("🕹️ Control de Navegación")
zoom_interactivo = st.sidebar.toggle("Desbloquear Zoom con ratón", value=False)
if zoom_interactivo:
    st.sidebar.info("Zoom activado. Usa la rueda del ratón sobre el mapa.")
else:
    st.sidebar.warning("Zoom bloqueado para facilitar el scroll de la página.")

# Otros filtros
cont_sel = st.sidebar.multiselect("Continentes:", options=sorted(df['continent'].unique()), default=['EU'])
tipos_sel = st.sidebar.multiselect("Tipos:", options=['large_airport', 'medium_airport'], default=['large_airport'])

df_view = df[df['continent'].isin(cont_sel) & df['type'].isin(tipos_sel)]

# 3. CREACIÓN DEL MAPA CON GRAPH OBJECTS (Para tener más control)
st.subheader(f"Mapa de Infraestructura ({len(df_view)} puntos)")

fig = go.Figure()

# Añadimos los aeropuertos
fig.add_trace(go.Scattermapbox(
    lat=df_view['latitude_deg'],
    lon=df_view['longitude_deg'],
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=9,
        color='rgb(255, 75, 75)',  # Rojo profesional
        opacity=0.7
    ),
    text=df_view['name'],
    hoverinfo='text'
))

# Diseño del mapa
fig.update_layout(
    mapbox_style="carto-darkmatter",
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    height=700,
    mapbox=dict(
        center=dict(lat=40, lon=-3),  # Centrado en España por defecto
        zoom=3
    )
)

# 4. MOSTRAR MAPA CON CONFIGURACIÓN DINÁMICA
# Aquí pasamos el valor del Toggle al parámetro 'scrollZoom'
st.plotly_chart(
    fig,
    use_container_width=True,
    config={'scrollZoom': zoom_interactivo}
)

# 5. TABLA DE DATOS ABAJO
st.write("### Detalle de aeropuertos seleccionados")
st.dataframe(df_view[['ident', 'name', 'continent', 'distancia_vecino_km']].head(50))