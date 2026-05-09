import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.neighbors import BallTree

# 1. CONFIGURACIÓN
st.set_page_config(page_title="TFG - Mapa Profesional", layout="wide")


@st.cache_data
def cargar_y_procesar():
    df = pd.read_csv("airports.csv")
    # Aseguramos que los tres tipos estén disponibles
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

# 2. BARRA LATERAL (Filtros corregidos)
st.sidebar.header("Panel de Control")

# Ahora permitimos elegir los 3 tipos
tipos_sel = st.sidebar.multiselect(
    "Selecciona tipos de aeropuerto:",
    options=['large_airport', 'medium_airport', 'small_airport'],
    default=['large_airport', 'medium_airport']  # Los pequeños son muchos, mejor activarlos a mano
)

lista_cont = sorted(df['continent'].unique().tolist())
cont_sel = st.sidebar.multiselect("Continentes:", options=lista_cont, default=['EU', 'NA', 'SA'])

# Filtro de datos
df_view = df[df['continent'].isin(cont_sel) & df['type'].isin(tipos_sel)]

# 3. CREACIÓN DEL MAPA CAPA POR CAPA (Para recuperar colores)
st.subheader(f"Infraestructura detectada: {len(df_view):,} puntos")

fig = go.Figure()

# Definimos los colores para cada tipo
colores = {
    "large_airport": "#FF4B4B",  # Rojo
    "medium_airport": "#1C83E1",  # Azul
    "small_airport": "#00FF7F"  # Verde
}

# Añadimos una traza (capa) diferente por cada tipo para tener colores y leyenda
for tipo in tipos_sel:
    df_tipo = df_view[df_view['type'] == tipo]

    fig.add_trace(go.Scattermapbox(
        lat=df_tipo['latitude_deg'],
        lon=df_tipo['longitude_deg'],
        mode='markers',
        name=tipo,  # Esto crea la leyenda automáticamente
        marker=go.scattermapbox.Marker(
            size=8 if tipo != 'small_airport' else 5,  # Los pequeños más finos para no saturar
            color=colores[tipo],
            opacity=0.7
        ),
        text=df_tipo['name'],
        hoverinfo='text'
    ))

# Diseño del mapa
fig.update_layout(
    mapbox_style="carto-darkmatter",
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    height=750,
    showlegend=True,
    legend=dict(
        yanchor="top", y=0.95, xanchor="left", x=0.02,
        bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")
    ),
    mapbox=dict(
        center=dict(lat=20, lon=0),
        zoom=1.5
    )
)

# 4. CONFIGURACIÓN DE ZOOM (UX Mejorada)
# Explicación para tu TFG: Activamos el scrollZoom pero avisamos al usuario.
# En la web, el click izquierdo se queda para mover (Pan) y la rueda para Zoom.
st.plotly_chart(
    fig,
    use_container_width=True,
    config={'scrollZoom': True}  # Activado por defecto para mayor comodidad0
)

st.info(
    "💡 **Consejo de navegación:** Usa la rueda del ratón para Zoom y mantén el click izquierdo para arrastrar el mapa.")

# 5. TABLA DE DATOS
with st.expander("Ver lista detallada de aeropuertos"):
    st.dataframe(df_view[['ident', 'name', 'type', 'continent', 'distancia_vecino_km']])