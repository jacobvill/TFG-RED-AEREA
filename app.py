import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.neighbors import BallTree
from trino.dbapi import connect
from trino.auth import OAuth2Authentication
from datetime import datetime, time as dt_time
import time as t_lib
import webbrowser

# ----------------------------------------------------------------
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ----------------------------------------------------------------
st.set_page_config(page_title="TFG - Analizador OpenSky & Aeropuertos", layout="wide")


@st.cache_data
def cargar_y_procesar():
    # Cargar dataset de aeropuertos
    df = pd.read_csv("airports.csv")

    # Filtrar tipos y limpiar coordenadas
    tipos_validos = ['small_airport', 'medium_airport', 'large_airport']
    df = df[df['type'].isin(tipos_validos)].copy()
    df = df.dropna(subset=['latitude_deg', 'longitude_deg'])

    # Corregir Norteamérica (NA) que Pandas confunde con Nulo
    df['continent'] = df['continent'].fillna('NA')

    # --- CÁLCULO DE AISLAMIENTO (BallTree) ---
    # Convertimos a radianes para usar la métrica Haversine (distancia en esfera)
    coords = np.radians(df[['latitude_deg', 'longitude_deg']].values)
    tree = BallTree(coords, metric='haversine')

    # Buscamos los 2 vecinos más cercanos (k=2 porque el 1 es el mismo punto)
    distancias, _ = tree.query(coords, k=2)
    df['distancia_vecino_km'] = distancias[:, 1] * 6371

    return df


df_base = cargar_y_procesar()

# ----------------------------------------------------------------
# 2. BARRA LATERAL (FILTROS Y CREDENCIALES)
# ----------------------------------------------------------------
st.sidebar.header("🔑 Conexión OpenSky Network")
user_trino = st.sidebar.text_input("Usuario OpenSky (minúsculas)").lower()

st.sidebar.divider()
st.sidebar.header("📅 Momento Histórico")
fecha_sel = st.sidebar.date_input("Día del análisis", datetime.now())
hora_sel = st.sidebar.selectbox("Hora (UTC)", list(range(24)), index=12)

st.sidebar.divider()
st.sidebar.header("🎯 Destino Específico")
# Selector de aeropuerto de destino (solo Large para facilitar búsqueda)
df_grandes = df_base[df_base['type'] == 'large_airport'].sort_values('name')
nombres_aero = df_grandes['name'].tolist()
aero_destino_nombre = st.sidebar.selectbox("Vuelos con destino a:", nombres_aero)
aero_destino_icao = df_grandes[df_grandes['name'] == aero_destino_nombre]['ident'].values[0]

st.sidebar.divider()
st.sidebar.header("🗺️ Visualización de Mapa")
tipos_sel = st.sidebar.multiselect(
    "Tipos de aeropuertos a mostrar:",
    ['large_airport', 'medium_airport', 'small_airport'],
    default=['large_airport', 'medium_airport']
)
cont_sel = st.sidebar.multiselect(
    "Continentes:",
    sorted(df_base['continent'].unique().tolist()),
    default=['EU']
)


# ----------------------------------------------------------------
# 3. LÓGICA DE CONSULTA A TRINO (SQL)
# ----------------------------------------------------------------
def consultar_vuelos_trino(fecha, hora):
    from datetime import datetime, time as dt_time

    dt_combinada = datetime.combine(fecha, dt_time(hora, 0))
    ts_hour = int(dt_combinada.timestamp())

    # Reutilizar conexión si ya existe en session_state
    if "trino_conn" not in st.session_state:
        st.session_state.trino_conn = connect(
            host="trino.opensky-network.org",
            port=443,
            user="jaltevil@myuax.com",
            auth=OAuth2Authentication(),
            http_scheme="https",
            catalog="minio",
            schema="osky",
            request_timeout=60.0
        )

    conn = st.session_state.trino_conn

    query = f"""
        SELECT icao24, callsign, lat, lon, velocity, heading, baroaltitude
        FROM state_vectors_data4
        WHERE hour = {ts_hour}
          AND time BETWEEN {ts_hour} AND {ts_hour} + 60
          AND onground = false
          AND lat IS NOT NULL
          AND lon IS NOT NULL
        LIMIT 300
    """

    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)
# ----------------------------------------------------------------
# 4. CONSTRUCCIÓN DEL MAPA (PLOTLY 6)
# ----------------------------------------------------------------
st.title("🌍 TFG: Análisis de Infraestructura y Flujos Aéreos")
st.subheader(f"Aeropuertos y vuelos hacia {aero_destino_nombre} ({aero_destino_icao})")

# Filtrar DataFrame para la vista actual
mask = df_base['continent'].isin(cont_sel) & df_base['type'].isin(tipos_sel)
df_view = df_base[mask]

fig = go.Figure()

# CAPA 1: AEROPUERTOS (Círculos de colores)
colores = {"large_airport": "#FF4B4B", "medium_airport": "#1C83E1", "small_airport": "#00FF7F"}
tamanos = {"large_airport": 10, "medium_airport": 7, "small_airport": 4}

for tipo in tipos_sel:
    df_t = df_view[df_view['type'] == tipo]
    fig.add_trace(go.Scattermap(
        lat=df_t['latitude_deg'],
        lon=df_t['longitude_deg'],
        mode='markers',
        name=tipo,
        marker=go.scattermap.Marker(
            size=tamanos[tipo],
            color=colores[tipo],
            opacity=0.6
        ),
        text=df_t['name'],
        hoverinfo='text'
    ))

# CAPA 2: AVIONES (Se activa al pulsar el botón)
if st.sidebar.button("✈️ Ejecutar Consulta OpenSky"):
    with st.spinner("⏳ Revisa el navegador — necesitas hacer login en OpenSky..."):
        try:
            df_vuelos = consultar_vuelos_trino(fecha_sel, hora_sel)

            if not df_vuelos.empty:
                fig.add_trace(go.Scattermap(
                    lat=df_vuelos['lat'],
                    lon=df_vuelos['lon'],
                    mode='markers',
                    name='Aviones en vuelo',
                    marker=go.scattermap.Marker(
                        size=10,
                        color='yellow',
                        symbol='airport'
                    ),
                    text=df_vuelos['callsign'].str.strip() + "<br>" +
                         "Vel: " + df_vuelos['velocity'].astype(str) + " m/s<br>" +
                         "Alt: " + df_vuelos['baroaltitude'].astype(str) + " m",
                    hoverinfo='text'
                ))
                st.success(f"✅ {len(df_vuelos)} aviones encontrados.")
            else:
                st.warning("No se encontraron vuelos en ese momento.")

        except Exception as e:
            st.error(f"Error: {e}")
            # Resetear conexión por si el token expiró
            if "trino_conn" in st.session_state:
                del st.session_state.trino_conn

# Configuración de diseño del mapa
fig.update_layout(
    map_style="carto-darkmatter",
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    height=750,
    showlegend=True,
    legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")),
    map=dict(
        center=dict(lat=40, lon=-3),  # Centrado en España por defecto
        zoom=3.5
    )
)

# Renderizar mapa con Zoom de rueda habilitado
st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

# ----------------------------------------------------------------
# 5. TABLA DE ANÁLISIS DE AISLAMIENTO
# ----------------------------------------------------------------
st.divider()
st.subheader("📊 Análisis de Aislamiento Geográfico")
col1, col2 = st.columns([2, 1])

with col1:
    st.write("Aeropuertos más aislados (con menos vecinos cerca) en la selección actual:")
    top_aislados = df_view.sort_values('distancia_vecino_km', ascending=False).head(15)
    st.dataframe(top_aislados[['ident', 'name', 'municipality', 'distancia_vecino_km']], use_container_width=True)

with col2:
    st.info("""
    **Metodología TFG:**
    - Se utiliza un algoritmo *BallTree* con métrica *Haversine*.
    - La distancia se calcula respecto al aeropuerto más cercano de cualquier tipo.
    - Valores altos indican puntos críticos de conectividad o zonas remotas.
    """)
