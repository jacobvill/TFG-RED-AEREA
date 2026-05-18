import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.neighbors import BallTree
from trino.dbapi import connect
from trino.auth import OAuth2Authentication
from datetime import datetime, time as dt_time

# ----------------------------------------------------------------
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ----------------------------------------------------------------
st.set_page_config(page_title="TFG - Analizador OpenSky & Aeropuertos", layout="wide")


@st.cache_data
def cargar_y_procesar():
    df = pd.read_csv("airports.csv")
    tipos_validos = ['small_airport', 'medium_airport', 'large_airport']
    df = df[df['type'].isin(tipos_validos)].copy()
    df = df.dropna(subset=['latitude_deg', 'longitude_deg'])
    df['continent'] = df['continent'].fillna('NA')

    coords = np.radians(df[['latitude_deg', 'longitude_deg']].values)
    tree = BallTree(coords, metric='haversine')
    distancias, _ = tree.query(coords, k=2)
    df['distancia_vecino_km'] = distancias[:, 1] * 6371
    return df


df_base = cargar_y_procesar()

# ----------------------------------------------------------------
# 2. BARRA LATERAL — FILTROS
# ----------------------------------------------------------------
st.sidebar.header("🔑 Conexión OpenSky")
user_trino = st.sidebar.text_input("Usuario OpenSky (email, minúsculas)", value="jaltevil@myuax.com").lower()

st.sidebar.divider()
st.sidebar.header("📅 Momento Histórico")
fecha_sel = st.sidebar.date_input("Día del análisis", datetime(2024, 1, 1))
hora_sel = st.sidebar.selectbox("Hora (UTC)", list(range(24)), index=12)

st.sidebar.divider()
st.sidebar.header("🗺️ Aeropuertos en mapa")
tipos_sel = st.sidebar.multiselect(
    "Tipos de aeropuertos:",
    ['large_airport', 'medium_airport', 'small_airport'],
    default=['large_airport', 'medium_airport']
)
cont_sel = st.sidebar.multiselect(
    "Continentes:",
    sorted(df_base['continent'].unique().tolist()),
    default=['EU']
)

st.sidebar.divider()
st.sidebar.header("🎯 Filtro de destino (visual)")

# Opción "Todos" + lista de aeropuertos según los filtros activos
df_filtrados_mapa = df_base[
    df_base['continent'].isin(cont_sel) &
    df_base['type'].isin(tipos_sel)
].sort_values('name')

opciones_destino = ["— Mostrar todos los aviones —"] + df_filtrados_mapa['name'].tolist()
aero_destino_nombre = st.sidebar.selectbox("Aeropuerto de destino:", opciones_destino)

# Calcular radio de filtro si hay destino seleccionado
radio_km = st.sidebar.slider(
    "Radio de proximidad al destino (km):",
    min_value=50, max_value=2000, value=500, step=50,
    help="Aviones dentro de este radio del aeropuerto destino"
) if aero_destino_nombre != "— Mostrar todos los aviones —" else None

st.sidebar.divider()
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    btn_consultar = st.button("✈️ Consultar OpenSky", use_container_width=True)
with col_btn2:
    btn_limpiar = st.button("🗑️ Limpiar mapa", use_container_width=True)


# ----------------------------------------------------------------
# 3. LÓGICA DE CONSULTA A TRINO
# ----------------------------------------------------------------
def consultar_vuelos_trino(fecha, hora, usuario):
    dt_combinada = datetime.combine(fecha, dt_time(hora, 0))
    ts_hour = int(dt_combinada.timestamp())

    # Reutilizar conexión si ya existe y el usuario no cambió
    if "trino_conn" not in st.session_state or st.session_state.get("trino_user") != usuario:
        st.session_state.trino_conn = connect(
            host="trino.opensky-network.org",
            port=443,
            user=usuario,
            auth=OAuth2Authentication(),
            http_scheme="https",
            catalog="minio",
            schema="osky",
            request_timeout=60.0
        )
        st.session_state.trino_user = usuario

    conn = st.session_state.trino_conn

    # Query segura: partición por hour obligatoria, ventana de 60s, con LIMIT
    query = f"""
        SELECT icao24, callsign, lat, lon, velocity, heading, baroaltitude
        FROM state_vectors_data4
        WHERE hour = {ts_hour}
          AND time BETWEEN {ts_hour} AND {ts_hour} + 60
          AND onground = false
          AND lat IS NOT NULL
          AND lon IS NOT NULL
        LIMIT 500
    """

    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=cols)


def filtrar_por_proximidad(df_vuelos, nombre_aeropuerto, radio_km):
    """Filtra aviones dentro de un radio del aeropuerto destino usando Haversine."""
    fila_aero = df_base[df_base['name'] == nombre_aeropuerto]
    if fila_aero.empty:
        return df_vuelos

    lat_dest = np.radians(fila_aero['latitude_deg'].values[0])
    lon_dest = np.radians(fila_aero['longitude_deg'].values[0])

    coords_aviones = np.radians(df_vuelos[['lat', 'lon']].values)
    # Fórmula Haversine vectorizada
    dlat = coords_aviones[:, 0] - lat_dest
    dlon = coords_aviones[:, 1] - lon_dest
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_dest) * np.cos(coords_aviones[:, 0]) * np.sin(dlon / 2) ** 2
    distancias_km = 2 * 6371 * np.arcsin(np.sqrt(a))

    df_vuelos = df_vuelos.copy()
    df_vuelos['dist_destino_km'] = distancias_km.round(1)
    return df_vuelos[df_vuelos['dist_destino_km'] <= radio_km]


# ----------------------------------------------------------------
# 4. ACCIONES DE BOTONES
# ----------------------------------------------------------------
if btn_limpiar:
    st.session_state.df_vuelos_raw = pd.DataFrame()
    st.session_state.vuelos_fecha = ""
    st.session_state.vuelos_hora = ""

if btn_consultar:
    if not user_trino:
        st.sidebar.error("Introduce tu usuario de OpenSky.")
    else:
        with st.spinner("⏳ Conectando con OpenSky... Revisa el navegador si es la primera vez (login OAuth2)."):
            try:
                df_raw = consultar_vuelos_trino(fecha_sel, hora_sel, user_trino)
                st.session_state.df_vuelos_raw = df_raw
                st.session_state.vuelos_fecha = str(fecha_sel)
                st.session_state.vuelos_hora = hora_sel
                st.sidebar.success(f"✅ {len(df_raw)} aviones descargados.")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
                if "trino_conn" in st.session_state:
                    del st.session_state.trino_conn

# ----------------------------------------------------------------
# 5. FILTRO VISUAL DE DESTINO (sobre datos ya descargados)
# ----------------------------------------------------------------
df_vuelos_raw = st.session_state.get("df_vuelos_raw", pd.DataFrame())

if not df_vuelos_raw.empty and aero_destino_nombre != "— Mostrar todos los aviones —":
    df_vuelos = filtrar_por_proximidad(df_vuelos_raw, aero_destino_nombre, radio_km)
    filtro_activo = True
else:
    df_vuelos = df_vuelos_raw.copy()
    filtro_activo = False

# ----------------------------------------------------------------
# 6. CONSTRUCCIÓN DEL MAPA
# ----------------------------------------------------------------
st.title("🌍 TFG: Análisis de Infraestructura y Flujos Aéreos")

# Info de la consulta activa
if not df_vuelos_raw.empty:
    fecha_label = st.session_state.get("vuelos_fecha", "")
    hora_label = st.session_state.get("vuelos_hora", "")
    if filtro_activo:
        st.info(
            f"📦 **{len(df_vuelos_raw)}** aviones descargados el **{fecha_label}** a las **{hora_label}:00 UTC** · "
            f"🎯 Mostrando **{len(df_vuelos)}** dentro de {radio_km} km de **{aero_destino_nombre}**"
        )
    else:
        st.info(
            f"📦 **{len(df_vuelos_raw)}** aviones en vuelo el **{fecha_label}** a las **{hora_label}:00 UTC** · "
            f"Usa el filtro de destino en la barra lateral para acotar."
        )

mask = df_base['continent'].isin(cont_sel) & df_base['type'].isin(tipos_sel)
df_view = df_base[mask]

fig = go.Figure()

# --- CAPA 1: AEROPUERTOS ---
colores = {"large_airport": "#FF4B4B", "medium_airport": "#1C83E1", "small_airport": "#00FF7F"}
tamanos = {"large_airport": 10, "medium_airport": 7, "small_airport": 4}

for tipo in tipos_sel:
    df_t = df_view[df_view['type'] == tipo]
    fig.add_trace(go.Scattermap(
        lat=df_t['latitude_deg'],
        lon=df_t['longitude_deg'],
        mode='markers',
        name=tipo,
        marker=go.scattermap.Marker(size=tamanos[tipo], color=colores[tipo], opacity=0.6),
        text=df_t['name'],
        hoverinfo='text'
    ))

# --- CAPA 2: AEROPUERTO DESTINO SELECCIONADO (destacado) ---
if filtro_activo:
    fila_dest = df_base[df_base['name'] == aero_destino_nombre]
    if not fila_dest.empty:
        fig.add_trace(go.Scattermap(
            lat=fila_dest['latitude_deg'],
            lon=fila_dest['longitude_deg'],
            mode='markers',
            name=f"🎯 {aero_destino_nombre}",
            marker=go.scattermap.Marker(size=18, color='orange', symbol='airport'),
            text=[f"DESTINO: {aero_destino_nombre}"],
            hoverinfo='text'
        ))

# --- CAPA 3: AVIONES ---
if not df_vuelos.empty:
    hover_text = (
        "✈️ " + df_vuelos['callsign'].str.strip() + "<br>" +
        "ICAO24: " + df_vuelos['icao24'] + "<br>" +
        "Vel: " + df_vuelos['velocity'].round(0).astype(str) + " m/s (" +
        (df_vuelos['velocity'] * 3.6).round(0).astype(str) + " km/h)<br>" +
        "Alt: " + df_vuelos['baroaltitude'].round(0).astype(str) + " m (" +
        (df_vuelos['baroaltitude'] * 3.281).round(0).astype(str) + " ft)<br>" +
        "Rumbo: " + df_vuelos['heading'].round(0).astype(str) + "°"
    )
    if filtro_activo and 'dist_destino_km' in df_vuelos.columns:
        hover_text = hover_text + "<br>Dist. destino: " + df_vuelos['dist_destino_km'].astype(str) + " km"

    fig.add_trace(go.Scattermap(
        lat=df_vuelos['lat'],
        lon=df_vuelos['lon'],
        mode='markers',
        name=f"Aviones ({len(df_vuelos)})",
        marker=go.scattermap.Marker(size=10, color='yellow', symbol='airport'),
        text=hover_text,
        hoverinfo='text'
    ))

# Centro del mapa: si hay destino seleccionado, centrar en él
map_center = dict(lat=40, lon=-3)
map_zoom = 3.5
if filtro_activo:
    fila_dest = df_base[df_base['name'] == aero_destino_nombre]
    if not fila_dest.empty:
        map_center = dict(
            lat=fila_dest['latitude_deg'].values[0],
            lon=fila_dest['longitude_deg'].values[0]
        )
        map_zoom = 4.0

fig.update_layout(
    map_style="carto-darkmatter",
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    height=720,
    showlegend=True,
    legend=dict(
        yanchor="top", y=0.98, xanchor="left", x=0.02,
        bgcolor="rgba(0,0,0,0.6)", font=dict(color="white")
    ),
    map=dict(center=map_center, zoom=map_zoom)
)

st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

# ----------------------------------------------------------------
# 7. TABLAS
# ----------------------------------------------------------------
if not df_vuelos.empty:
    st.divider()
    st.subheader(f"📋 Vuelos mostrados ({len(df_vuelos)})")

    df_tabla = df_vuelos.copy()
    df_tabla['callsign'] = df_tabla['callsign'].str.strip()
    df_tabla['vel_kmh'] = (df_tabla['velocity'] * 3.6).round(0)
    df_tabla['alt_ft'] = (df_tabla['baroaltitude'] * 3.281).round(0)

    columnas = ['callsign', 'icao24', 'lat', 'lon', 'vel_kmh', 'alt_ft', 'heading']
    nombres = {
        'callsign': 'Vuelo', 'icao24': 'ICAO24',
        'lat': 'Latitud', 'lon': 'Longitud',
        'vel_kmh': 'Vel (km/h)', 'alt_ft': 'Altitud (ft)', 'heading': 'Rumbo (°)'
    }
    if filtro_activo and 'dist_destino_km' in df_tabla.columns:
        columnas.append('dist_destino_km')
        nombres['dist_destino_km'] = f'Dist. a {aero_destino_nombre[:20]}... (km)'

    st.dataframe(df_tabla[columnas].rename(columns=nombres), use_container_width=True)

st.divider()
st.subheader("📊 Análisis de Aislamiento Geográfico")
col1, col2 = st.columns([2, 1])

with col1:
    st.write("Aeropuertos más aislados en la selección actual:")
    top_aislados = df_view.sort_values('distancia_vecino_km', ascending=False).head(15)
    st.dataframe(
        top_aislados[['ident', 'name', 'municipality', 'distancia_vecino_km']],
        use_container_width=True
    )

with col2:
    st.info("""
    **Metodología TFG:**
    - Algoritmo *BallTree* con métrica *Haversine*.
    - Distancia al aeropuerto más cercano de cualquier tipo.
    - Valores altos = zonas remotas o puntos críticos de conectividad.
    """)