import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.neighbors import BallTree
from trino.dbapi import connect
from trino.auth import OAuth2Authentication
from datetime import datetime, time as dt_time, timezone

# ================================================================
# 1. CONFIGURACIÓN
# ================================================================
st.set_page_config(page_title="TFG - Analizador OpenSky & Aeropuertos", layout="wide")

# Bounding boxes por continente (lon_min, lat_min, lon_max, lat_max)
BBOXES = {
    'EU': (-25.0,  29.0,  45.0, 81.2),
    'NA': (-176.6,  7.4, -52.0, 83.1),
    'SA': (-109.4,-55.0, -32.4, 12.4),
    'AS': (  26.0,-12.2, 180.0, 80.8),
    'AF': ( -25.1,-34.8,  63.4, 37.2),
    'OC': ( 110.0,-46.9, 180.0, 28.2),
    'AN': (-180.0,-90.0, 180.0,-60.0),
}


# ================================================================
# 2. CARGA DE AEROPUERTOS
# ================================================================
@st.cache_data
def cargar_aeropuertos():
    df = pd.read_csv("airports.csv")
    tipos_validos = ['small_airport', 'medium_airport', 'large_airport']
    df = df[df['type'].isin(tipos_validos)].copy()
    df = df.dropna(subset=['latitude_deg', 'longitude_deg'])
    df['continent'] = df['continent'].fillna('NA')

    coords = np.radians(df[['latitude_deg', 'longitude_deg']].values)
    tree = BallTree(coords, metric='haversine')
    distancias, _ = tree.query(coords, k=2)
    df['distancia_vecino_km'] = (distancias[:, 1] * 6371).round(1)
    return df


df_base = cargar_aeropuertos()


# ================================================================
# 3. HELPERS
# ================================================================
def get_conexion(usuario):
    if "trino_conn" not in st.session_state or st.session_state.get("trino_user") != usuario:
        st.session_state.trino_conn = connect(
            host="trino.opensky-network.org",
            port=443,
            user=usuario,
            auth=OAuth2Authentication(),
            http_scheme="https",
            catalog="minio",
            schema="osky",
            request_timeout=120.0
        )
        st.session_state.trino_user = usuario
    return st.session_state.trino_conn


def calcular_bbox(continentes):
    lon_min, lat_min, lon_max, lat_max = 180, 90, -180, -90
    for c in continentes:
        if c in BBOXES:
            lnmin, ltmin, lnmax, ltmax = BBOXES[c]
            lon_min = min(lon_min, lnmin)
            lat_min = min(lat_min, ltmin)
            lon_max = max(lon_max, lnmax)
            lat_max = max(lat_max, ltmax)
    return lat_min - 2, lat_max + 2, lon_min - 2, lon_max + 2


# ================================================================
# 4. CONSULTA A TRINO
# ================================================================
def consultar_opensky(fecha, hora, minuto, usuario, continentes):
    """
    Devuelve TODOS los aviones en vuelo en la zona y momento indicados.

    Estrategia (siguiendo las guías de OpenSky):
    - Filtra SIEMPRE por 'hour' (partición obligatoria).
    - Ventana de 60 s alrededor del minuto elegido.
    - GROUP BY icao24 con MAX_BY → exactamente 1 fila por avión, la más reciente.
    - Sin LIMIT artificial: devuelve todos los aviones reales en el área.
    - Bbox geográfico para no escanear datos innecesarios.
    """

    # UTC explícito — datetime naive usa la hora local del PC
    dt_utc    = datetime(fecha.year, fecha.month, fecha.day, hora, minuto, 0, tzinfo=timezone.utc)
    ts_exacto = int(dt_utc.timestamp())
    ts_hour   = ts_exacto - (ts_exacto % 3600)   # inicio de hora → partición

    lat_min, lat_max, lon_min, lon_max = calcular_bbox(continentes)

    conn = get_conexion(usuario)

    # ----------------------------------------------------------------
    # Query correcta según la documentación de OpenSky:
    # GROUP BY icao24 + MAX_BY → 1 posición por avión (la más reciente)
    # Sin LIMIT → devuelve TODOS los aviones del área sin truncar
    # ----------------------------------------------------------------
    query = f"""
        SELECT
            icao24,
            MAX_BY(callsign,     time) AS callsign,
            MAX_BY(lat,          time) AS lat,
            MAX_BY(lon,          time) AS lon,
            MAX_BY(velocity,     time) AS velocity,
            MAX_BY(heading,      time) AS heading,
            MAX_BY(baroaltitude, time) AS baroaltitude
        FROM state_vectors_data4
        WHERE hour     = {ts_hour}
          AND time     BETWEEN {ts_exacto} AND {ts_exacto} + 60
          AND onground = false
          AND lat      BETWEEN {lat_min} AND {lat_max}
          AND lon      BETWEEN {lon_min} AND {lon_max}
          AND lat      IS NOT NULL
          AND lon      IS NOT NULL
        GROUP BY icao24
    """

    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    df   = pd.DataFrame(rows, columns=cols)
    return df.reset_index(drop=True)


# ================================================================
# 5. FILTRO POR PROXIMIDAD
# ================================================================
def filtrar_por_proximidad(df_vuelos, nombre_aeropuerto, radio_km):
    fila = df_base[df_base['name'] == nombre_aeropuerto]
    if fila.empty or df_vuelos.empty:
        return df_vuelos

    lat_d = np.radians(fila['latitude_deg'].values[0])
    lon_d = np.radians(fila['longitude_deg'].values[0])

    coords = np.radians(df_vuelos[['lat', 'lon']].values)
    dlat   = coords[:, 0] - lat_d
    dlon   = coords[:, 1] - lon_d
    a      = np.sin(dlat/2)**2 + np.cos(lat_d) * np.cos(coords[:, 0]) * np.sin(dlon/2)**2
    dist   = (2 * 6371 * np.arcsin(np.sqrt(a))).round(1)

    df_out = df_vuelos.copy()
    df_out['dist_destino_km'] = dist
    return df_out[df_out['dist_destino_km'] <= radio_km].reset_index(drop=True)


# ================================================================
# 6. BARRA LATERAL
# ================================================================
st.sidebar.header("🔑 Conexión OpenSky")
user_trino = st.sidebar.text_input(
    "Usuario OpenSky (email, minúsculas)",
    value="jaltevil@myuax.com"
).lower()

st.sidebar.divider()
st.sidebar.header("📅 Momento del análisis (UTC)")
fecha_sel  = st.sidebar.date_input("Día", datetime(2024, 1, 1))
hora_sel   = st.sidebar.selectbox("Hora (UTC)", list(range(24)), index=12)
minuto_sel = st.sidebar.selectbox("Minuto (UTC)", list(range(0, 60, 5)), index=0)

st.sidebar.divider()
st.sidebar.header("🗺️ Aeropuertos en mapa")
tipos_sel = st.sidebar.multiselect(
    "Tipos:",
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

df_para_destino = df_base[
    df_base['continent'].isin(cont_sel) &
    df_base['type'].isin(tipos_sel)
].sort_values('name')

opciones_destino = ["— Todos los aviones —"] + df_para_destino['name'].tolist()
destino_sel   = st.sidebar.selectbox("Aeropuerto de destino:", opciones_destino)
filtro_activo = destino_sel != "— Todos los aviones —"

radio_km = None
if filtro_activo:
    radio_km = st.sidebar.slider(
        "Radio de proximidad (km):",
        min_value=100, max_value=3000, value=800, step=100,
        help="Aviones dentro de este radio del aeropuerto destino"
    )

st.sidebar.divider()
st.sidebar.caption("⚠️ Pulsa 'Consultar' cada vez que cambies fecha u hora.")
col1_btn, col2_btn = st.sidebar.columns(2)
with col1_btn:
    btn_consultar = st.button("✈️ Consultar OpenSky", use_container_width=True)
with col2_btn:
    btn_limpiar = st.button("🗑️ Limpiar mapa", use_container_width=True)


# ================================================================
# 7. ACCIONES DE BOTONES
# ================================================================
if btn_limpiar:
    for k in ['df_vuelos_raw', 'vuelos_fecha', 'vuelos_hora', 'vuelos_minuto']:
        st.session_state[k] = None

if btn_consultar:
    if not user_trino:
        st.sidebar.error("Introduce tu usuario de OpenSky.")
    elif not cont_sel:
        st.sidebar.error("Selecciona al menos un continente.")
    else:
        with st.spinner("⏳ Consultando OpenSky... (puede tardar 15-30 s)"):
            try:
                df_raw = consultar_opensky(
                    fecha_sel, hora_sel, minuto_sel, user_trino, cont_sel
                )
                st.session_state['df_vuelos_raw'] = df_raw
                st.session_state['vuelos_fecha']  = str(fecha_sel)
                st.session_state['vuelos_hora']   = hora_sel
                st.session_state['vuelos_minuto'] = minuto_sel

                if df_raw.empty:
                    st.sidebar.warning("Sin resultados. Prueba otra fecha/hora.")
                else:
                    st.sidebar.success(f"✅ {len(df_raw)} aviones encontrados.")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
                if "trino_conn" in st.session_state:
                    del st.session_state['trino_conn']


# ================================================================
# 8. PREPARAR DATOS
# ================================================================
df_vuelos_raw = st.session_state.get('df_vuelos_raw')
if df_vuelos_raw is None:
    df_vuelos_raw = pd.DataFrame()

if not df_vuelos_raw.empty and filtro_activo and radio_km:
    df_vuelos = filtrar_por_proximidad(df_vuelos_raw, destino_sel, radio_km)
else:
    df_vuelos = df_vuelos_raw.copy()

mask    = df_base['continent'].isin(cont_sel) & df_base['type'].isin(tipos_sel)
df_view = df_base[mask]


# ================================================================
# 9. TÍTULO E INFO BANNER
# ================================================================
st.title("🌍 TFG: Análisis de Infraestructura y Flujos Aéreos")

if not df_vuelos_raw.empty:
    f_label = st.session_state.get('vuelos_fecha', '')
    h_label = st.session_state.get('vuelos_hora', 0)
    m_label = st.session_state.get('vuelos_minuto', 0)
    ts_str  = f"**{f_label}** a las **{h_label:02d}:{m_label:02d} UTC**"

    if filtro_activo:
        st.info(
            f"📦 **{len(df_vuelos_raw)}** aviones · {ts_str} · "
            f"🎯 Mostrando **{len(df_vuelos)}** dentro de **{radio_km} km** de **{destino_sel}**"
        )
    else:
        st.info(
            f"📦 **{len(df_vuelos_raw)}** aviones en vuelo · {ts_str} · "
            f"Usa el filtro de destino para acotar."
        )


# ================================================================
# 10. MAPA
# ================================================================
fig = go.Figure()

# --- CAPA 1: AEROPUERTOS ---
colores = {"large_airport": "#FF4B4B", "medium_airport": "#1C83E1", "small_airport": "#00FF7F"}
tamanos = {"large_airport": 10,        "medium_airport": 7,         "small_airport": 4}

for tipo in tipos_sel:
    df_t = df_view[df_view['type'] == tipo]
    if df_t.empty:
        continue
    fig.add_trace(go.Scattermap(
        lat=df_t['latitude_deg'],
        lon=df_t['longitude_deg'],
        mode='markers',
        name=tipo,
        marker=go.scattermap.Marker(size=tamanos[tipo], color=colores[tipo], opacity=0.6),
        text=df_t['name'],
        hoverinfo='text'
    ))

# --- CAPA 2: AEROPUERTO DESTINO DESTACADO ---
if filtro_activo:
    fd = df_base[df_base['name'] == destino_sel]
    if not fd.empty:
        icao_d = fd['ident'].values[0]
        fig.add_trace(go.Scattermap(
            lat=fd['latitude_deg'],
            lon=fd['longitude_deg'],
            mode='markers+text',
            name="🎯 Destino",
            text=[icao_d],
            textposition="top right",
            marker=go.scattermap.Marker(size=20, color='orange'),
            hovertext=[f"🎯 {destino_sel} ({icao_d})"],
            hoverinfo='text'
        ))

# --- CAPA 3: AVIONES ---
if not df_vuelos.empty:
    dv = df_vuelos.copy()
    dv['callsign']     = dv['callsign'].fillna('').str.strip()
    dv['velocity']     = pd.to_numeric(dv['velocity'],     errors='coerce').fillna(0)
    dv['baroaltitude'] = pd.to_numeric(dv['baroaltitude'], errors='coerce').fillna(0)
    dv['heading']      = pd.to_numeric(dv['heading'],      errors='coerce').fillna(0)

    vel_kmh = (dv['velocity'] * 3.6).round(0).astype(int).astype(str)
    alt_ft  = (dv['baroaltitude'] * 3.281).round(0).astype(int).astype(str)

    hover = (
        "✈️ <b>" + dv['callsign'] + "</b><br>" +
        "ICAO24: "  + dv['icao24'] + "<br>" +
        "Vel: "     + vel_kmh + " km/h<br>" +
        "Alt: "     + alt_ft  + " ft<br>" +
        "Rumbo: "   + dv['heading'].round(0).astype(int).astype(str) + "°"
    )
    if filtro_activo and 'dist_destino_km' in dv.columns:
        hover = hover + "<br>Dist. destino: " + dv['dist_destino_km'].astype(str) + " km"

    fig.add_trace(go.Scattermap(
        lat=dv['lat'],
        lon=dv['lon'],
        mode='markers',
        name=f"✈️ Aviones ({len(dv)})",
        marker=go.scattermap.Marker(size=10, color='yellow'),
        text=hover,
        hoverinfo='text'
    ))

# Centro del mapa
if filtro_activo:
    fd = df_base[df_base['name'] == destino_sel]
    map_center = (
        dict(lat=fd['latitude_deg'].values[0], lon=fd['longitude_deg'].values[0])
        if not fd.empty else dict(lat=40, lon=-3)
    )
elif not df_view.empty:
    map_center = dict(
        lat=df_view['latitude_deg'].mean(),
        lon=df_view['longitude_deg'].mean()
    )
else:
    map_center = dict(lat=40, lon=-3)

fig.update_layout(
    map_style="carto-darkmatter",
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    height=720,
    showlegend=True,
    legend=dict(
        yanchor="top", y=0.98, xanchor="left", x=0.02,
        bgcolor="rgba(0,0,0,0.6)", font=dict(color="white")
    ),
    map=dict(center=map_center, zoom=3.5)
)

st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})


# ================================================================
# 11. TABLA DE VUELOS
# ================================================================
if not df_vuelos.empty:
    st.divider()
    st.subheader(f"📋 Vuelos en pantalla ({len(df_vuelos)})")

    dt = df_vuelos.copy()
    dt['callsign'] = dt['callsign'].fillna('').str.strip()
    dt['vel_kmh']  = (pd.to_numeric(dt['velocity'],     errors='coerce').fillna(0) * 3.6).round(0).astype(int)
    dt['alt_ft']   = (pd.to_numeric(dt['baroaltitude'], errors='coerce').fillna(0) * 3.281).round(0).astype(int)
    dt['heading']  = pd.to_numeric(dt['heading'],  errors='coerce').fillna(0).round(1)
    dt['lat']      = dt['lat'].round(4)
    dt['lon']      = dt['lon'].round(4)

    cols_t  = ['callsign', 'icao24', 'lat', 'lon', 'vel_kmh', 'alt_ft', 'heading']
    nombres = {
        'callsign': 'Vuelo',      'icao24': 'ICAO24',
        'lat':      'Latitud',    'lon':    'Longitud',
        'vel_kmh':  'Vel (km/h)', 'alt_ft': 'Altitud (ft)',
        'heading':  'Rumbo (°)'
    }

    if filtro_activo and 'dist_destino_km' in dt.columns:
        cols_t.append('dist_destino_km')
        nc = destino_sel[:25] + "..." if len(destino_sel) > 25 else destino_sel
        nombres['dist_destino_km'] = f'Dist. {nc} (km)'
        dt = dt.sort_values('dist_destino_km')

    st.dataframe(dt[cols_t].rename(columns=nombres), use_container_width=True)


# ================================================================
# 12. TABLA DE AISLAMIENTO
# ================================================================
st.divider()
st.subheader(" Análisis de Aislamiento Geográfico")
col_a, col_b = st.columns([2, 1])

with col_a:
    st.write("Aeropuertos más aislados en la selección actual:")
    top = df_view.sort_values('distancia_vecino_km', ascending=False).head(15)
    st.dataframe(
        top[['ident', 'name', 'municipality', 'distancia_vecino_km']].rename(columns={
            'ident': 'ICAO', 'name': 'Nombre',
            'municipality': 'Ciudad', 'distancia_vecino_km': 'Dist. vecino (km)'
        }),
        use_container_width=True
    )

with col_b:
    st.info("""
    **Metodología TFG:**
    - Algoritmo *BallTree* con métrica *Haversine*.
    - Distancia al aeropuerto más cercano de cualquier tipo.
    - Valores altos = zonas remotas o puntos críticos de conectividad.
    """)