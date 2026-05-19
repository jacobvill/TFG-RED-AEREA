import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.neighbors import BallTree
from trino.dbapi import connect
from trino.auth import OAuth2Authentication
from datetime import datetime, time as dt_time, timezone

# ================================================================
# 1. CONFIGURACIÓN
# ================================================================
st.set_page_config(page_title="TFG - Analizador OpenSky & Aeropuertos", layout="wide")

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
# 4. CONSULTAS A TRINO
# ================================================================
def consultar_state_vectors(fecha, hora, minuto, usuario, continentes):
    """
    Todos los aviones en vuelo en la zona y momento indicados.
    GROUP BY icao24 + MAX_BY → 1 posición por avión, sin LIMIT.
    """
    dt_utc    = datetime(fecha.year, fecha.month, fecha.day, hora, minuto, 0, tzinfo=timezone.utc)
    ts_exacto = int(dt_utc.timestamp())
    ts_hour   = ts_exacto - (ts_exacto % 3600)
    lat_min, lat_max, lon_min, lon_max = calcular_bbox(continentes)
    conn = get_conexion(usuario)

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
    return pd.DataFrame(rows, columns=cols).reset_index(drop=True)


def consultar_flights_data4(fecha, usuario, icao_origen=None, icao_destino=None):
    """
    Consulta flights_data4 para el día indicado.
    Devuelve un DataFrame con icao24, callsign, estdepartureairport, estarrivalairport.

    - Si icao_origen Y icao_destino: solo vuelos con ese origen Y ese destino.
    - Si solo icao_origen: todos los vuelos que salieron de ese aeropuerto.
    - Si solo icao_destino: todos los vuelos que llegaron a ese aeropuerto.
    - Nota: estarrivalairport puede ser NULL en muchos vuelos (OpenSky no siempre lo detecta).

    Usa partición 'day' (no 'hour') — obligatorio según la doc de OpenSky.
    No incluye la columna 'track' (array enorme, no la necesitamos).
    """
    dt_utc  = datetime(fecha.year, fecha.month, fecha.day, 0, 0, 0, tzinfo=timezone.utc)
    ts_day  = int(dt_utc.timestamp())
    conn    = get_conexion(usuario)

    # Construir filtros WHERE dinámicamente
    filtros = [f"day = {ts_day}"]
    if icao_origen:
        filtros.append(f"estdepartureairport = '{icao_origen}'")
    if icao_destino:
        filtros.append(f"estarrivalairport = '{icao_destino}'")

    where_clause = " AND ".join(filtros)

    query = f"""
        SELECT
            icao24,
            callsign,
            estdepartureairport,
            estarrivalairport,
            firstseen,
            lastseen
        FROM flights_data4
        WHERE {where_clause}
          AND icao24 IS NOT NULL
    """
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=cols)

    # Limpiar callsign
    if not df.empty:
        df['callsign'] = df['callsign'].fillna('').str.strip()

    return df


# ================================================================
# 5. BARRA LATERAL
# ================================================================
st.sidebar.header("🔑 Conexión OpenSky")
user_trino = st.sidebar.text_input(
    "Usuario OpenSky (email, minúsculas)", value="jaltevil@myuax.com"
).lower()

st.sidebar.divider()
st.sidebar.header("📅 Momento del análisis (UTC)")
fecha_sel  = st.sidebar.date_input("Día", datetime(2024, 1, 1))
hora_sel   = st.sidebar.selectbox("Hora (UTC)", list(range(24)), index=12)
minuto_sel = st.sidebar.selectbox("Minuto (UTC)", list(range(0, 60, 5)), index=0)

st.sidebar.divider()
st.sidebar.header("🗺️ Aeropuertos en mapa")
tipos_sel = st.sidebar.multiselect(
    "Tipos:", ['large_airport', 'medium_airport', 'small_airport'],
    default=['large_airport', 'medium_airport']
)
cont_sel = st.sidebar.multiselect(
    "Continentes:", sorted(df_base['continent'].unique().tolist()), default=['EU']
)

# Aeropuertos del área visible para los desplegables
df_aeros_visibles = df_base[
    df_base['continent'].isin(cont_sel) &
    df_base['type'].isin(tipos_sel)
].sort_values('name')

st.sidebar.divider()
st.sidebar.header("🛫 Origen")
opciones_origen = ["— Todos los orígenes —"] + df_aeros_visibles['name'].tolist()
origen_sel      = st.sidebar.selectbox("Aeropuerto de origen:", opciones_origen)
origen_activo   = origen_sel != "— Todos los orígenes —"
icao_origen     = None
if origen_activo:
    icao_origen = df_aeros_visibles[df_aeros_visibles['name'] == origen_sel]['ident'].values[0]
    st.sidebar.caption(f"ICAO: `{icao_origen}`")

st.sidebar.divider()
st.sidebar.header("🛬 Destino")
opciones_destino = ["— Todos los destinos —"] + df_aeros_visibles['name'].tolist()
destino_sel      = st.sidebar.selectbox("Aeropuerto de destino:", opciones_destino)
destino_activo   = destino_sel != "— Todos los destinos —"
icao_destino     = None
if destino_activo:
    icao_destino = df_aeros_visibles[df_aeros_visibles['name'] == destino_sel]['ident'].values[0]
    st.sidebar.caption(f"ICAO: `{icao_destino}`")
    st.sidebar.info(
        "⚠️ OpenSky no siempre detecta el aeropuerto de llegada. "
        "Algunos vuelos con ese destino pueden no aparecer si el campo `estarrivalairport` es nulo."
    )

st.sidebar.divider()
st.sidebar.caption("⚠️ Pulsa 'Consultar' cada vez que cambies fecha u hora.")
col1_btn, col2_btn = st.sidebar.columns(2)
with col1_btn:
    btn_consultar = st.button("✈️ Consultar OpenSky", use_container_width=True)
with col2_btn:
    btn_limpiar = st.button("🗑️ Limpiar mapa", use_container_width=True)


# ================================================================
# 6. ACCIONES DE BOTONES
# ================================================================
if btn_limpiar:
    for k in ['df_vuelos_raw', 'df_flights_info', 'vuelos_fecha', 'vuelos_hora', 'vuelos_minuto']:
        st.session_state[k] = None

if btn_consultar:
    if not user_trino:
        st.sidebar.error("Introduce tu usuario de OpenSky.")
    elif not cont_sel:
        st.sidebar.error("Selecciona al menos un continente.")
    else:
        # --- Query 1: posiciones en tiempo real ---
        with st.spinner("⏳ [1/2] Descargando posiciones de aviones (state_vectors)..."):
            try:
                df_raw = consultar_state_vectors(
                    fecha_sel, hora_sel, minuto_sel, user_trino, cont_sel
                )
                st.session_state['df_vuelos_raw'] = df_raw
                st.session_state['vuelos_fecha']  = str(fecha_sel)
                st.session_state['vuelos_hora']   = hora_sel
                st.session_state['vuelos_minuto'] = minuto_sel
                st.session_state['df_flights_info'] = None

                if df_raw.empty:
                    st.sidebar.warning("Sin posiciones. Prueba otra fecha/hora.")
                else:
                    st.sidebar.success(f"✅ {len(df_raw)} aviones descargados.")
            except Exception as e:
                st.sidebar.error(f"Error state_vectors: {e}")
                if "trino_conn" in st.session_state:
                    del st.session_state['trino_conn']

        # --- Query 2: flights_data4 (solo si hay filtro origen o destino) ---
        if (origen_activo or destino_activo) and st.session_state.get('df_vuelos_raw') is not None:
            label = []
            if origen_activo:  label.append(f"origen {icao_origen}")
            if destino_activo: label.append(f"destino {icao_destino}")

            with st.spinner(f"⏳ [2/2] Consultando vuelos con {' y '.join(label)} en flights_data4..."):
                try:
                    df_flights = consultar_flights_data4(
                        fecha_sel, user_trino,
                        icao_origen  = icao_origen  if origen_activo  else None,
                        icao_destino = icao_destino if destino_activo else None
                    )
                    st.session_state['df_flights_info'] = df_flights

                    if df_flights.empty:
                        st.sidebar.warning(
                            f"flights_data4 no encontró vuelos con esos filtros. "
                            f"Prueba otro aeropuerto o fecha."
                        )
                    else:
                        st.sidebar.info(f"🗓️ {len(df_flights)} vuelos en flights_data4 ese día.")
                except Exception as e:
                    st.sidebar.error(f"Error flights_data4: {e}")


# ================================================================
# 7. PREPARAR DATOS PARA EL MAPA
# ================================================================
df_vuelos_raw = st.session_state.get('df_vuelos_raw')
if df_vuelos_raw is None:
    df_vuelos_raw = pd.DataFrame()

df_flights_info = st.session_state.get('df_flights_info')
if df_flights_info is None:
    df_flights_info = pd.DataFrame()

df_vuelos = df_vuelos_raw.copy()

# Filtrar por icao24 de flights_data4 (si hay filtro origen o destino activo)
if not df_vuelos.empty and (origen_activo or destino_activo) and not df_flights_info.empty:
    icao24_validos = set(df_flights_info['icao24'].unique())
    df_vuelos = df_vuelos[df_vuelos['icao24'].isin(icao24_validos)].reset_index(drop=True)

# Enriquecer df_vuelos con info de origen/destino de flights_data4
if not df_vuelos.empty and not df_flights_info.empty:
    # Quedarse con la fila más reciente por icao24 (lastseen más alto)
    df_flights_dedup = df_flights_info.sort_values('lastseen').drop_duplicates('icao24', keep='last')
    cols_join = ['icao24', 'estdepartureairport', 'estarrivalairport']
    df_vuelos = df_vuelos.merge(
        df_flights_dedup[cols_join],
        on='icao24', how='left'
    )

mask    = df_base['continent'].isin(cont_sel) & df_base['type'].isin(tipos_sel)
df_view = df_base[mask]


# ================================================================
# 8. TÍTULO E INFO BANNER
# ================================================================
st.title("🌍 TFG: Análisis de Infraestructura y Flujos Aéreos")

if not df_vuelos_raw.empty:
    f_label = st.session_state.get('vuelos_fecha', '')
    h_label = st.session_state.get('vuelos_hora', 0)
    m_label = st.session_state.get('vuelos_minuto', 0)
    ts_str  = f"**{f_label}** a las **{h_label:02d}:{m_label:02d} UTC**"

    partes = [f"📦 **{len(df_vuelos_raw)}** aviones descargados · {ts_str}"]
    if origen_activo:
        partes.append(f"🛫 Origen: **{origen_sel}** (`{icao_origen}`)")
    if destino_activo:
        partes.append(f"🛬 Destino: **{destino_sel}** (`{icao_destino}`)")
    partes.append(f"✈️ Mostrando **{len(df_vuelos)}** aviones")
    st.info(" · ".join(partes))


# ================================================================
# 9. MAPA PRINCIPAL
# ================================================================
fig = go.Figure()

colores = {"large_airport": "#FF4B4B", "medium_airport": "#1C83E1", "small_airport": "#00FF7F"}
tamanos = {"large_airport": 10,        "medium_airport": 7,         "small_airport": 4}

for tipo in tipos_sel:
    df_t = df_view[df_view['type'] == tipo]
    if df_t.empty:
        continue
    fig.add_trace(go.Scattermap(
        lat=df_t['latitude_deg'], lon=df_t['longitude_deg'],
        mode='markers', name=tipo,
        marker=go.scattermap.Marker(size=tamanos[tipo], color=colores[tipo], opacity=0.6),
        text=df_t['name'], hoverinfo='text'
    ))

# Aeropuerto origen destacado (cian)
if origen_activo:
    fo = df_base[df_base['name'] == origen_sel]
    if not fo.empty:
        fig.add_trace(go.Scattermap(
            lat=fo['latitude_deg'], lon=fo['longitude_deg'],
            mode='markers+text', name=f"🛫 {icao_origen}",
            text=[icao_origen], textposition="top right",
            marker=go.scattermap.Marker(size=22, color='cyan'),
            hovertext=[f"🛫 ORIGEN: {origen_sel} ({icao_origen})"],
            hoverinfo='text'
        ))

# Aeropuerto destino destacado (naranja)
if destino_activo:
    fd = df_base[df_base['name'] == destino_sel]
    if not fd.empty:
        fig.add_trace(go.Scattermap(
            lat=fd['latitude_deg'], lon=fd['longitude_deg'],
            mode='markers+text', name=f"🛬 {icao_destino}",
            text=[icao_destino], textposition="top right",
            marker=go.scattermap.Marker(size=22, color='orange'),
            hovertext=[f"🛬 DESTINO: {destino_sel} ({icao_destino})"],
            hoverinfo='text'
        ))

# Aviones
if not df_vuelos.empty:
    dv = df_vuelos.copy()
    dv['callsign']     = dv['callsign'].fillna('').str.strip()
    dv['velocity']     = pd.to_numeric(dv['velocity'],     errors='coerce').fillna(0)
    dv['baroaltitude'] = pd.to_numeric(dv['baroaltitude'], errors='coerce').fillna(0)
    dv['heading']      = pd.to_numeric(dv['heading'],      errors='coerce').fillna(0)

    vel_kmh = (dv['velocity'] * 3.6).round(0).astype(int).astype(str)
    alt_ft  = (dv['baroaltitude'] * 3.281).round(0).astype(int).astype(str)

    hover = "✈️ <b>" + dv['callsign'] + "</b><br>" + "ICAO24: " + dv['icao24'] + "<br>"

    # Añadir origen y destino al hover si están disponibles
    if 'estdepartureairport' in dv.columns:
        hover = hover + "Origen: " + dv['estdepartureairport'].fillna('desconocido') + "<br>"
    if 'estarrivalairport' in dv.columns:
        hover = hover + "Destino: " + dv['estarrivalairport'].fillna('desconocido') + "<br>"

    hover = (
        hover +
        "Vel: "   + vel_kmh + " km/h<br>" +
        "Alt: "   + alt_ft  + " ft<br>"   +
        "Rumbo: " + dv['heading'].round(0).astype(int).astype(str) + "°"
    )

    fig.add_trace(go.Scattermap(
        lat=dv['lat'], lon=dv['lon'],
        mode='markers', name=f"✈️ Aviones ({len(dv)})",
        marker=go.scattermap.Marker(size=10, color='yellow'),
        text=hover, hoverinfo='text'
    ))

# Centro del mapa
if destino_activo:
    fd = df_base[df_base['name'] == destino_sel]
    map_center = dict(lat=fd['latitude_deg'].values[0], lon=fd['longitude_deg'].values[0]) if not fd.empty else dict(lat=40, lon=-3)
elif origen_activo:
    fo = df_base[df_base['name'] == origen_sel]
    map_center = dict(lat=fo['latitude_deg'].values[0], lon=fo['longitude_deg'].values[0]) if not fo.empty else dict(lat=40, lon=-3)
elif not df_view.empty:
    map_center = dict(lat=df_view['latitude_deg'].mean(), lon=df_view['longitude_deg'].mean())
else:
    map_center = dict(lat=40, lon=-3)

fig.update_layout(
    map_style="carto-darkmatter",
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    height=720, showlegend=True,
    legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                bgcolor="rgba(0,0,0,0.6)", font=dict(color="white")),
    map=dict(center=map_center, zoom=3.5)
)
st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})


# ================================================================
# 10. TABLA DE VUELOS
# ================================================================
if not df_vuelos.empty:
    st.divider()
    st.subheader(f"📋 Vuelos en pantalla ({len(df_vuelos)})")

    dt = df_vuelos.copy()
    dt['callsign'] = dt['callsign'].fillna('').str.strip()
    dt['vel_kmh']  = (pd.to_numeric(dt['velocity'],     errors='coerce').fillna(0) * 3.6).round(0).astype(int)
    dt['alt_ft']   = (pd.to_numeric(dt['baroaltitude'], errors='coerce').fillna(0) * 3.281).round(0).astype(int)
    dt['heading']  = pd.to_numeric(dt['heading'], errors='coerce').fillna(0).round(1)
    dt['lat']      = dt['lat'].round(4)
    dt['lon']      = dt['lon'].round(4)

    cols_t  = ['callsign', 'icao24', 'lat', 'lon', 'vel_kmh', 'alt_ft', 'heading']
    nombres = {
        'callsign': 'Vuelo', 'icao24': 'ICAO24',
        'lat': 'Latitud', 'lon': 'Longitud',
        'vel_kmh': 'Vel (km/h)', 'alt_ft': 'Altitud (ft)', 'heading': 'Rumbo (°)'
    }

    if 'estdepartureairport' in dt.columns:
        cols_t.insert(2, 'estdepartureairport')
        nombres['estdepartureairport'] = 'Origen'
    if 'estarrivalairport' in dt.columns:
        cols_t.insert(3, 'estarrivalairport')
        nombres['estarrivalairport'] = 'Destino'

    st.dataframe(dt[cols_t].rename(columns=nombres), use_container_width=True)


# ================================================================
# 11. ANÁLISIS DE AISLAMIENTO GEOGRÁFICO
# ================================================================
st.divider()
st.subheader("📊 Análisis de Aislamiento Geográfico")

col_ctrl1, col_ctrl2 = st.columns([1, 2])
with col_ctrl1:
    dist_min = st.slider(
        "Mostrar aeropuertos a más de X km del más cercano:",
        min_value=0, max_value=2000, value=100, step=25
    )
    tipos_ais = st.multiselect(
        "Tipos a analizar:",
        ['large_airport', 'medium_airport', 'small_airport'],
        default=['large_airport', 'medium_airport', 'small_airport'],
        key="tipos_aislamiento"
    )

with col_ctrl2:
    df_ais_base = df_view[df_view['type'].isin(tipos_ais)] if tipos_ais else df_view
    df_aislados = df_ais_base[df_ais_base['distancia_vecino_km'] >= dist_min].sort_values(
        'distancia_vecino_km', ascending=False
    )
    c1, c2 = st.columns(2)
    c1.metric("Aeropuertos en selección", len(df_ais_base))
    c2.metric(f"Aislados (>{dist_min} km)", len(df_aislados))

if not df_ais_base.empty:
    fig_hist = px.histogram(
        df_ais_base, x='distancia_vecino_km', nbins=50,
        title="Distribución de distancias al aeropuerto más cercano",
        labels={'distancia_vecino_km': 'Distancia al vecino (km)'},
        color_discrete_sequence=['#1C83E1']
    )
    fig_hist.add_vline(x=dist_min, line_dash="dash", line_color="red",
                       annotation_text=f"Umbral: {dist_min} km", annotation_position="top right")
    fig_hist.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='white', height=280, margin=dict(t=40, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

if not df_aislados.empty:
    col_map, col_tab = st.columns([3, 2])
    with col_map:
        fig_ais = px.scatter_map(
            df_ais_base, lat='latitude_deg', lon='longitude_deg',
            color='distancia_vecino_km', size='distancia_vecino_km', size_max=18,
            color_continuous_scale='RdYlGn_r',
            hover_name='name',
            hover_data={'ident': True, 'municipality': True,
                        'distancia_vecino_km': ':.1f', 'type': True,
                        'latitude_deg': False, 'longitude_deg': False},
            labels={'distancia_vecino_km': 'Dist. vecino (km)'},
            title="Aeropuertos por nivel de aislamiento (rojo = más aislado)",
            map_style='carto-darkmatter', zoom=2,
        )
        fig_ais.update_layout(height=420, margin=dict(r=0, t=40, l=0, b=0),
                              coloraxis_colorbar=dict(title="km", thickness=12))
        st.plotly_chart(fig_ais, use_container_width=True)

    with col_tab:
        st.write(f"**Top aeropuertos más aislados** (>{dist_min} km):")
        st.dataframe(
            df_aislados[['ident', 'name', 'municipality', 'type', 'distancia_vecino_km']].head(20).rename(columns={
                'ident': 'ICAO', 'name': 'Nombre', 'municipality': 'Ciudad',
                'type': 'Tipo', 'distancia_vecino_km': 'Dist. vecino (km)'
            }),
            use_container_width=True, height=380
        )