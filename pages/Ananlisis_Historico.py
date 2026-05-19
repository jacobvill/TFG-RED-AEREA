"""
pages/2_📅_Análisis_Histórico.py
TFG: Consulta histórica via Trino (state_vectors + flights_data4)
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.neighbors import BallTree
from trino.dbapi import connect
from trino.auth import OAuth2Authentication
from datetime import datetime, time as dt_time, timezone

st.set_page_config(page_title="Análisis Histórico", page_icon="📅", layout="wide")
st.title("📅 Análisis Histórico de Tráfico Aéreo")
st.caption("Consultas SQL sobre OpenSky via Trino · state_vectors_data4 + flights_data4")

# ================================================================
# CONSTANTES
# ================================================================
BBOXES = {
    'EU': (-25.0, 29.0,  45.0, 81.2),
    'NA': (-176.6, 7.4, -52.0, 83.1),
    'SA': (-109.4,-55.0, -32.4, 12.4),
    'AS': (  26.0,-12.2, 180.0, 80.8),
    'AF': ( -25.1,-34.8,  63.4, 37.2),
    'OC': ( 110.0,-46.9, 180.0, 28.2),
}

# ================================================================
# CARGA DE AEROPUERTOS
# ================================================================
@st.cache_data
def cargar_aeropuertos():
    df = pd.read_csv("airports.csv")
    df = df[df["type"].isin(["small_airport","medium_airport","large_airport"])].copy()
    df = df.dropna(subset=["latitude_deg","longitude_deg"])
    df["continent"] = df["continent"].fillna("NA")
    coords = np.radians(df[["latitude_deg","longitude_deg"]].values)
    tree = BallTree(coords, metric="haversine")
    dists, _ = tree.query(coords, k=2)
    df["distancia_vecino_km"] = (dists[:, 1] * 6371).round(1)
    return df

df_base = cargar_aeropuertos()

# ================================================================
# CONEXIÓN TRINO
# ================================================================
def get_trino(usuario):
    if "trino_conn" not in st.session_state or st.session_state.get("trino_user") != usuario:
        st.session_state.trino_conn = connect(
            host="trino.opensky-network.org", port=443,
            user=usuario, auth=OAuth2Authentication(),
            http_scheme="https", catalog="minio", schema="osky",
            request_timeout=120.0
        )
        st.session_state.trino_user = usuario
    return st.session_state.trino_conn

def calcular_bbox(continentes):
    lon_min, lat_min, lon_max, lat_max = 180, 90, -180, -90
    for c in continentes:
        if c in BBOXES:
            a, b, c2, d = BBOXES[c]
            lon_min = min(lon_min, a); lat_min = min(lat_min, b)
            lon_max = max(lon_max, c2); lat_max = max(lat_max, d)
    return lat_min-2, lat_max+2, lon_min-2, lon_max+2

# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.header("🔑 Conexión Trino / OpenSky")
user_trino = st.sidebar.text_input("Usuario (email)", value="jaltevil@myuax.com").lower()

st.sidebar.divider()
st.sidebar.header("📅 Momento (UTC)")
fecha_sel  = st.sidebar.date_input("Día", datetime(2024, 1, 16))
hora_sel   = st.sidebar.selectbox("Hora", list(range(24)), index=12)
minuto_sel = st.sidebar.selectbox("Minuto", list(range(0, 60, 5)), index=0)

st.sidebar.divider()
st.sidebar.header("🗺️ Aeropuertos en mapa")
tipos_sel = st.sidebar.multiselect(
    "Tipos:", ["large_airport","medium_airport","small_airport"],
    default=["large_airport","medium_airport"]
)
cont_sel = st.sidebar.multiselect(
    "Continentes:", sorted(df_base["continent"].unique().tolist()), default=["EU"]
)

# Filtros origen / destino
df_aeros_vis = df_base[
    df_base["continent"].isin(cont_sel) & df_base["type"].isin(tipos_sel)
].sort_values("name")

st.sidebar.divider()
st.sidebar.header("🛫 Filtro origen")
origen_opts = ["— Todos —"] + df_aeros_vis["name"].tolist()
origen_sel  = st.sidebar.selectbox("Origen:", origen_opts)
origen_act  = origen_sel != "— Todos —"
icao_origen = df_aeros_vis[df_aeros_vis["name"]==origen_sel]["ident"].values[0] if origen_act else None

st.sidebar.header("🛬 Filtro destino")
dest_opts  = ["— Todos —"] + df_aeros_vis["name"].tolist()
dest_sel   = st.sidebar.selectbox("Destino:", dest_opts)
dest_act   = dest_sel != "— Todos —"
icao_dest  = df_aeros_vis[df_aeros_vis["name"]==dest_sel]["ident"].values[0] if dest_act else None
if dest_act:
    st.sidebar.caption("⚠️ OpenSky no siempre detecta el destino: algunos vuelos pueden no aparecer.")

st.sidebar.divider()
c1b, c2b = st.sidebar.columns(2)
btn_consultar = c1b.button("✈️ Consultar", use_container_width=True, type="primary")
btn_limpiar   = c2b.button("🗑️ Limpiar",  use_container_width=True)
st.sidebar.caption("⚠️ Pulsa Consultar cada vez que cambies fecha u hora.")

# ================================================================
# QUERIES
# ================================================================
def query_state_vectors(fecha, hora, minuto, usuario, continentes):
    dt_utc    = datetime(fecha.year, fecha.month, fecha.day, hora, minuto, 0, tzinfo=timezone.utc)
    ts        = int(dt_utc.timestamp())
    ts_hour   = ts - (ts % 3600)
    lat_min, lat_max, lon_min, lon_max = calcular_bbox(continentes)
    conn = get_trino(usuario)
    q = f"""
        SELECT icao24,
               MAX_BY(callsign,     time) AS callsign,
               MAX_BY(lat,          time) AS lat,
               MAX_BY(lon,          time) AS lon,
               MAX_BY(velocity,     time) AS velocity,
               MAX_BY(heading,      time) AS heading,
               MAX_BY(baroaltitude, time) AS baroaltitude
        FROM state_vectors_data4
        WHERE hour     = {ts_hour}
          AND time     BETWEEN {ts} AND {ts} + 60
          AND onground = false
          AND lat      BETWEEN {lat_min} AND {lat_max}
          AND lon      BETWEEN {lon_min} AND {lon_max}
          AND lat IS NOT NULL AND lon IS NOT NULL
        GROUP BY icao24
    """
    cur = conn.cursor(); cur.execute(q)
    rows = cur.fetchall(); cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols).reset_index(drop=True)


def query_flights_data4(fecha, usuario, icao_origen=None, icao_destino=None):
    dt_utc  = datetime(fecha.year, fecha.month, fecha.day, 0, 0, 0, tzinfo=timezone.utc)
    ts_day  = int(dt_utc.timestamp())
    conn    = get_trino(usuario)
    filtros = [f"day = {ts_day}", "icao24 IS NOT NULL"]
    if icao_origen:  filtros.append(f"estdepartureairport = '{icao_origen}'")
    if icao_destino: filtros.append(f"estarrivalairport   = '{icao_destino}'")
    q = f"""
        SELECT icao24, TRIM(callsign) AS callsign,
               estdepartureairport AS origen, estarrivalairport AS destino
        FROM flights_data4
        WHERE {' AND '.join(filtros)}
    """
    cur = conn.cursor(); cur.execute(q)
    rows = cur.fetchall(); cols = [d[0] for d in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["callsign"] = df["callsign"].fillna("").str.strip()
    return df

# ================================================================
# ACCIONES
# ================================================================
if btn_limpiar:
    for k in ["hist_vuelos","hist_flights","hist_fecha","hist_hora","hist_minuto"]:
        st.session_state[k] = None

if btn_consultar:
    if not user_trino:
        st.sidebar.error("Introduce tu usuario.")
    elif not cont_sel:
        st.sidebar.error("Selecciona al menos un continente.")
    else:
        with st.spinner("⏳ [1/2] Descargando posiciones (state_vectors)..."):
            try:
                df_sv = query_state_vectors(fecha_sel, hora_sel, minuto_sel, user_trino, cont_sel)
                st.session_state["hist_vuelos"]  = df_sv
                st.session_state["hist_fecha"]   = str(fecha_sel)
                st.session_state["hist_hora"]    = hora_sel
                st.session_state["hist_minuto"]  = minuto_sel
                st.session_state["hist_flights"] = None
                if df_sv.empty:
                    st.sidebar.warning("Sin resultados. Prueba otra hora.")
                else:
                    st.sidebar.success(f"✅ {len(df_sv):,} aviones")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
                if "trino_conn" in st.session_state:
                    del st.session_state["trino_conn"]

        if (origen_act or dest_act) and st.session_state.get("hist_vuelos") is not None:
            with st.spinner("⏳ [2/2] Filtrando por origen/destino (flights_data4)..."):
                try:
                    df_fl = query_flights_data4(fecha_sel, user_trino, icao_origen, icao_dest)
                    st.session_state["hist_flights"] = df_fl
                    st.sidebar.info(f"🗓️ {len(df_fl):,} vuelos con esos filtros ese día.")
                except Exception as e:
                    st.sidebar.error(f"Error flights_data4: {e}")

# ================================================================
# PREPARAR DATOS
# ================================================================
df_sv      = st.session_state.get("hist_vuelos")
df_fl      = st.session_state.get("hist_flights")
if df_sv is None: df_sv = pd.DataFrame()
if df_fl is None: df_fl = pd.DataFrame()

df_vuelos = df_sv.copy()

# Aplicar filtro origen/destino
if not df_vuelos.empty and (origen_act or dest_act) and not df_fl.empty:
    icao24_ok = set(df_fl["icao24"].unique())
    df_vuelos = df_vuelos[df_vuelos["icao24"].isin(icao24_ok)].reset_index(drop=True)

# Enriquecer con origen/destino
if not df_vuelos.empty and not df_fl.empty:
    df_fl_dedup = df_fl.drop_duplicates("icao24", keep="last")
    df_vuelos = df_vuelos.merge(df_fl_dedup[["icao24","origen","destino"]], on="icao24", how="left")

mask    = df_base["continent"].isin(cont_sel) & df_base["type"].isin(tipos_sel)
df_view = df_base[mask]

# ================================================================
# BANNER
# ================================================================
st.title("📅 Análisis Histórico")

if not df_sv.empty:
    f_label = st.session_state.get("hist_fecha","")
    h_label = st.session_state.get("hist_hora", 0)
    m_label = st.session_state.get("hist_minuto", 0)
    partes = [f"📦 **{len(df_sv):,}** aviones descargados · **{f_label} {h_label:02d}:{m_label:02d} UTC**"]
    if origen_act: partes.append(f"🛫 Origen: **{origen_sel}**")
    if dest_act:   partes.append(f"🛬 Destino: **{dest_sel}**")
    partes.append(f"✈️ Mostrando: **{len(df_vuelos):,}**")
    st.info(" · ".join(partes))

    # Botones de acción
    ca, cb = st.columns(2)
    with ca:
        if st.button("🔬 Enviar al Simulador", use_container_width=True, type="primary"):
            st.session_state["datos_sim"] = {
                "fuente":     "historico",
                "label":      f"Histórico · {f_label} {h_label:02d}:{m_label:02d} UTC · {', '.join(cont_sel)}",
                "df":         df_vuelos.copy(),
                "fecha":      st.session_state.get("hist_fecha"),
                "continente": cont_sel[0] if len(cont_sel)==1 else "EU"
            }
            st.success("✅ Datos enviados. Ve a **Simulador** en el menú lateral.")
    with cb:
        csv = df_vuelos.to_csv(index=False)
        st.download_button(
            "⬇️ Descargar CSV",
            data=csv,
            file_name=f"historico_{f_label}_{h_label:02d}{m_label:02d}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ================================================================
# MAPA
# ================================================================
COLORES = {"large_airport":"#FF4B4B","medium_airport":"#1C83E1","small_airport":"#00FF7F"}
TAMANOS = {"large_airport":10,"medium_airport":7,"small_airport":4}

fig = go.Figure()

for tipo in tipos_sel:
    df_t = df_view[df_view["type"]==tipo]
    if df_t.empty: continue
    fig.add_trace(go.Scattermap(
        lat=df_t["latitude_deg"], lon=df_t["longitude_deg"],
        mode="markers", name=tipo,
        marker=go.scattermap.Marker(size=TAMANOS[tipo], color=COLORES[tipo], opacity=0.6),
        text=df_t["name"], hoverinfo="text"
    ))

if origen_act:
    fo = df_base[df_base["name"]==origen_sel]
    if not fo.empty:
        fig.add_trace(go.Scattermap(
            lat=fo["latitude_deg"], lon=fo["longitude_deg"],
            mode="markers+text", name=f"🛫 {icao_origen}",
            text=[icao_origen], textposition="top right",
            marker=go.scattermap.Marker(size=22, color="cyan"),
            hovertext=[f"🛫 ORIGEN: {origen_sel}"], hoverinfo="text"
        ))

if dest_act:
    fd = df_base[df_base["name"]==dest_sel]
    if not fd.empty:
        fig.add_trace(go.Scattermap(
            lat=fd["latitude_deg"], lon=fd["longitude_deg"],
            mode="markers+text", name=f"🛬 {icao_dest}",
            text=[icao_dest], textposition="top right",
            marker=go.scattermap.Marker(size=22, color="orange"),
            hovertext=[f"🛬 DESTINO: {dest_sel}"], hoverinfo="text"
        ))

if not df_vuelos.empty:
    dv = df_vuelos.copy()
    dv["callsign"]     = dv["callsign"].fillna("").str.strip()
    dv["velocity"]     = pd.to_numeric(dv["velocity"],     errors="coerce").fillna(0)
    dv["baroaltitude"] = pd.to_numeric(dv["baroaltitude"], errors="coerce").fillna(0)
    dv["heading"]      = pd.to_numeric(dv["heading"],      errors="coerce").fillna(0)
    vel_kmh = (dv["velocity"] * 3.6).round(0).astype(int).astype(str)
    alt_ft  = (dv["baroaltitude"] * 3.281).round(0).astype(int).astype(str)
    hover   = ("✈️ <b>" + dv["callsign"] + "</b><br>" +
               "ICAO24: " + dv["icao24"] + "<br>")
    if "origen" in dv.columns:
        hover += "Origen: "  + dv["origen"].fillna("?") + "<br>"
        hover += "Destino: " + dv["destino"].fillna("?") + "<br>"
    hover += "Vel: " + vel_kmh + " km/h  |  Alt: " + alt_ft + " ft"

    fig.add_trace(go.Scattermap(
        lat=dv["lat"], lon=dv["lon"], mode="markers",
        name=f"✈️ Aviones ({len(dv):,})",
        marker=go.scattermap.Marker(size=9, color="yellow"),
        text=hover, hoverinfo="text"
    ))

# Centro del mapa
if dest_act:
    fd = df_base[df_base["name"]==dest_sel]
    mc = dict(lat=fd["latitude_deg"].values[0], lon=fd["longitude_deg"].values[0]) if not fd.empty else dict(lat=40, lon=-3)
elif origen_act:
    fo = df_base[df_base["name"]==origen_sel]
    mc = dict(lat=fo["latitude_deg"].values[0], lon=fo["longitude_deg"].values[0]) if not fo.empty else dict(lat=40, lon=-3)
elif not df_view.empty:
    mc = dict(lat=df_view["latitude_deg"].mean(), lon=df_view["longitude_deg"].mean())
else:
    mc = dict(lat=40, lon=-3)

fig.update_layout(
    map_style="carto-darkmatter",
    margin={"r":0,"t":0,"l":0,"b":0}, height=680,
    showlegend=True,
    legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                bgcolor="rgba(0,0,0,0.6)", font=dict(color="white")),
    map=dict(center=mc, zoom=3.5)
)
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

# ================================================================
# TABLA DE VUELOS
# ================================================================
if not df_vuelos.empty:
    st.divider()
    st.subheader(f"📋 Vuelos en pantalla ({len(df_vuelos):,})")
    dt = df_vuelos.copy()
    dt["callsign"] = dt["callsign"].fillna("").str.strip()
    dt["vel_kmh"]  = (pd.to_numeric(dt["velocity"],     errors="coerce").fillna(0) * 3.6).round(0).astype(int)
    dt["alt_ft"]   = (pd.to_numeric(dt["baroaltitude"], errors="coerce").fillna(0) * 3.281).round(0).astype(int)
    dt["lat"]      = dt["lat"].round(4)
    dt["lon"]      = dt["lon"].round(4)

    cols_t  = ["callsign","icao24","lat","lon","vel_kmh","alt_ft"]
    nombres = {"callsign":"Vuelo","icao24":"ICAO24","lat":"Lat","lon":"Lon",
               "vel_kmh":"Vel (km/h)","alt_ft":"Alt (ft)"}
    if "origen"  in dt.columns: cols_t.insert(2,"origen");  nombres["origen"]  = "Origen"
    if "destino" in dt.columns: cols_t.insert(3,"destino"); nombres["destino"] = "Destino"

    st.dataframe(dt[cols_t].rename(columns=nombres), use_container_width=True)

# ================================================================
# ANÁLISIS DE AISLAMIENTO
# ================================================================
st.divider()
st.subheader("📊 Análisis de Aislamiento Geográfico")

import plotly.express as px

col_ctrl, col_stats = st.columns([1,2])
with col_ctrl:
    dist_min   = st.slider("Aeropuertos a más de X km del más cercano:", 0, 2000, 100, 25)
    tipos_ais  = st.multiselect("Tipos:", ["large_airport","medium_airport","small_airport"],
                                default=["large_airport","medium_airport","small_airport"],
                                key="tipos_ais")
with col_stats:
    df_ais  = df_view[df_view["type"].isin(tipos_ais)] if tipos_ais else df_view
    df_isol = df_ais[df_ais["distancia_vecino_km"] >= dist_min].sort_values("distancia_vecino_km", ascending=False)
    m1, m2 = st.columns(2)
    m1.metric("Aeropuertos en selección", f"{len(df_ais):,}")
    m2.metric(f"Aislados (>{dist_min} km)", f"{len(df_isol):,}")

if not df_ais.empty:
    fig_h = px.histogram(df_ais, x="distancia_vecino_km", nbins=50,
                         color_discrete_sequence=["#1C83E1"],
                         labels={"distancia_vecino_km":"Distancia al vecino (km)"})
    fig_h.add_vline(x=dist_min, line_dash="dash", line_color="red",
                    annotation_text=f"Umbral: {dist_min} km", annotation_position="top right")
    fig_h.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font_color="white", height=260, margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig_h, use_container_width=True)

if not df_isol.empty:
    cm, ct = st.columns([3,2])
    with cm:
        fig_m = px.scatter_map(
            df_ais, lat="latitude_deg", lon="longitude_deg",
            color="distancia_vecino_km", size="distancia_vecino_km", size_max=18,
            color_continuous_scale="RdYlGn_r",
            hover_name="name",
            hover_data={"ident":True,"municipality":True,"distancia_vecino_km":":.1f",
                        "latitude_deg":False,"longitude_deg":False},
            map_style="carto-darkmatter", zoom=2
        )
        fig_m.update_layout(height=400, margin=dict(r=0,t=0,l=0,b=0),
                             coloraxis_colorbar=dict(title="km",thickness=12))
        st.plotly_chart(fig_m, use_container_width=True)
    with ct:
        st.dataframe(
            df_isol[["ident","name","municipality","type","distancia_vecino_km"]].head(20).rename(columns={
                "ident":"ICAO","name":"Aeropuerto","municipality":"Ciudad",
                "type":"Tipo","distancia_vecino_km":"Dist. vecino (km)"
            }),
            use_container_width=True, height=380
        )