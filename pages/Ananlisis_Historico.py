"""
pages/2_Analisis_Historico.py
TFG: Consulta histórica via Trino + trayectoria de vuelo
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.neighbors import BallTree
from trino.dbapi import connect
from trino.auth import OAuth2Authentication
from datetime import datetime, time as dt_time, timezone

st.set_page_config(page_title="Análisis Histórico", page_icon="📅", layout="wide")

BBOXES = {
    "EU": (-25.0,29.0,45.0,81.2), "NA": (-176.6,7.4,-52.0,83.1),
    "SA": (-109.4,-55.0,-32.4,12.4), "AS": (26.0,-12.2,180.0,80.8),
    "AF": (-25.1,-34.8,63.4,37.2),  "OC": (110.0,-46.9,180.0,28.2),
}

@st.cache_data
def cargar_aeropuertos():
    df = pd.read_csv("airports.csv")
    df = df[df["type"].isin(["small_airport","medium_airport","large_airport"])].copy()
    df = df.dropna(subset=["latitude_deg","longitude_deg"])
    df["continent"] = df["continent"].fillna("NA")
    coords = np.radians(df[["latitude_deg","longitude_deg"]].values)
    tree = BallTree(coords, metric="haversine")
    dists, _ = tree.query(coords, k=2)
    df["distancia_vecino_km"] = (dists[:,1]*6371).round(1)
    return df

df_base = cargar_aeropuertos()

def get_trino(usuario):
    if "trino_conn" not in st.session_state or st.session_state.get("trino_user") != usuario:
        st.session_state.trino_conn = connect(
            host="trino.opensky-network.org", port=443,
            user=usuario, auth=OAuth2Authentication(),
            http_scheme="https", catalog="minio", schema="osky", request_timeout=120.0
        )
        st.session_state.trino_user = usuario
    return st.session_state.trino_conn

def calcular_bbox(continentes):
    lon_min,lat_min,lon_max,lat_max = 180,90,-180,-90
    for c in continentes:
        if c in BBOXES:
            a,b,c2,d = BBOXES[c]
            lon_min=min(lon_min,a); lat_min=min(lat_min,b)
            lon_max=max(lon_max,c2); lat_max=max(lat_max,d)
    return lat_min-2, lat_max+2, lon_min-2, lon_max+2

def query_state_vectors(fecha, hora, minuto, usuario, continentes):
    dt_utc = datetime(fecha.year,fecha.month,fecha.day,hora,minuto,0,tzinfo=timezone.utc)
    ts = int(dt_utc.timestamp())
    ts_hour = ts - (ts%3600)
    lat_min,lat_max,lon_min,lon_max = calcular_bbox(continentes)
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
        WHERE hour={ts_hour} AND time BETWEEN {ts} AND {ts}+60
          AND onground=false
          AND lat BETWEEN {lat_min} AND {lat_max}
          AND lon BETWEEN {lon_min} AND {lon_max}
          AND lat IS NOT NULL AND lon IS NOT NULL
        GROUP BY icao24
    """
    cur=conn.cursor(); cur.execute(q)
    rows=cur.fetchall(); cols=[d[0] for d in cur.description]
    return pd.DataFrame(rows,columns=cols).reset_index(drop=True)

def query_flights_data4(fecha, usuario, icao_origen=None, icao_destino=None):
    dt_utc = datetime(fecha.year,fecha.month,fecha.day,0,0,0,tzinfo=timezone.utc)
    ts_day = int(dt_utc.timestamp())
    conn = get_trino(usuario)
    filtros = [f"day={ts_day}","icao24 IS NOT NULL"]
    if icao_origen:  filtros.append(f"estdepartureairport='{icao_origen}'")
    if icao_destino: filtros.append(f"estarrivalairport='{icao_destino}'")
    q = f"""
        SELECT icao24, TRIM(callsign) AS callsign,
               estdepartureairport AS origen, estarrivalairport AS destino
        FROM flights_data4 WHERE {' AND '.join(filtros)}
    """
    cur=conn.cursor(); cur.execute(q)
    rows=cur.fetchall(); cols=[d[0] for d in cur.description]
    df = pd.DataFrame(rows,columns=cols)
    if not df.empty: df["callsign"]=df["callsign"].fillna("").str.strip()
    return df

def query_trayectoria(icao24, fecha, hora, usuario):
    """
    Carga la trayectoria real de un avión en una ventana de ±2 horas.
    Filtra siempre por la partición 'hour' según las reglas de OpenSky.
    """
    dt_utc = datetime(fecha.year,fecha.month,fecha.day,hora,0,0,tzinfo=timezone.utc)
    ts_center = int(dt_utc.timestamp())
    ts_start  = ts_center - 2*3600
    ts_end    = ts_center + 2*3600

    # Calcular todas las horas afectadas (partición obligatoria)
    hours = set()
    t = ts_start - (ts_start%3600)
    while t <= ts_end:
        hours.add(t); t += 3600
    hours_str = ",".join(str(h) for h in sorted(hours))

    conn = get_trino(usuario)
    q = f"""
        SELECT time, lat, lon, baroaltitude, velocity, heading
        FROM state_vectors_data4
        WHERE hour IN ({hours_str})
          AND time BETWEEN {ts_start} AND {ts_end}
          AND icao24='{icao24}'
          AND lat IS NOT NULL AND lon IS NOT NULL
        ORDER BY time
    """
    cur=conn.cursor(); cur.execute(q)
    rows=cur.fetchall(); cols=[d[0] for d in cur.description]
    return pd.DataFrame(rows,columns=cols)

# ── SIDEBAR ──────────────────────────────────────────────────────
st.sidebar.header("🔑 Conexión Trino")
user_trino = st.sidebar.text_input("Usuario (email)",value="jaltevil@myuax.com").lower()

st.sidebar.divider()
st.sidebar.header("📅 Momento (UTC)")
fecha_sel  = st.sidebar.date_input("Día", datetime(2024,1,16))
hora_sel   = st.sidebar.selectbox("Hora",list(range(24)),index=12)
minuto_sel = st.sidebar.selectbox("Minuto",list(range(0,60,5)),index=0)

st.sidebar.divider()
st.sidebar.header("🗺️ Aeropuertos en mapa")
tipos_sel = st.sidebar.multiselect("Tipos:",["large_airport","medium_airport","small_airport"],
                                   default=["large_airport","medium_airport"])
cont_sel  = st.sidebar.multiselect("Continentes:",
                                   sorted(df_base["continent"].unique().tolist()),default=["EU"])

df_aeros = df_base[df_base["continent"].isin(cont_sel) & df_base["type"].isin(tipos_sel)].sort_values("name")

st.sidebar.divider()
st.sidebar.header("🛫 Filtros de ruta")
origen_sel  = st.sidebar.selectbox("Origen:",  ["— Todos —"]+df_aeros["name"].tolist())
destino_sel = st.sidebar.selectbox("Destino:", ["— Todos —"]+df_aeros["name"].tolist())
origen_act  = origen_sel  != "— Todos —"
destino_act = destino_sel != "— Todos —"
icao_origen = df_aeros[df_aeros["name"]==origen_sel]["ident"].values[0]  if origen_act  else None
icao_destino= df_aeros[df_aeros["name"]==destino_sel]["ident"].values[0] if destino_act else None

st.sidebar.divider()
c1b,c2b = st.sidebar.columns(2)
btn_consultar = c1b.button("✈️ Consultar", use_container_width=True, type="primary")
btn_limpiar   = c2b.button("🗑️ Limpiar",  use_container_width=True)
st.sidebar.caption("⚠️ Pulsa Consultar al cambiar fecha u hora.")

# ── ACCIONES ─────────────────────────────────────────────────────
if btn_limpiar:
    for k in ["hist_sv","hist_fl","hist_fecha","hist_hora","hist_minuto","hist_trayectoria"]:
        st.session_state[k] = None

if btn_consultar:
    if not cont_sel:
        st.sidebar.error("Selecciona al menos un continente.")
    else:
        with st.spinner("⏳ [1/2] Descargando posiciones..."):
            try:
                df_sv = query_state_vectors(fecha_sel,hora_sel,minuto_sel,user_trino,cont_sel)
                st.session_state.update({
                    "hist_sv":df_sv,"hist_fecha":str(fecha_sel),
                    "hist_hora":hora_sel,"hist_minuto":minuto_sel,
                    "hist_fl":None,"hist_trayectoria":None
                })
                if df_sv.empty: st.sidebar.warning("Sin resultados.")
                else: st.sidebar.success(f"✅ {len(df_sv):,} aviones")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
                if "trino_conn" in st.session_state: del st.session_state["trino_conn"]

        if (origen_act or destino_act) and st.session_state.get("hist_sv") is not None:
            with st.spinner("⏳ [2/2] Filtrando por ruta (flights_data4)..."):
                try:
                    df_fl = query_flights_data4(fecha_sel,user_trino,icao_origen,icao_destino)
                    st.session_state["hist_fl"] = df_fl
                    st.sidebar.info(f"🗓️ {len(df_fl):,} vuelos con esa ruta ese día.")
                except Exception as e:
                    st.sidebar.error(f"Error flights_data4: {e}")

# ── DATOS ─────────────────────────────────────────────────────────
df_sv = st.session_state.get("hist_sv")
if df_sv is None: df_sv = pd.DataFrame()
df_fl = st.session_state.get("hist_fl")
if df_fl is None: df_fl = pd.DataFrame()
df_vuelos = df_sv.copy()

if not df_vuelos.empty and (origen_act or destino_act) and not df_fl.empty:
    ok = set(df_fl["icao24"].unique())
    df_vuelos = df_vuelos[df_vuelos["icao24"].isin(ok)].reset_index(drop=True)

if not df_vuelos.empty and not df_fl.empty:
    dedup = df_fl.drop_duplicates("icao24",keep="last")
    df_vuelos = df_vuelos.merge(dedup[["icao24","origen","destino"]],on="icao24",how="left")

mask = df_base["continent"].isin(cont_sel) & df_base["type"].isin(tipos_sel)
df_view = df_base[mask]

# ── UI ────────────────────────────────────────────────────────────
st.title("📅 Análisis Histórico de Tráfico Aéreo")
st.caption("Consultas SQL sobre OpenSky via Trino · state_vectors_data4 + flights_data4")

if not df_sv.empty:
    f=st.session_state.get("hist_fecha",""); h=st.session_state.get("hist_hora",0); m=st.session_state.get("hist_minuto",0)
    partes = [f"📦 **{len(df_sv):,}** aviones · **{f} {h:02d}:{m:02d} UTC**"]
    if origen_act:  partes.append(f"🛫 Origen: **{origen_sel}**")
    if destino_act: partes.append(f"🛬 Destino: **{destino_sel}**")
    partes.append(f"✈️ Mostrando: **{len(df_vuelos):,}**")
    st.info(" · ".join(partes))

    ca,cb = st.columns(2)
    with ca:
        if st.button("🔬 Enviar al Simulador", use_container_width=True, type="primary"):
            st.session_state["datos_sim"] = {
                "fuente":"historico","label":f"Histórico · {f} {h:02d}:{m:02d}",
                "df":df_vuelos.copy(),"fecha":f,
                "continente":cont_sel[0] if len(cont_sel)==1 else "EU"
            }
            st.success("✅ Enviado. Ve a **Simulador** →")
    with cb:
        st.download_button("⬇️ Descargar CSV",data=df_vuelos.to_csv(index=False),
            file_name=f"historico_{f}_{h:02d}{m:02d}.csv",mime="text/csv",use_container_width=True)

# Mapa
COLORES = {"large_airport":"#FF4B4B","medium_airport":"#1C83E1","small_airport":"#00FF7F"}
TAMANOS = {"large_airport":10,"medium_airport":7,"small_airport":4}

fig = go.Figure()
for tipo in tipos_sel:
    df_t = df_view[df_view["type"]==tipo]
    if df_t.empty: continue
    fig.add_trace(go.Scattermap(
        lat=df_t["latitude_deg"],lon=df_t["longitude_deg"],mode="markers",name=tipo,
        marker=go.scattermap.Marker(size=TAMANOS[tipo],color=COLORES[tipo],opacity=0.6),
        text=df_t["name"],hoverinfo="text"
    ))

if origen_act:
    fo = df_base[df_base["name"]==origen_sel]
    if not fo.empty:
        fig.add_trace(go.Scattermap(lat=fo["latitude_deg"],lon=fo["longitude_deg"],
            mode="markers+text",name=f"🛫 {icao_origen}",text=[icao_origen],textposition="top right",
            marker=go.scattermap.Marker(size=22,color="cyan"),
            hovertext=[f"🛫 ORIGEN: {origen_sel}"],hoverinfo="text"))

if destino_act:
    fd = df_base[df_base["name"]==destino_sel]
    if not fd.empty:
        fig.add_trace(go.Scattermap(lat=fd["latitude_deg"],lon=fd["longitude_deg"],
            mode="markers+text",name=f"🛬 {icao_destino}",text=[icao_destino],textposition="top right",
            marker=go.scattermap.Marker(size=22,color="orange"),
            hovertext=[f"🛬 DESTINO: {destino_sel}"],hoverinfo="text"))

if not df_vuelos.empty:
    dv=df_vuelos.copy()
    dv["callsign"]=dv["callsign"].fillna("").str.strip()
    dv["velocity"]=pd.to_numeric(dv["velocity"],errors="coerce").fillna(0)
    dv["baroaltitude"]=pd.to_numeric(dv["baroaltitude"],errors="coerce").fillna(0)
    vel_kmh=(dv["velocity"]*3.6).round(0).astype(int).astype(str)
    alt_ft=(dv["baroaltitude"]*3.281).round(0).astype(int).astype(str)
    hover="✈️ <b>"+dv["callsign"]+"</b><br>"+"ICAO24: "+dv["icao24"]+"<br>"
    if "origen"  in dv.columns: hover+="Origen: " +dv["origen"].fillna("?")+"<br>"
    if "destino" in dv.columns: hover+="Destino: "+dv["destino"].fillna("?")+"<br>"
    hover+="Vel: "+vel_kmh+" km/h · Alt: "+alt_ft+" ft"
    fig.add_trace(go.Scattermap(lat=dv["lat"],lon=dv["lon"],mode="markers",
        name=f"✈️ Aviones ({len(dv):,})",
        marker=go.scattermap.Marker(size=9,color="yellow"),
        text=hover,hoverinfo="text"))

if destino_act:
    fd=df_base[df_base["name"]==destino_sel]; mc=dict(lat=fd["latitude_deg"].values[0],lon=fd["longitude_deg"].values[0]) if not fd.empty else dict(lat=40,lon=-3)
elif origen_act:
    fo=df_base[df_base["name"]==origen_sel]; mc=dict(lat=fo["latitude_deg"].values[0],lon=fo["longitude_deg"].values[0]) if not fo.empty else dict(lat=40,lon=-3)
elif not df_view.empty:
    mc=dict(lat=df_view["latitude_deg"].mean(),lon=df_view["longitude_deg"].mean())
else:
    mc=dict(lat=40,lon=-3)

fig.update_layout(map_style="carto-darkmatter",margin={"r":0,"t":0,"l":0,"b":0},height=660,
    showlegend=True,legend=dict(yanchor="top",y=0.98,xanchor="left",x=0.02,
    bgcolor="rgba(0,0,0,0.6)",font=dict(color="white")),map=dict(center=mc,zoom=3.5))
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom":True})

# ── TRAYECTORIA ───────────────────────────────────────────────────
st.divider()
st.subheader("🛤️ Trayectoria de un vuelo")
st.caption("Selecciona un avión para ver su ruta real en la base de datos de OpenSky (±2 horas)")

if not df_vuelos.empty:
    df_vuelos["label"] = df_vuelos["callsign"].fillna("") + " (" + df_vuelos["icao24"] + ")"
    labels  = ["— Selecciona un avión —"] + df_vuelos["label"].tolist()
    sel_avion = st.selectbox("Avión:", labels)

    if sel_avion != "— Selecciona un avión —":
        icao24_sel = sel_avion.split("(")[-1].rstrip(")")
        btn_tray = st.button("📍 Cargar trayectoria", type="primary")

        if btn_tray:
            with st.spinner(f"Cargando trayectoria de {icao24_sel}..."):
                try:
                    df_tray = query_trayectoria(icao24_sel, fecha_sel, hora_sel, user_trino)
                    st.session_state["hist_trayectoria"] = df_tray
                    st.session_state["hist_tray_icao"]   = icao24_sel
                except Exception as e:
                    st.error(f"Error: {e}")

        df_tray = st.session_state.get("hist_trayectoria")
        icao_tray = st.session_state.get("hist_tray_icao","")

        if df_tray is not None and not df_tray.empty and icao_tray == icao24_sel:
            st.success(f"✅ {len(df_tray)} puntos de trayectoria · "
                       f"Duración: {(df_tray['time'].max()-df_tray['time'].min())//60} min")

            alt_ft_t = (pd.to_numeric(df_tray["baroaltitude"],errors="coerce").fillna(0)*3.281).round(0).astype(int)
            vel_t    = (pd.to_numeric(df_tray["velocity"],errors="coerce").fillna(0)*3.6).round(0).astype(int)
            hover_t  = ("🛤️ " + icao24_sel + "<br>" +
                        "Alt: " + alt_ft_t.astype(str) + " ft<br>" +
                        "Vel: " + vel_t.astype(str) + " km/h")

            fig_t = go.Figure()
            # Línea de trayectoria
            fig_t.add_trace(go.Scattermap(
                lat=df_tray["lat"], lon=df_tray["lon"],
                mode="lines+markers",
                name=f"Trayectoria {icao24_sel}",
                line=dict(width=3, color="cyan"),
                marker=go.scattermap.Marker(size=5, color="white"),
                text=hover_t, hoverinfo="text"
            ))
            # Punto de inicio y fin
            fig_t.add_trace(go.Scattermap(
                lat=[df_tray["lat"].iloc[0], df_tray["lat"].iloc[-1]],
                lon=[df_tray["lon"].iloc[0], df_tray["lon"].iloc[-1]],
                mode="markers+text",
                name="Inicio / Fin",
                text=["▶ Inicio","■ Fin"],
                textposition=["top right","top right"],
                marker=go.scattermap.Marker(size=14, color=["lime","red"]),
                hoverinfo="text"
            ))
            mc_t = dict(lat=df_tray["lat"].mean(), lon=df_tray["lon"].mean())
            fig_t.update_layout(
                map_style="carto-darkmatter",
                margin={"r":0,"t":0,"l":0,"b":0}, height=450,
                map=dict(center=mc_t, zoom=4),
                showlegend=True,
                legend=dict(yanchor="top",y=0.98,xanchor="left",x=0.02,
                            bgcolor="rgba(0,0,0,0.6)",font=dict(color="white"))
            )
            st.plotly_chart(fig_t, use_container_width=True, config={"scrollZoom":True})

            # Perfil de altitud
            df_tray["minuto"] = ((df_tray["time"] - df_tray["time"].min()) / 60).round(0)
            df_tray["alt_ft"] = (pd.to_numeric(df_tray["baroaltitude"],errors="coerce").fillna(0)*3.281).round(0)
            fig_alt = px.line(df_tray, x="minuto", y="alt_ft",
                              title=f"Perfil de altitud — {icao24_sel}",
                              labels={"minuto":"Minutos desde inicio","alt_ft":"Altitud (ft)"})
            fig_alt.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                                  font_color="white",height=250)
            st.plotly_chart(fig_alt, use_container_width=True)
        elif df_tray is not None and df_tray.empty:
            st.warning("No se encontraron datos de trayectoria para este vuelo en la ventana seleccionada.")
else:
    st.info("Primero consulta datos históricos para ver trayectorias.")

# ── AISLAMIENTO ───────────────────────────────────────────────────
st.divider()
st.subheader("📊 Análisis de Aislamiento Geográfico")
cc1,cc2 = st.columns([1,2])
with cc1:
    dist_min = st.slider("Aeropuertos a más de X km del más cercano:",0,2000,100,25)
    tipos_ais = st.multiselect("Tipos:",["large_airport","medium_airport","small_airport"],
                               default=["large_airport","medium_airport","small_airport"],key="tais")
with cc2:
    df_ais  = df_view[df_view["type"].isin(tipos_ais)] if tipos_ais else df_view
    df_isol = df_ais[df_ais["distancia_vecino_km"]>=dist_min].sort_values("distancia_vecino_km",ascending=False)
    m1,m2 = st.columns(2)
    m1.metric("Aeropuertos en selección",f"{len(df_ais):,}")
    m2.metric(f"Aislados (>{dist_min} km)",f"{len(df_isol):,}")

if not df_ais.empty:
    fig_h=px.histogram(df_ais,x="distancia_vecino_km",nbins=50,color_discrete_sequence=["#1C83E1"],
                       labels={"distancia_vecino_km":"Distancia al vecino (km)"})
    fig_h.add_vline(x=dist_min,line_dash="dash",line_color="red",
                    annotation_text=f"Umbral: {dist_min}km",annotation_position="top right")
    fig_h.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                        font_color="white",height=250,margin=dict(t=10,b=10))
    st.plotly_chart(fig_h, use_container_width=True)

if not df_isol.empty:
    cm_col, ct_col = st.columns([3,2])
    with cm_col:
        fig_m = px.scatter_map(df_ais,lat="latitude_deg",lon="longitude_deg",
            color="distancia_vecino_km",size="distancia_vecino_km",size_max=18,
            color_continuous_scale="RdYlGn_r",hover_name="name",
            hover_data={"ident":True,"municipality":True,"distancia_vecino_km":":.1f",
                        "latitude_deg":False,"longitude_deg":False},
            map_style="carto-darkmatter",zoom=2)
        fig_m.update_layout(height=380,margin=dict(r=0,t=0,l=0,b=0),
                            coloraxis_colorbar=dict(title="km",thickness=12))
        st.plotly_chart(fig_m, use_container_width=True)
    with ct_col:
        st.dataframe(df_isol[["ident","name","municipality","type","distancia_vecino_km"]].head(20).rename(
            columns={"ident":"ICAO","name":"Aeropuerto","municipality":"Ciudad",
                     "type":"Tipo","distancia_vecino_km":"Dist. vecino (km)"}),
            use_container_width=True, height=360)