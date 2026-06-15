"""
app.py — HOME: Tráfico Aéreo en Tiempo Real
TFG: Simulación y Análisis del Impacto Operativo de la Red Aérea Global
Jacob Altenburger Villar · UAX 2026
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="TFG – Red Aérea Global", page_icon="✈️", layout="wide")

# ================================================================
# TOKEN MANAGER — renueva el token OAuth2 automáticamente
# Los tokens de OpenSky expiran a los 30 minutos.
# ================================================================
TOKEN_URL = ("https://auth.opensky-network.org/auth/realms/"
             "opensky-network/protocol/openid-connect/token")

class TokenManager:
    def __init__(self):
        self.token = None
        self.expires_at = None

    def get(self, client_id, client_secret):
        if self.token and self.expires_at and datetime.now() < self.expires_at:
            return self.token, None
        return self._refresh(client_id, client_secret)

    def _refresh(self, client_id, client_secret):
        try:
            r = requests.post(TOKEN_URL, data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            }, timeout=15)
            if r.status_code == 200:
                data = r.json()
                self.token = data["access_token"]
                self.expires_at = datetime.now() + timedelta(seconds=data.get("expires_in", 1800) - 60)
                return self.token, None
            return None, f"HTTP {r.status_code}: {r.text[:150]}"
        except Exception as e:
            return None, str(e)

if "token_mgr" not in st.session_state:
    st.session_state.token_mgr = TokenManager()
tm = st.session_state.token_mgr

# Categorías OpenSky (extended=1)
CATS = {
    0:"Sin info", 1:"Sin cat. ADS-B", 2:"Ligera", 3:"Pequeña",
    4:"Grande (A320/B737)", 5:"High Vortex (B757)", 6:"Pesada (A380/B747)",
    7:"Alto rendimiento", 8:"Helicóptero", 9:"Planeador",
    10:"Más ligero que el aire", 11:"Paracaidista", 12:"Ultraligero",
    14:"Dron/UAV", 15:"Vehículo espacial",
}
CATS_COMERCIAL = {3, 4, 5, 6}
COLOR_CAT = {
    "Grande (A320/B737)": "#FFE000",
    "Pesada (A380/B747)": "#FF8C00",
    "High Vortex (B757)": "#FFC300",
    "Pequeña":            "#90EE90",
    "Helicóptero":        "#87CEEB",
}

REGIONES = {
    "Europa":       (29.0, 81.2, -25.0, 45.0),
    "Norteamérica": ( 7.4, 83.1,-176.6,-52.0),
    "Sudamérica":   (-55.0,12.4,-109.4,-32.4),
    "Asia":         (-12.2,80.8,  26.0,180.0),
    "África":       (-35.0,37.2, -25.0, 63.4),
    "Mundial":      (-90.0,90.0,-180.0,180.0),
}

def coste_creditos(bbox):
    area = (bbox[1]-bbox[0]) * (bbox[3]-bbox[2])
    return 1 if area<=25 else 2 if area<=100 else 3 if area<=400 else 4

def get_vuelos_live(token, bbox, solo_comercial):
    lat_min, lat_max, lon_min, lon_max = bbox
    try:
        r = requests.get(
            "https://opensky-network.org/api/states/all",
            headers={"Authorization": f"Bearer {token}"},
            params={"lamin":lat_min,"lamax":lat_max,"lomin":lon_min,"lomax":lon_max,"extended":1},
            timeout=20
        )
        creditos = r.headers.get("X-Rate-Limit-Remaining", "?")
        if r.status_code == 200:
            states = r.json().get("states", [])
            if not states:
                return pd.DataFrame(), creditos, None
            cols = ["icao24","callsign","origin_country","time_position","last_contact",
                    "longitude","latitude","baro_altitude","on_ground","velocity",
                    "true_track","vertical_rate","sensors","geo_altitude",
                    "squawk","spi","position_source","category"]
            df = pd.DataFrame(states, columns=cols)
            df = df[df["on_ground"]==False].dropna(subset=["latitude","longitude"]).copy()
            df["callsign"]      = df["callsign"].fillna("").str.strip()
            df["velocity"]      = pd.to_numeric(df["velocity"],      errors="coerce").fillna(0)
            df["baro_altitude"] = pd.to_numeric(df["baro_altitude"], errors="coerce").fillna(0)
            df["true_track"]    = pd.to_numeric(df["true_track"],    errors="coerce").fillna(0)
            df["category"]      = pd.to_numeric(df["category"],      errors="coerce").fillna(0).astype(int)
            df["cat_str"]       = df["category"].map(CATS).fillna("Otro")
            if solo_comercial:
                df = df[df["category"].isin(CATS_COMERCIAL)].reset_index(drop=True)
            return df, creditos, None
        elif r.status_code == 401: return pd.DataFrame(), "?", "❌ Token inválido."
        elif r.status_code == 429:
            retry = r.headers.get("X-Rate-Limit-Retry-After-Seconds","?")
            return pd.DataFrame(), "0", f"⏳ Sin créditos. Espera {retry}s."
        else: return pd.DataFrame(), "?", f"HTTP {r.status_code}"
    except Exception as e:
        return pd.DataFrame(), "?", str(e)

# ── SIDEBAR ──────────────────────────────────────────────────────
st.sidebar.header("🔑 Credenciales OpenSky API")
client_id     = st.sidebar.text_input("clientId", value="jaltevil@myuax.com-api-client")
client_secret = st.sidebar.text_input("clientSecret", type="password")

st.sidebar.divider()
st.sidebar.header("🗺️ Filtros")
region_sel     = st.sidebar.selectbox("Región:", list(REGIONES.keys()))
solo_comercial = st.sidebar.toggle("Solo aviación comercial", value=True,
    help="Excluye drones, planeadores y vehículos de superficie")

bb     = REGIONES[region_sel]
coste  = coste_creditos(bb)
st.sidebar.caption(f"💳 Esta consulta cuesta **{coste} crédito{'s' if coste>1 else ''}**.")

st.sidebar.divider()
# Estado del token
if tm.token and tm.expires_at:
    mins = max(0, int((tm.expires_at - datetime.now()).total_seconds()/60))
    col = "🟢" if mins>5 else "🟡"
    st.sidebar.caption(f"{col} Token {'activo · expira en '+str(mins)+' min' if mins>0 else 'renovándose...'}")
else:
    st.sidebar.caption("⚪ Sin token — se obtendrá al consultar")

if "live_creditos" in st.session_state:
    st.sidebar.metric("💳 Créditos restantes", st.session_state["live_creditos"])

btn_act = st.sidebar.button("🔄 Actualizar tráfico en vivo",
                             use_container_width=True, type="primary")
st.sidebar.caption("⚠️ Máx. 1 consulta / 10 s")

# ── ACCIÓN ───────────────────────────────────────────────────────
if btn_act:
    if not client_id or not client_secret:
        st.sidebar.error("Rellena clientId y clientSecret.")
        st.stop()
    with st.spinner("🔐 Verificando token..."):
        token, err = tm.get(client_id, client_secret)
    if err: st.error(err); st.stop()

    with st.spinner("📡 Consultando OpenSky..."):
        df_live, creditos, err = get_vuelos_live(token, bb, solo_comercial)

    st.session_state["live_creditos"] = creditos
    if err:
        st.error(err)
    else:
        st.session_state.update({
            "df_live": df_live, "live_region": region_sel,
            "live_ts": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            "live_comercial": solo_comercial,
        })
        st.sidebar.success(f"✅ {len(df_live):,} aviones · {creditos} créditos")

# ── VISUALIZACIÓN ─────────────────────────────────────────────────
st.title("✈️ Red Aérea Global — Tiempo Real")
st.caption("TFG · Jacob Altenburger Villar · UAX 2026 · OpenSky Network REST API")

df_live = st.session_state.get("df_live", pd.DataFrame())

if not df_live.empty:
    ts  = st.session_state.get("live_ts","")
    reg = st.session_state.get("live_region","")
    com = st.session_state.get("live_comercial", True)
    st.info(f"📡 **{len(df_live):,}** aviones · Región: **{reg}** · **{ts}** "
            f"{'· Solo comercial' if com else ''}")

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("✈️ Aviones",          f"{len(df_live):,}")
    k2.metric("🌍 Países de origen", f"{df_live['origin_country'].nunique()}")
    k3.metric("💨 Vel. media",       f"{(df_live['velocity']*3.6).mean():.0f} km/h")
    k4.metric("📏 Alt. media",       f"{(df_live['baro_altitude']*3.281).mean():.0f} ft")
    k5.metric("📡 ADS-B directo",    f"{(df_live['position_source']==0).sum():,}")

    ca, cb = st.columns(2)
    with ca:
        if st.button("🔬 Enviar al Simulador", use_container_width=True, type="primary"):
            st.session_state["datos_sim"] = {
                "fuente": "live", "label": f"API en vivo · {reg} · {ts}",
                "df": df_live.rename(columns={
                    "latitude":"lat","longitude":"lon",
                    "true_track":"heading","baro_altitude":"baroaltitude"
                }).copy(),
                "fecha": None, "continente": reg
            }
            st.success("✅ Enviado. Ve a **Simulador** →")
    with cb:
        st.download_button("⬇️ Descargar CSV",
            data=df_live.to_csv(index=False),
            file_name=f"live_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv", use_container_width=True)

        # Mapa — todos los aviones en amarillo simple
        vel_kmh = (df_live["velocity"] * 3.6).round(0).astype(int).astype(str)
        alt_ft = (df_live["baro_altitude"] * 3.281).round(0).astype(int).astype(str)
        hover = ("✈️ <b>" + df_live["callsign"] + "</b><br>" +
                 "País: " + df_live["origin_country"] + "<br>" +
                 "ICAO24: " + df_live["icao24"] + "<br>" +
                 "Vel: " + vel_kmh + " km/h · Alt: " + alt_ft + " ft<br>" +
                 "Rumbo: " + df_live["true_track"].round(0).astype(int).astype(str) + "°")

        fig = go.Figure()


        # Capa 1: aeropuertos principales de la región
        @st.cache_data
        def cargar_aeropuertos():
            df = pd.read_csv("airports.csv")
            df = df[df["type"].isin(["large_airport", "medium_airport"])].copy()
            df = df.dropna(subset=["latitude_deg", "longitude_deg"])
            df["continent"] = df["continent"].fillna("NA")
            return df


        df_aeros = cargar_aeropuertos()
        # Filtrar por región aproximada
        bb = REGIONES[region_sel]
        if region_sel != "Mundial":
            df_aeros_vis = df_aeros[
                (df_aeros["latitude_deg"] >= bb[0]) & (df_aeros["latitude_deg"] <= bb[1]) &
                (df_aeros["longitude_deg"] >= bb[2]) & (df_aeros["longitude_deg"] <= bb[3])
                ]
        else:
            df_aeros_vis = df_aeros

        # Large airports
        df_lg = df_aeros_vis[df_aeros_vis["type"] == "large_airport"]
        if not df_lg.empty:
            fig.add_trace(go.Scattermap(
                lat=df_lg["latitude_deg"], lon=df_lg["longitude_deg"],
                mode="markers", name="Large airport",
                marker=go.scattermap.Marker(size=8, color="#FF4B4B", opacity=0.6),
                text=df_lg["name"], hoverinfo="text"
            ))

        # Medium airports
        df_md = df_aeros_vis[df_aeros_vis["type"] == "medium_airport"]
        if not df_md.empty:
            fig.add_trace(go.Scattermap(
                lat=df_md["latitude_deg"], lon=df_md["longitude_deg"],
                mode="markers", name="Medium airport",
                marker=go.scattermap.Marker(size=5, color="#1C83E1", opacity=0.5),
                text=df_md["name"], hoverinfo="text"
            ))

        # Capa 2: aviones
        fig.add_trace(go.Scattermap(
            lat=df_live["latitude"], lon=df_live["longitude"],
            mode="markers", name=f"Aviones ({len(df_live):,})",
            marker=go.scattermap.Marker(size=6, color="yellow", opacity=0.85),
            text=hover, hoverinfo="text"
        ))

        fig.update_layout(
            map_style="carto-darkmatter",
            margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=620,
            map=dict(center=dict(lat=(bb[0] + bb[1]) / 2, lon=(bb[2] + bb[3]) / 2),
                     zoom=3.0 if region_sel != "Mundial" else 1.5),
            showlegend=True,
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                        bgcolor="rgba(0,0,0,0.6)", font=dict(color="white"))
        )
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

        # Solo top países (quitamos el pie de categorías)
        st.divider()
        st.subheader("🌐 Top 15 países con más aviones ahora")
        top = df_live["origin_country"].value_counts().head(15).reset_index()
        top.columns = ["País", "Aviones"]
        fig_bar = px.bar(top, x="Aviones", y="País", orientation="h",
                         color="Aviones", color_continuous_scale="YlOrRd")
        fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font_color="white", height=400,
                              yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.markdown("""
    ### 👋 Bienvenido al TFG de Análisis de la Red Aérea Global
    - 🌍 **Tiempo Real** — tráfico aéreo ahora mismo (esta página)
    - 📅 **Análisis Histórico** — consulta días concretos via Trino
    - 🔬 **Simulador** — cierra aeropuertos, ve el efecto cascada, calcula CO₂

    **Para empezar:** introduce tus credenciales → pulsa *Actualizar tráfico en vivo* →
    """)