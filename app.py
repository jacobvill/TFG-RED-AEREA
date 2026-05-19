"""
app.py — Página principal: Tráfico Aéreo en Tiempo Real
TFG: Simulación y Análisis del Impacto Operativo de la Red Aérea Global
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import numpy as np
from datetime import datetime, timezone

st.set_page_config(
    page_title="TFG – Red Aérea Global",
    page_icon="✈",
    layout="wide"
)

# ================================================================
# CABECERA
# ================================================================
st.title("✈ Red Aérea Global — Tiempo Real")
st.caption(
    "TFG · Jacob Altenburger Villar · UAX 2026 · "
    "Datos via OpenSky Network REST API"
)

# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.header("🔑 Credenciales OpenSky API")
client_id     = st.sidebar.text_input("clientId", value="jaltevil@myuax.com-api-client")
client_secret = st.sidebar.text_input("clientSecret", type="password")
user_b = pass_b = None  # no se usa

st.sidebar.divider()
st.sidebar.header("🗺️ Filtros")
REGIONES = {
    "Europa":       (29.0,  81.2, -25.0,  45.0),
    "Norteamérica": ( 7.4,  83.1,-176.6, -52.0),
    "Sudamérica":   (-55.0, 12.4,-109.4, -32.4),
    "Asia":         (-12.2, 80.8,  26.0, 180.0),
    "África":       (-35.0, 37.2, -25.0,  63.4),
    "Mundial":      (-90.0, 90.0,-180.0, 180.0),
}
region_sel = st.sidebar.selectbox("Región:", list(REGIONES.keys()), index=0)

st.sidebar.divider()
btn_actualizar = st.sidebar.button(
    "🔄 Actualizar tráfico en vivo", use_container_width=True, type="primary"
)
st.sidebar.caption("⚠️ OpenSky limita a 1 consulta / 10 s. Usa el botón manualmente.")

# ================================================================
# FUNCIONES
# ================================================================
OPENSKY_TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/"
    "opensky-network/protocol/openid-connect/token"
)

def obtener_token(client_id, client_secret):
    try:
        r = requests.post(
            OPENSKY_TOKEN_URL,
            data={"grant_type": "client_credentials",
                  "client_id": client_id, "client_secret": client_secret},
            timeout=15
        )
        if r.status_code == 200:
            return r.json().get("access_token"), None
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, str(e)


def get_vuelos_live(bbox, token=None, user=None, password=None):
    lat_min, lat_max, lon_min, lon_max = bbox
    params  = {"lamin": lat_min, "lamax": lat_max,
                "lomin": lon_min, "lomax": lon_max}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    auth    = (user, password) if user else None

    try:
        r = requests.get(
            "https://opensky-network.org/api/states/all",
            headers=headers, auth=auth, params=params, timeout=20
        )
        if r.status_code == 200:
            states = r.json().get("states", [])
            if not states:
                return pd.DataFrame(), "Sin aviones en esta región."
            cols = ["icao24","callsign","origin_country","time_position",
                    "last_contact","longitude","latitude","baro_altitude",
                    "on_ground","velocity","true_track","vertical_rate",
                    "sensors","geo_altitude","squawk","spi","position_source"]
            df = pd.DataFrame(states, columns=cols)
            df = df[df["on_ground"] == False].dropna(subset=["latitude","longitude"]).copy()
            df["callsign"]      = df["callsign"].fillna("").str.strip()
            df["velocity"]      = pd.to_numeric(df["velocity"],      errors="coerce").fillna(0)
            df["baro_altitude"] = pd.to_numeric(df["baro_altitude"], errors="coerce").fillna(0)
            df["true_track"]    = pd.to_numeric(df["true_track"],    errors="coerce").fillna(0)
            return df, None
        elif r.status_code == 401:
            return pd.DataFrame(), "❌ Credenciales incorrectas."
        elif r.status_code == 429:
            return pd.DataFrame(), "⏳ Rate limit. Espera unos segundos."
        else:
            return pd.DataFrame(), f"HTTP {r.status_code}"
    except Exception as e:
        return pd.DataFrame(), str(e)

# ================================================================
# ACCIÓN: ACTUALIZAR
# ================================================================
if btn_actualizar:
    if not client_id or not client_secret:
        st.sidebar.error("Rellena clientId y clientSecret.")
        st.stop()

    with st.spinner("🔐 Obteniendo token..."):
        token, err = obtener_token(client_id, client_secret)
    if err:
        st.error(err); st.stop()

    with st.spinner("📡 Consultando OpenSky..."):
        df_live, err = get_vuelos_live(
            REGIONES[region_sel], token=token
        )
    if err:
        st.error(err)
    else:
        st.session_state["df_live"]     = df_live
        st.session_state["live_region"] = region_sel
        st.session_state["live_ts"]     = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        st.sidebar.success(f"✅ {len(df_live):,} aviones")
# ================================================================
# VISUALIZACIÓN
# ================================================================
df_live = st.session_state.get("df_live", pd.DataFrame())

if not df_live.empty:
    ts  = st.session_state.get("live_ts", "")
    reg = st.session_state.get("live_region", "")

    # Banner
    st.info(
        f"📡 **{len(df_live):,}** aviones en vuelo · "
        f"Región: **{reg}** · Captura: **{ts}**"
    )

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("✈ Aviones en vuelo",  f"{len(df_live):,}")
    k2.metric("🌍 Países de origen",  f"{df_live['origin_country'].nunique()}")
    k3.metric("💨 Vel. media",        f"{(df_live['velocity']*3.6).mean():.0f} km/h")
    k4.metric("📏 Alt. media",        f"{(df_live['baro_altitude']*3.281).mean():.0f} ft")

    # Botones de acción
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔬 Enviar al Simulador", use_container_width=True, type="primary"):
            st.session_state["datos_sim"] = {
                "fuente":     "live",
                "label":      f"API en vivo · {reg} · {ts}",
                "df":         df_live.rename(columns={
                                  "latitude":      "lat",
                                  "longitude":     "lon",
                                  "true_track":    "heading",
                                  "baro_altitude": "baroaltitude"
                              }),
                "fecha":      None,
                "continente": reg
            }
            st.success("✅ Datos enviados. Ve a **Simulador** en el menú lateral.")
    with col_b:
        csv = df_live.to_csv(index=False)
        st.download_button(
            "⬇️ Descargar CSV",
            data=csv,
            file_name=f"opensky_live_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Mapa
    vel_kmh = (df_live["velocity"] * 3.6).round(0).astype(int).astype(str)
    alt_ft  = (df_live["baro_altitude"] * 3.281).round(0).astype(int).astype(str)
    hover   = (
        "✈ <b>" + df_live["callsign"] + "</b><br>" +
        "País: "   + df_live["origin_country"] + "<br>" +
        "ICAO24: " + df_live["icao24"] + "<br>" +
        "Vel: "    + vel_kmh + " km/h  |  Alt: " + alt_ft + " ft<br>" +
        "Rumbo: "  + df_live["true_track"].round(0).astype(int).astype(str) + "°"
    )
    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=df_live["latitude"], lon=df_live["longitude"],
        mode="markers",
        name=f"Aviones ({len(df_live):,})",
        marker=go.scattermap.Marker(size=6, color="yellow", opacity=0.85),
        text=hover, hoverinfo="text"
    ))
    bb = REGIONES[region_sel]
    fig.update_layout(
        map_style="carto-darkmatter",
        margin={"r":0,"t":0,"l":0,"b":0}, height=620,
        map=dict(center=dict(lat=(bb[0]+bb[1])/2, lon=(bb[2]+bb[3])/2),
                 zoom=3.0 if region_sel != "Mundial" else 1.5),
        showlegend=True,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                    bgcolor="rgba(0,0,0,0.6)", font=dict(color="white"))
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    # Top países
    st.divider()
    st.subheader("🌐 Top 15 países con más aviones ahora")
    top = df_live["origin_country"].value_counts().head(15).reset_index()
    top.columns = ["País", "Aviones"]
    fig_bar = px.bar(top, x="Aviones", y="País", orientation="h",
                     color="Aviones", color_continuous_scale="YlOrRd")
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="white", height=400,
        yaxis=dict(autorange="reversed"), coloraxis_showscale=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.markdown("""
    ### 👋 Bienvenido al TFG de Análisis de la Red Aérea Global

    Esta herramienta permite:
    - 🌍 **Ver el tráfico aéreo en tiempo real** (esta página)
    - 📅 **Analizar días históricos** con datos de OpenSky/Trino
    - 🔬 **Simular crisis** — cierre de aeropuertos, rutas alternativas, impacto en CO₂

    **Para empezar:** introduce tus credenciales y pulsa *Actualizar tráfico en vivo* →
    """)