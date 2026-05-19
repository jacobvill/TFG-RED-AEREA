import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime, timezone

st.set_page_config(page_title="Tiempo Real - OpenSky", layout="wide")
st.title("🌍 Tráfico Aéreo en Tiempo Real")
st.caption("Datos en tiempo real via OpenSky REST API · Actualiza manualmente para respetar los límites de la API.")

# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.header("🔑 Credenciales OpenSky")
user_rt = st.sidebar.text_input("Usuario (sin @dominio)", value="jaltevil")
pass_rt = st.sidebar.text_input("Password", type="password")

st.sidebar.divider()
st.sidebar.header("🗺️ Región")
REGIONES = {
    "Europa":        (29.0,  81.2, -25.0,  45.0),
    "Norteamérica":  ( 7.4,  83.1,-176.6, -52.0),
    "Sudamérica":    (-55.0, 12.4,-109.4, -32.4),
    "Asia":          (-12.2, 80.8,  26.0, 180.0),
    "África":        (-35.0, 37.2, -25.0,  63.4),
    "Mundial":       (-90.0, 90.0,-180.0, 180.0),
}
region_sel = st.sidebar.selectbox("Región a monitorizar:", list(REGIONES.keys()), index=0)
btn_actualizar = st.sidebar.button("🔄 Actualizar ahora", use_container_width=True, type="primary")
st.sidebar.caption("⚠️ OpenSky limita a 1 consulta cada 10 s para usuarios registrados. Usa solo el botón.")

# ================================================================
# FUNCIÓN DE CONSULTA REST API
# ================================================================
def get_vuelos_en_vivo(user, password, bbox):
    lat_min, lat_max, lon_min, lon_max = bbox
    try:
        resp = requests.get(
            "https://opensky-network.org/api/states/all",
            auth=(user, password),
            params={
                "lamin": lat_min, "lamax": lat_max,
                "lomin": lon_min, "lomax": lon_max
            },
            timeout=20
        )
        if resp.status_code == 200:
            data  = resp.json()
            states = data.get("states", [])
            cols = [
                "icao24","callsign","origin_country","time_position","last_contact",
                "longitude","latitude","baro_altitude","on_ground","velocity",
                "true_track","vertical_rate","sensors","geo_altitude",
                "squawk","spi","position_source"
            ]
            df = pd.DataFrame(states, columns=cols)
            df = df[df["on_ground"] == False].copy()
            df = df.dropna(subset=["latitude","longitude"])
            df["callsign"]      = df["callsign"].fillna("").str.strip()
            df["velocity"]      = pd.to_numeric(df["velocity"],      errors="coerce").fillna(0)
            df["baro_altitude"] = pd.to_numeric(df["baro_altitude"], errors="coerce").fillna(0)
            df["true_track"]    = pd.to_numeric(df["true_track"],    errors="coerce").fillna(0)
            return df, None
        elif resp.status_code == 401:
            return pd.DataFrame(), "❌ Credenciales incorrectas."
        elif resp.status_code == 429:
            return pd.DataFrame(), "⏳ Rate limit alcanzado. Espera unos segundos."
        else:
            return pd.DataFrame(), f"Error HTTP {resp.status_code}"
    except Exception as e:
        return pd.DataFrame(), f"Error de conexión: {e}"

# ================================================================
# LÓGICA PRINCIPAL
# ================================================================
if btn_actualizar:
    if not user_rt or not pass_rt:
        st.sidebar.error("Introduce usuario y password.")
    else:
        with st.spinner("📡 Consultando OpenSky en tiempo real..."):
            bbox = REGIONES[region_sel]
            df_live, error = get_vuelos_en_vivo(user_rt, pass_rt, bbox)
            if error:
                st.error(error)
            else:
                st.session_state["df_live"]       = df_live
                st.session_state["live_region"]   = region_sel
                st.session_state["live_timestamp"] = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

df_live = st.session_state.get("df_live", pd.DataFrame())

# ================================================================
# MÉTRICAS Y MAPA
# ================================================================
if not df_live.empty:
    ts  = st.session_state.get("live_timestamp", "")
    reg = st.session_state.get("live_region", "")

    st.info(f"📡 **{len(df_live):,}** aviones en vuelo · Región: **{reg}** · Captura a las **{ts}**")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✈️ Aviones en vuelo",   f"{len(df_live):,}")
    c2.metric("🌍 Países de origen",   f"{df_live['origin_country'].nunique()}")
    c3.metric("💨 Vel. media (km/h)",  f"{(df_live['velocity'] * 3.6).mean():.0f}")
    c4.metric("📏 Alt. media (ft)",    f"{(df_live['baro_altitude'] * 3.281).mean():.0f}")

    # Hover text
    vel_kmh = (df_live["velocity"] * 3.6).round(0).astype(int).astype(str)
    alt_ft  = (df_live["baro_altitude"] * 3.281).round(0).astype(int).astype(str)
    hover   = (
        "✈️ <b>" + df_live["callsign"] + "</b><br>" +
        "País: " + df_live["origin_country"] + "<br>" +
        "ICAO24: " + df_live["icao24"] + "<br>" +
        "Vel: " + vel_kmh + " km/h<br>" +
        "Alt: " + alt_ft + " ft<br>" +
        "Rumbo: " + df_live["true_track"].round(0).astype(int).astype(str) + "°"
    )

    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=df_live["latitude"],
        lon=df_live["longitude"],
        mode="markers",
        name=f"Aviones en vivo ({len(df_live):,})",
        marker=go.scattermap.Marker(size=6, color="yellow", opacity=0.85),
        text=hover,
        hoverinfo="text"
    ))

    bbox_sel = REGIONES[region_sel]
    center_lat = (bbox_sel[0] + bbox_sel[1]) / 2
    center_lon = (bbox_sel[2] + bbox_sel[3]) / 2
    zoom = 3.0 if region_sel != "Mundial" else 1.5

    fig.update_layout(
        map_style="carto-darkmatter",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=650,
        map=dict(center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
        showlegend=True,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                    bgcolor="rgba(0,0,0,0.6)", font=dict(color="white"))
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    # Top países
    st.divider()
    st.subheader("🌐 Distribución por país de origen")
    top_paises = df_live["origin_country"].value_counts().head(15).reset_index()
    top_paises.columns = ["País", "Aviones"]
    import plotly.express as px
    fig_bar = px.bar(top_paises, x="Aviones", y="País", orientation="h",
                     color="Aviones", color_continuous_scale="Blues",
                     title="Top 15 países con más aviones en vuelo ahora")
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="white", height=400,
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.info("👆 Introduce tus credenciales y pulsa **Actualizar ahora** para ver el tráfico en vivo.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/World_map_blank_without_borders.svg/1280px-World_map_blank_without_borders.svg.png",
             caption="El mapa se cargará al actualizar", use_column_width=True)