"""
pages/3_🔬_Simulador.py
TFG: Simulador de Crisis Aérea
- Acepta datos de la página de Tiempo Real o Análisis Histórico
- También puede descargar un día completo de flights_data4 desde Trino
- Construye grafo con NetworkX
- Calcula betweenness y degree centrality
- Simula cierre/reducción de capacidad de aeropuertos
- Calcula impacto en CO₂ y combustible
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from sklearn.neighbors import BallTree
from trino.dbapi import connect
from trino.auth import OAuth2Authentication
from datetime import datetime, timezone

st.set_page_config(page_title="Simulador de Crisis", page_icon="🔬", layout="wide")
st.title("🔬 Simulador de Crisis Aérea")
st.caption("Modelado de red · Centralidad · Simulación de cierres · Impacto CO₂")

# ================================================================
# CONSTANTES
# ================================================================
CO2_KG_POR_KM  = 16.0   # ~5 kg combustible/km × 3.16 kg CO2/kg (A320 medio)
FUEL_KG_POR_KM = 5.0
BBOXES = {
    'EU': (-25.0, 29.0,  45.0, 81.2),
    'NA': (-176.6, 7.4, -52.0, 83.1),
    'SA': (-109.4,-55.0, -32.4, 12.4),
    'AS': (  26.0,-12.2, 180.0, 80.8),
    'AF': ( -25.1,-34.8,  63.4, 37.2),
}

# ================================================================
# AEROPUERTOS
# ================================================================
@st.cache_data
def cargar_aeropuertos():
    df = pd.read_csv("airports.csv")
    df = df[df["type"].isin(["small_airport","medium_airport","large_airport"])].copy()
    df = df.dropna(subset=["latitude_deg","longitude_deg"])
    df["continent"] = df["continent"].fillna("NA")
    return df

df_airports = cargar_aeropuertos()

# ================================================================
# TRINO
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

# ================================================================
# FUNCIONES DE ANÁLISIS
# ================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians(lat2-lat1)/2)**2
         + np.cos(phi1)*np.cos(phi2)*np.sin(np.radians(lon2-lon1)/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))


def estimar_destinos_desde_posicion(df_vuelos, df_airports):
    """
    Para datos sin origen/destino (snapshot de posición),
    proyecta la posición 45 min hacia adelante según heading y velocidad
    y asigna el aeropuerto más cercano a esa proyección como destino estimado.
    """
    col_lat = "lat" if "lat" in df_vuelos.columns else "latitude"
    col_lon = "lon" if "lon" in df_vuelos.columns else "longitude"
    col_hdg = "heading" if "heading" in df_vuelos.columns else "true_track"
    col_vel = "velocity"

    df = df_vuelos[[col_lat, col_lon, col_hdg, col_vel, "icao24","callsign"]].dropna().copy()
    df["vel_kmh"] = pd.to_numeric(df[col_vel], errors="coerce").fillna(0) * 3.6
    hdg_rad       = np.radians(pd.to_numeric(df[col_hdg], errors="coerce").fillna(0))

    # Proyección ~45 min
    dist_km = df["vel_kmh"] * 0.75
    df["lat_proj"] = pd.to_numeric(df[col_lat], errors="coerce") + np.cos(hdg_rad) * (dist_km / 111.32)
    df["lon_proj"] = pd.to_numeric(df[col_lon], errors="coerce") + np.sin(hdg_rad) * (dist_km / 111.32)
    df = df.dropna(subset=["lat_proj","lon_proj"])

    df_aero = df_airports[df_airports["type"].isin(["large_airport","medium_airport"])].dropna(
        subset=["latitude_deg","longitude_deg"]
    ).reset_index(drop=True)

    coords_aero = np.radians(df_aero[["latitude_deg","longitude_deg"]].values)
    coords_proj = np.radians(df[["lat_proj","lon_proj"]].values)

    tree = BallTree(coords_aero, metric="haversine")
    _, idxs = tree.query(coords_proj, k=1)

    df["destino_est"] = df_aero.iloc[idxs.flatten()]["ident"].values
    df = df.rename(columns={col_lat:"lat", col_lon:"lon"})
    return df[["icao24","callsign","lat","lon","destino_est"]]


@st.cache_data(show_spinner=False)
def descargar_flights_data4(fecha_str, continente, usuario):
    fecha  = datetime.strptime(fecha_str, "%Y-%m-%d")
    dt_utc = datetime(fecha.year, fecha.month, fecha.day, 0, 0, 0, tzinfo=timezone.utc)
    ts_day = int(dt_utc.timestamp())

    aeros_cont = df_airports[df_airports["continent"] == continente]["ident"].tolist()
    if not aeros_cont: return pd.DataFrame()

    conn = get_trino(usuario)
    q = f"""
        SELECT icao24,
               TRIM(callsign)       AS callsign,
               estdepartureairport  AS origen,
               estarrivalairport    AS destino
        FROM flights_data4
        WHERE day                 = {ts_day}
          AND estdepartureairport IS NOT NULL
          AND estarrivalairport   IS NOT NULL
          AND estdepartureairport != estarrivalairport
          AND icao24              IS NOT NULL
    """
    cur = conn.cursor(); cur.execute(q)
    rows = cur.fetchall(); cols = [d[0] for d in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df = df[df["origen"].isin(aeros_cont) | df["destino"].isin(aeros_cont)]
    return df.reset_index(drop=True)


def construir_grafo(df_vuelos_rutas):
    """Construye DiGraph a partir de pares origen→destino."""
    G = nx.DiGraph()
    aero_info = df_airports.set_index("ident")[["name","latitude_deg","longitude_deg","type"]].to_dict("index")

    rutas = df_vuelos_rutas.groupby(["origen","destino"]).size().reset_index(name="vuelos")

    for _, row in rutas.iterrows():
        src, dst, w = row["origen"], row["destino"], row["vuelos"]
        for n in [src, dst]:
            if n not in G.nodes and n in aero_info:
                i = aero_info[n]
                G.add_node(n, nombre=i["name"], lat=i["latitude_deg"],
                           lon=i["longitude_deg"], tipo=i["type"])
        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst, vuelos=w)
    return G


def calcular_centralidad(G):
    deg_in  = dict(G.in_degree(weight="vuelos"))
    deg_out = dict(G.out_degree(weight="vuelos"))
    deg_tot = {n: deg_in.get(n,0)+deg_out.get(n,0) for n in G.nodes()}

    k = min(200, max(2, len(G.nodes())-1))
    bet = nx.betweenness_centrality(G, k=k, weight="vuelos", normalized=True)

    aero_n = df_airports.set_index("ident")["name"].to_dict()
    aero_lat = df_airports.set_index("ident")["latitude_deg"].to_dict()
    aero_lon = df_airports.set_index("ident")["longitude_deg"].to_dict()

    df_m = pd.DataFrame({
        "icao":        list(deg_tot.keys()),
        "vuelos_tot":  [deg_tot[n] for n in deg_tot],
        "vuelos_in":   [deg_in.get(n,0) for n in deg_tot],
        "vuelos_out":  [deg_out.get(n,0) for n in deg_tot],
        "betweenness": [bet.get(n,0) for n in deg_tot],
    })
    df_m["nombre"] = df_m["icao"].map(aero_n).fillna(df_m["icao"])
    df_m["lat"]    = df_m["icao"].map(aero_lat)
    df_m["lon"]    = df_m["icao"].map(aero_lon)
    df_m = df_m.dropna(subset=["lat","lon"]).sort_values("betweenness", ascending=False).reset_index(drop=True)
    return df_m


def simular_cierre(G, df_metricas, df_airports, icao_cerrado, capacidad_pct):
    if icao_cerrado not in G.nodes:
        return pd.DataFrame(), {}, 0, 0

    # Vuelos afectados (con destino al aeropuerto)
    afectados = [
        {"origen": s, "destino": d, "vuelos": int(data["vuelos"] * (1-capacidad_pct/100))}
        for s, d, data in G.in_edges(icao_cerrado, data=True)
        if int(data["vuelos"] * (1-capacidad_pct/100)) > 0
    ]
    total_vuelos = sum(v["vuelos"] for v in afectados)
    if total_vuelos == 0:
        return pd.DataFrame(), {}, 0, 0

    info_c = df_airports[df_airports["ident"] == icao_cerrado]
    if info_c.empty: return pd.DataFrame(), {}, 0, 0
    lat_c, lon_c = info_c["latitude_deg"].values[0], info_c["longitude_deg"].values[0]

    df_alt = df_airports[
        (df_airports["ident"] != icao_cerrado) &
        (df_airports["type"].isin(["large_airport","medium_airport"]))
    ].dropna(subset=["latitude_deg","longitude_deg"]).copy()

    coords = np.radians(df_alt[["latitude_deg","longitude_deg"]].values)
    tree   = BallTree(coords, metric="haversine")
    dists, idxs = tree.query(np.radians([[lat_c, lon_c]]), k=min(5, len(df_alt)))

    df_alternativas = df_alt.iloc[idxs[0]].copy()
    df_alternativas["dist_km"] = (dists[0] * 6371).round(1)
    df_alternativas = df_alternativas[["ident","name","type","latitude_deg","longitude_deg","dist_km"]]

    dist_desvio     = df_alternativas["dist_km"].iloc[0]
    km_extra        = total_vuelos * dist_desvio
    co2_extra_kg    = km_extra * CO2_KG_POR_KM
    fuel_extra_kg   = km_extra * FUEL_KG_POR_KM

    resumen = {
        "icao":              icao_cerrado,
        "nombre":            info_c["name"].values[0],
        "capacidad":         f"{capacidad_pct}%",
        "vuelos_afectados":  total_vuelos,
        "alternativa":       df_alternativas.iloc[0]["name"],
        "dist_desvio_km":    dist_desvio,
        "km_extra":          round(km_extra),
        "co2_t":             round(co2_extra_kg / 1000, 1),
        "fuel_t":            round(fuel_extra_kg / 1000, 1),
    }
    return df_alternativas, resumen, total_vuelos, co2_extra_kg

# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.header("📥 Fuente de datos")

datos_ext = st.session_state.get("datos_sim")
if datos_ext:
    st.sidebar.success(f"📨 Datos recibidos:\n**{datos_ext['label']}**")
    usar_ext = st.sidebar.toggle("Usar datos recibidos", value=True)
else:
    usar_ext = False
    st.sidebar.info("No hay datos enviados desde otra página.\nDescarga desde Trino:")

st.sidebar.divider()
st.sidebar.header("📅 Descargar de Trino")
user_sim  = st.sidebar.text_input("Usuario Trino", value="jaltevil@myuax.com").lower()
fecha_sim = st.sidebar.date_input("Día", datetime(2024, 1, 16))
cont_sim  = st.sidebar.selectbox("Continente", list(BBOXES.keys()), index=0)
btn_trino = st.sidebar.button("📊 Cargar desde Trino", use_container_width=True, type="primary")
st.sidebar.caption("⚠️ 30-60 s · Los datos se cachean.")

# ================================================================
# CARGA DE DATOS
# ================================================================
if btn_trino:
    with st.spinner("⏳ Descargando flights_data4 y construyendo grafo..."):
        try:
            df_rutas = descargar_flights_data4(str(fecha_sim), cont_sim, user_sim)
            if df_rutas.empty:
                st.sidebar.error("Sin datos con origen+destino para ese día/región.")
            else:
                G       = construir_grafo(df_rutas)
                df_met  = calcular_centralidad(G)
                st.session_state["sim_G"]     = G
                st.session_state["sim_met"]   = df_met
                st.session_state["sim_rutas"] = df_rutas
                st.session_state["sim_label"] = f"Trino · {fecha_sim} · {cont_sim}"
                st.session_state["sim_fuente"]= "trino"
                st.sidebar.success(
                    f"✅ {len(df_rutas):,} vuelos · "
                    f"{len(G.nodes())} aeropuertos · {len(G.edges())} rutas"
                )
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            if "trino_conn" in st.session_state:
                del st.session_state["trino_conn"]

# Datos recibidos de otra página (live o histórico)
if usar_ext and datos_ext:
    df_ext = datos_ext["df"]
    fuente = datos_ext["fuente"]

    with st.spinner("⚙️ Procesando datos recibidos..."):
        if "origen" in df_ext.columns and "destino" in df_ext.columns:
            # Tiene rutas completas → grafo directo
            df_ext = df_ext.dropna(subset=["origen","destino"])
            df_ext = df_ext.rename(columns={"origen":"origen","destino":"destino"})
            if not df_ext.empty:
                G2     = construir_grafo(df_ext)
                df_met2 = calcular_centralidad(G2)
                st.session_state["sim_G"]      = G2
                st.session_state["sim_met"]    = df_met2
                st.session_state["sim_rutas"]  = df_ext
                st.session_state["sim_label"]  = datos_ext["label"]
                st.session_state["sim_fuente"] = fuente
        else:
            # Solo posiciones → estimar destinos
            df_dest = estimar_destinos_desde_posicion(df_ext, df_airports)
            if not df_dest.empty:
                df_dest = df_dest.rename(columns={"destino_est":"destino"})
                df_dest["origen"] = "ESTIMADO"
                G2      = nx.DiGraph()
                counts  = df_dest["destino"].value_counts().reset_index()
                counts.columns = ["destino","vuelos"]
                aero_info = df_airports.set_index("ident")[["name","latitude_deg","longitude_deg","type"]].to_dict("index")
                for _, row in counts.iterrows():
                    dst, w = row["destino"], row["vuelos"]
                    if dst in aero_info:
                        i = aero_info[dst]
                        G2.add_node(dst, nombre=i["name"], lat=i["latitude_deg"],
                                    lon=i["longitude_deg"], tipo=i["type"])
                        G2.add_edge("ORIGEN_MÚLTIPLE", dst, vuelos=w)
                df_met2  = calcular_centralidad(G2)
                df_rutas2 = df_dest
                st.session_state["sim_G"]      = G2
                st.session_state["sim_met"]    = df_met2
                st.session_state["sim_rutas"]  = df_rutas2
                st.session_state["sim_label"]  = datos_ext["label"] + " (destinos estimados)"
                st.session_state["sim_fuente"] = "live_estimado"

# ================================================================
# VISUALIZACIÓN PRINCIPAL
# ================================================================
G       = st.session_state.get("sim_G")
df_met  = st.session_state.get("sim_met",   pd.DataFrame())
df_rut  = st.session_state.get("sim_rutas", pd.DataFrame())
label   = st.session_state.get("sim_label", "")
fuente  = st.session_state.get("sim_fuente", "")

if G and not df_met.empty:
    st.success(
        f"📊 **{label}** · "
        f"{len(G.nodes())} aeropuertos · {len(G.edges())} rutas · "
        f"{df_rut.shape[0]:,} vuelos"
    )

    tab1, tab2, tab3 = st.tabs(["📈 Red & Centralidad", "🗺️ Mapa del Grafo", "💥 Simulador de Crisis"])

    # ──────────────────────────────────────────────────────────
    # TAB 1: RED & CENTRALIDAD
    # ──────────────────────────────────────────────────────────
    with tab1:
        st.subheader("🏆 Aeropuertos más críticos de la red")
        st.caption(
            "**Betweenness centrality**: fracción de rutas óptimas que pasan por cada aeropuerto. "
            "Alto betweenness = aeropuerto puente — si falla, muchas rutas quedan cortadas."
        )

        top20 = df_met.head(20).copy()
        top20["bet_%"] = (top20["betweenness"] * 100).round(2)

        col_g, col_t = st.columns([2,1])
        with col_g:
            fig_b = px.bar(
                top20, x="bet_%", y="icao", orientation="h",
                hover_name="nombre",
                hover_data={"vuelos_tot":True, "vuelos_in":True, "vuelos_out":True, "bet_%":True},
                color="bet_%", color_continuous_scale="Reds",
                labels={"bet_%":"Betweenness (%)", "icao":"Aeropuerto"},
                title="Top 20 — Betweenness Centrality"
            )
            fig_b.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=480, yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_b, use_container_width=True)

        with col_t:
            st.dataframe(
                top20[["icao","nombre","vuelos_tot","bet_%"]].rename(columns={
                    "icao":"ICAO","nombre":"Aeropuerto",
                    "vuelos_tot":"Vuelos","bet_%":"Betweenness (%)"
                }),
                use_container_width=True, height=450
            )

        st.divider()
        c_deg, c_bet = st.columns(2)
        with c_deg:
            fig_d = px.histogram(
                df_met, x="vuelos_tot", nbins=60,
                title="Distribución de conectividad (Degree)",
                labels={"vuelos_tot":"Total vuelos"}, color_discrete_sequence=["#1C83E1"]
            )
            fig_d.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                 font_color="white", height=280)
            st.plotly_chart(fig_d, use_container_width=True)

        with c_bet:
            fig_be = px.histogram(
                df_met, x="betweenness", nbins=60,
                title="Distribución de Betweenness Centrality",
                labels={"betweenness":"Betweenness"}, color_discrete_sequence=["#FF4B4B"]
            )
            fig_be.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  font_color="white", height=280)
            st.plotly_chart(fig_be, use_container_width=True)

    # ──────────────────────────────────────────────────────────
    # TAB 2: MAPA DEL GRAFO
    # ──────────────────────────────────────────────────────────
    with tab2:
        st.subheader("🗺️ Mapa de la red aérea")
        top_n = st.slider("Top N aeropuertos por criticidad:", 10, 150, 40, 10)
        top_a = df_met.head(top_n)

        fig_g = go.Figure()
        icaos_top = set(top_a["icao"])

        # Aristas
        for src, dst, data in G.edges(data=True):
            if src in icaos_top and dst in icaos_top:
                si = df_airports[df_airports["ident"]==src]
                di = df_airports[df_airports["ident"]==dst]
                if si.empty or di.empty: continue
                fig_g.add_trace(go.Scattermap(
                    lat=[si["latitude_deg"].values[0], di["latitude_deg"].values[0], None],
                    lon=[si["longitude_deg"].values[0], di["longitude_deg"].values[0], None],
                    mode="lines",
                    line=dict(width=max(0.5, data["vuelos"]/80), color="rgba(100,150,255,0.25)"),
                    hoverinfo="none", showlegend=False
                ))

        # Nodos
        max_b = top_a["betweenness"].max() or 1
        sizes = (10 + 28 * top_a["betweenness"] / max_b).round(0)
        fig_g.add_trace(go.Scattermap(
            lat=top_a["lat"], lon=top_a["lon"],
            mode="markers+text",
            text=top_a["icao"], textposition="top center",
            textfont=dict(size=9, color="white"),
            marker=go.scattermap.Marker(
                size=sizes, color=top_a["betweenness"],
                colorscale="Reds", showscale=True,
                colorbar=dict(title="Betweenness", thickness=12)
            ),
            hovertext=(
                "<b>" + top_a["nombre"] + "</b><br>" +
                "ICAO: " + top_a["icao"] + "<br>" +
                "Vuelos: " + top_a["vuelos_tot"].astype(str) + "<br>" +
                "Betweenness: " + (top_a["betweenness"]*100).round(2).astype(str) + "%"
            ),
            hoverinfo="text",
            name="Aeropuertos (tamaño = criticidad)"
        ))

        fig_g.update_layout(
            map_style="carto-darkmatter",
            margin={"r":0,"t":0,"l":0,"b":0}, height=650,
            showlegend=False,
            map=dict(center=dict(lat=top_a["lat"].mean(), lon=top_a["lon"].mean()), zoom=3)
        )
        st.plotly_chart(fig_g, use_container_width=True, config={"scrollZoom": True})

    # ──────────────────────────────────────────────────────────
    # TAB 3: SIMULADOR DE CRISIS
    # ──────────────────────────────────────────────────────────
    with tab3:
        st.subheader("💥 Simulador de Cierre / Reducción de Capacidad")

        col_cfg, col_res = st.columns([1, 2])

        with col_cfg:
            st.markdown("### ⚙️ Configurar escenario")

            # Selector de aeropuerto (top 100 por criticidad)
            opc = df_met.head(100).copy()
            opc["label"] = opc["icao"] + " — " + opc["nombre"].str[:35]
            aero_label = st.selectbox("Aeropuerto afectado:", opc["label"].tolist())
            icao_sim   = aero_label.split(" — ")[0]

            capacidad = st.slider(
                "Capacidad operativa restante (%):",
                0, 90, 0, 10,
                help="0% = cierre total · 50% = reducción a la mitad"
            )

            tipo_incidencia = st.selectbox(
                "Tipo de incidencia:",
                ["🌋 Cierre de espacio aéreo",
                 "🌧️ Condiciones meteorológicas extremas",
                 "🔧 Mantenimiento de emergencia",
                 "⚔️ Conflicto geopolítico / restricción",
                 "💻 Fallo de sistemas de control"],
                index=0
            )

            btn_sim = st.button("▶️ Ejecutar simulación",
                                use_container_width=True, type="primary")

        with col_res:
            if btn_sim:
                with st.spinner(f"Simulando escenario en {icao_sim}..."):
                    df_alt, resumen, n_af, co2_kg = simular_cierre(
                        G, df_met, df_airports, icao_sim, capacidad
                    )

                if resumen:
                    st.markdown(f"### 📋 Resultados: {tipo_incidencia}")
                    st.markdown(
                        f"**{resumen['nombre']}** ({icao_sim}) · "
                        f"Capacidad: **{capacidad}%** · "
                        f"Alternativa principal: **{resumen['alternativa']}** "
                        f"({resumen['dist_desvio_km']} km)"
                    )

                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("✈️ Vuelos afectados",   f"{resumen['vuelos_afectados']:,}")
                    k2.metric("📏 Km extra totales",   f"{resumen['km_extra']:,} km")
                    k3.metric("🌡️ CO₂ extra",          f"{resumen['co2_t']:,} t")
                    k4.metric("⛽ Combustible extra",  f"{resumen['fuel_t']:,} t")

                    # Mapa del escenario
                    info_c = df_airports[df_airports["ident"]==icao_sim]
                    fig_s  = go.Figure()

                    if not info_c.empty:
                        fig_s.add_trace(go.Scattermap(
                            lat=info_c["latitude_deg"], lon=info_c["longitude_deg"],
                            mode="markers+text", name="❌ Cerrado",
                            text=[icao_sim], textposition="top right",
                            marker=go.scattermap.Marker(size=24, color="red"),
                            hovertext=[f"❌ {resumen['nombre']} — Capacidad {capacidad}%"],
                            hoverinfo="text"
                        ))
                        fig_s.add_trace(go.Scattermap(
                            lat=df_alt["latitude_deg"], lon=df_alt["longitude_deg"],
                            mode="markers+text", name="✅ Alternativas",
                            text=df_alt["ident"], textposition="top right",
                            marker=go.scattermap.Marker(size=16, color="lime"),
                            hovertext=df_alt["name"] + "<br>" + df_alt["dist_km"].astype(str) + " km",
                            hoverinfo="text"
                        ))
                        lat_c = info_c["latitude_deg"].values[0]
                        lon_c = info_c["longitude_deg"].values[0]
                        for _, alt in df_alt.iterrows():
                            fig_s.add_trace(go.Scattermap(
                                lat=[lat_c, alt["latitude_deg"], None],
                                lon=[lon_c, alt["longitude_deg"], None],
                                mode="lines",
                                line=dict(width=2, color="rgba(0,255,100,0.5)"),
                                hoverinfo="none", showlegend=False
                            ))
                        fig_s.update_layout(
                            map_style="carto-darkmatter",
                            margin={"r":0,"t":0,"l":0,"b":0}, height=370,
                            showlegend=True,
                            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                                        bgcolor="rgba(0,0,0,0.6)", font=dict(color="white")),
                            map=dict(center=dict(lat=lat_c, lon=lon_c), zoom=4)
                        )
                        st.plotly_chart(fig_s, use_container_width=True)

                    # Tabla de alternativas
                    st.markdown("**🛬 Aeropuertos alternativos:**")
                    st.dataframe(
                        df_alt.rename(columns={"ident":"ICAO","name":"Aeropuerto",
                                               "type":"Tipo","dist_km":"Dist. (km)"}
                                      )[["ICAO","Aeropuerto","Tipo","Dist. (km)"]],
                        use_container_width=True
                    )

                    # Impacto medioambiental
                    st.divider()
                    st.subheader("🌱 Impacto Medioambiental")
                    co2_t       = resumen["co2_t"]
                    arboles     = int(co2_t * 1000 / 21.77)
                    km_coche    = int(co2_t * 1000 / 0.21)
                    coste_usd   = int(resumen["fuel_t"] * 800)

                    e1, e2, e3 = st.columns(3)
                    e1.metric("🌳 Árboles/año para compensar", f"{arboles:,}")
                    e2.metric("🚗 Equiv. en km de coche",      f"{km_coche:,}")
                    e3.metric("💵 Coste extra combustible",    f"${coste_usd:,}")

                    st.info(
                        f"El escenario **{tipo_incidencia}** en **{resumen['nombre']}** "
                        f"afecta a **{resumen['vuelos_afectados']:,} vuelos**, "
                        f"genera **{co2_t:,} t de CO₂ extra** "
                        f"equivalentes a plantar **{arboles:,} árboles** durante un año."
                    )
                else:
                    st.warning(
                        "No se encontraron vuelos con destino a este aeropuerto "
                        "en los datos cargados."
                    )
            else:
                st.info("👈 Configura el escenario y pulsa **Ejecutar simulación**.")

                # Metodología
                with st.expander("📖 Metodología del simulador"):
                    st.markdown("""
                    **Construcción del grafo**
                    - Nodos: aeropuertos con al menos un vuelo en el período analizado
                    - Aristas: rutas origen→destino con peso = número de vuelos

                    **Betweenness Centrality** (aproximado, k=200 muestras)
                    - Mide qué fracción de los caminos mínimos de la red pasan por cada nodo
                    - Un aeropuerto con alta betweenness es un *hub puente* — su cierre fragmenta la red

                    **Simulación de cierre**
                    - Identifica todos los vuelos con destino al aeropuerto afectado
                    - Aplica el % de reducción de capacidad
                    - Busca los 5 aeropuertos alternativos más cercanos
                    - Calcula el desvío en km y el impacto en CO₂

                    **Modelo de emisiones (ICAO)**
                    - Consumo medio: ~5 kg combustible / km (A320)
                    - Factor de emisión: 3,16 kg CO₂ / kg de queroseno
                    - **CO₂ extra ≈ vuelos × km de desvío × 16 kg/km**
                    """)

else:
    st.info(
        "**Para empezar:**\n\n"
        "1. Envía datos desde **Tiempo Real** o **Análisis Histórico** usando el botón "
        "🔬 *Enviar al Simulador*\n\n"
        "2. O descarga directamente desde Trino usando la barra lateral"
    )