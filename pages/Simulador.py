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

st.set_page_config(page_title="Simulador - TFG", layout="wide")
st.title("🔬 Simulador de Crisis Aérea")
st.caption("Modelado mediante grafos · Análisis de centralidad · Simulación de cierres · Impacto en CO₂")

# ================================================================
# CONSTANTES
# ================================================================
CO2_KG_POR_KM   = 16.0   # ~5 kg combustible/km × 3.16 kg CO2/kg (avión medio)
FUEL_KG_POR_KM  = 5.0    # kg combustible/km (A320 medio)

BBOXES = {
    'EU': (-25.0, 29.0,  45.0, 81.2),
    'NA': (-176.6, 7.4, -52.0, 83.1),
    'SA': (-109.4,-55.0, -32.4, 12.4),
    'AS': (  26.0,-12.2, 180.0, 80.8),
    'AF': ( -25.1,-34.8,  63.4, 37.2),
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
    return df

df_airports = cargar_aeropuertos()

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

# ================================================================
# FUNCIÓN: DESCARGA VUELOS DEL DÍA (flights_data4)
# ================================================================
@st.cache_data(show_spinner=False)
def descargar_vuelos_dia(fecha_str, continente, usuario):
    """
    Descarga todos los vuelos del día con origen Y destino conocidos.
    Filtra nulos en ambos campos — datos limpios para construir el grafo.
    """
    fecha = datetime.strptime(fecha_str, "%Y-%m-%d")
    dt_utc  = datetime(fecha.year, fecha.month, fecha.day, 0, 0, 0, tzinfo=timezone.utc)
    ts_day  = int(dt_utc.timestamp())

    lnmin, ltmin, lnmax, ltmax = BBOXES.get(continente, (-180,-90,180,90))

    # Aeropuertos del continente seleccionado para filtrar
    aeros_cont = df_airports[df_airports["continent"] == continente]["ident"].tolist()
    if not aeros_cont:
        return pd.DataFrame()

    conn = get_trino(usuario)

    query = f"""
        SELECT
            icao24,
            TRIM(callsign) AS callsign,
            estdepartureairport AS origen,
            estarrivalairport   AS destino,
            firstseen,
            lastseen
        FROM flights_data4
        WHERE day                  = {ts_day}
          AND estdepartureairport  IS NOT NULL
          AND estarrivalairport    IS NOT NULL
          AND estdepartureairport != estarrivalairport
    """
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    df = pd.DataFrame(rows, columns=cols)

    # Filtrar al menos un extremo en el continente elegido
    df = df[
        df["origen"].isin(aeros_cont) | df["destino"].isin(aeros_cont)
    ].reset_index(drop=True)

    return df

# ================================================================
# FUNCIÓN: CONSTRUIR GRAFO
# ================================================================
def construir_grafo(df_vuelos, df_airports):
    G = nx.DiGraph()

    # Añadir nodos con atributos geográficos
    aero_dict = df_airports.set_index("ident")[["name","latitude_deg","longitude_deg","type"]].to_dict("index")

    conteo_rutas = df_vuelos.groupby(["origen","destino"]).size().reset_index(name="vuelos")

    for _, row in conteo_rutas.iterrows():
        src, dst, w = row["origen"], row["destino"], row["vuelos"]

        if src not in G.nodes and src in aero_dict:
            info = aero_dict[src]
            G.add_node(src, nombre=info["name"], lat=info["latitude_deg"],
                       lon=info["longitude_deg"], tipo=info["type"])

        if dst not in G.nodes and dst in aero_dict:
            info = aero_dict[dst]
            G.add_node(dst, nombre=info["name"], lat=info["latitude_deg"],
                       lon=info["longitude_deg"], tipo=info["type"])

        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst, vuelos=w)

    return G

# ================================================================
# FUNCIÓN: MÉTRICAS DE CENTRALIDAD
# ================================================================
@st.cache_data(show_spinner=False)
def calcular_metricas(fecha_str, continente, usuario):
    df_vuelos = descargar_vuelos_dia(fecha_str, continente, usuario)
    if df_vuelos.empty:
        return None, pd.DataFrame(), pd.DataFrame()

    G = construir_grafo(df_vuelos, df_airports)

    # Degree: suma de vuelos entrantes + salientes
    degree_in  = dict(G.in_degree(weight="vuelos"))
    degree_out = dict(G.out_degree(weight="vuelos"))
    degree_tot = {n: degree_in.get(n,0) + degree_out.get(n,0) for n in G.nodes()}

    # Betweenness centrality (aproximado con k muestras — más rápido)
    k_samples = min(200, len(G.nodes()))
    betweenness = nx.betweenness_centrality(G, k=k_samples, weight="vuelos", normalized=True)

    df_metricas = pd.DataFrame({
        "icao":        list(degree_tot.keys()),
        "vuelos_tot":  [degree_tot[n] for n in degree_tot],
        "vuelos_in":   [degree_in.get(n,0) for n in degree_tot],
        "vuelos_out":  [degree_out.get(n,0) for n in degree_tot],
        "betweenness": [betweenness.get(n,0) for n in degree_tot],
    })

    # Añadir nombre del aeropuerto
    aero_nombres = df_airports.set_index("ident")["name"].to_dict()
    aero_lat     = df_airports.set_index("ident")["latitude_deg"].to_dict()
    aero_lon     = df_airports.set_index("ident")["longitude_deg"].to_dict()
    df_metricas["nombre"] = df_metricas["icao"].map(aero_nombres).fillna(df_metricas["icao"])
    df_metricas["lat"]    = df_metricas["icao"].map(aero_lat)
    df_metricas["lon"]    = df_metricas["icao"].map(aero_lon)
    df_metricas = df_metricas.dropna(subset=["lat","lon"])
    df_metricas = df_metricas.sort_values("betweenness", ascending=False).reset_index(drop=True)

    return G, df_metricas, df_vuelos

# ================================================================
# FUNCIÓN: DISTANCIA HAVERSINE
# ================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ================================================================
# FUNCIÓN: SIMULACIÓN DE CIERRE
# ================================================================
def simular_cierre(G, df_metricas, df_airports, icao_cerrado, capacidad_pct):
    """
    Simula el cierre (o reducción de capacidad) de un aeropuerto.
    - capacidad_pct=0  → cierre total
    - capacidad_pct=50 → solo acepta 50% de los vuelos previstos
    """
    if icao_cerrado not in G.nodes:
        return pd.DataFrame(), {}, 0, 0, 0

    # Vuelos con destino al aeropuerto afectado
    vuelos_afectados = [
        {"origen": src, "destino": dst, "vuelos": data["vuelos"]}
        for src, dst, data in G.in_edges(icao_cerrado, data=True)
    ]

    if capacidad_pct > 0:
        # Solo afecta el exceso de capacidad
        vuelos_afectados = [
            {**v, "vuelos": max(0, int(v["vuelos"] * (1 - capacidad_pct/100)))}
            for v in vuelos_afectados
        ]
    vuelos_afectados = [v for v in vuelos_afectados if v["vuelos"] > 0]

    total_vuelos_afectados = sum(v["vuelos"] for v in vuelos_afectados)

    if total_vuelos_afectados == 0:
        return pd.DataFrame(), {}, 0, 0, 0

    # Posición del aeropuerto cerrado
    info_cerrado = df_airports[df_airports["ident"] == icao_cerrado]
    if info_cerrado.empty:
        return pd.DataFrame(), {}, 0, 0, 0

    lat_c = info_cerrado["latitude_deg"].values[0]
    lon_c = info_cerrado["longitude_deg"].values[0]

    # Aeropuertos alternativos (excluir el cerrado, solo large/medium)
    df_alt = df_airports[
        (df_airports["ident"] != icao_cerrado) &
        (df_airports["type"].isin(["large_airport","medium_airport"]))
    ].copy().dropna(subset=["latitude_deg","longitude_deg"])

    # Top 5 alternativos más cercanos al aeropuerto cerrado
    coords_alt = np.radians(df_alt[["latitude_deg","longitude_deg"]].values)
    punto      = np.radians([[lat_c, lon_c]])
    tree       = BallTree(coords_alt, metric="haversine")
    dists, idxs = tree.query(punto, k=min(5, len(df_alt)))

    df_alternativas = df_alt.iloc[idxs[0]].copy()
    df_alternativas["dist_al_cerrado_km"] = (dists[0] * 6371).round(1)
    df_alternativas = df_alternativas[["ident","name","type","latitude_deg","longitude_deg","dist_al_cerrado_km"]]

    # Aeropuerto alternativo principal (el más cercano)
    mejor_alt     = df_alternativas.iloc[0]
    dist_desvio   = mejor_alt["dist_al_cerrado_km"]

    # Impacto total
    km_extra_total   = total_vuelos_afectados * dist_desvio
    co2_extra_kg     = km_extra_total * CO2_KG_POR_KM
    fuel_extra_kg    = km_extra_total * FUEL_KG_POR_KM

    resumen = {
        "aeropuerto_cerrado": icao_cerrado,
        "capacidad_operativa": f"{capacidad_pct}%",
        "vuelos_afectados":    total_vuelos_afectados,
        "alternativa_principal": mejor_alt["name"],
        "dist_desvio_km":      dist_desvio,
        "km_extra_total":      round(km_extra_total),
        "co2_extra_toneladas": round(co2_extra_kg / 1000, 1),
        "fuel_extra_toneladas": round(fuel_extra_kg / 1000, 1),
    }

    return df_alternativas, resumen, total_vuelos_afectados, co2_extra_kg, km_extra_total

# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.header("🔑 Conexión OpenSky")
user_sim = st.sidebar.text_input("Usuario Trino (email)", value="jaltevil@myuax.com").lower()

st.sidebar.divider()
st.sidebar.header("📅 Datos a analizar")
fecha_sim  = st.sidebar.date_input("Día", datetime(2024, 1, 16))
cont_sim   = st.sidebar.selectbox("Región", list(BBOXES.keys()), index=0)

btn_cargar = st.sidebar.button("📊 Cargar datos y construir grafo",
                               use_container_width=True, type="primary")
st.sidebar.caption("⚠️ Puede tardar 30-60 s. Los datos se cachean para el resto de la sesión.")

# ================================================================
# PASO 1: CARGAR DATOS
# ================================================================
if btn_cargar:
    if not user_sim:
        st.sidebar.error("Introduce tu usuario.")
    else:
        with st.spinner("⏳ Descargando vuelos de flights_data4 y construyendo grafo..."):
            G, df_metricas, df_vuelos = calcular_metricas(str(fecha_sim), cont_sim, user_sim)
            if df_vuelos is not None and not df_vuelos.empty:
                st.session_state["sim_G"]         = G
                st.session_state["sim_metricas"]  = df_metricas
                st.session_state["sim_vuelos"]    = df_vuelos
                st.session_state["sim_fecha"]     = str(fecha_sim)
                st.session_state["sim_continente"]= cont_sim
                st.sidebar.success(f"✅ {len(df_vuelos):,} vuelos · {len(G.nodes())} aeropuertos · {len(G.edges())} rutas")
            else:
                st.sidebar.error("No se encontraron vuelos con origen y destino conocidos.")

G           = st.session_state.get("sim_G")
df_metricas = st.session_state.get("sim_metricas", pd.DataFrame())
df_vuelos   = st.session_state.get("sim_vuelos",   pd.DataFrame())

# ================================================================
# PASO 2: ANÁLISIS DEL GRAFO (si hay datos)
# ================================================================
if G and not df_metricas.empty:
    fecha_label = st.session_state.get("sim_fecha", "")
    cont_label  = st.session_state.get("sim_continente", "")

    st.success(f"📊 Grafo construido · **{fecha_label}** · **{cont_label}** · "
               f"{len(G.nodes())} aeropuertos · {len(G.edges())} rutas · "
               f"{df_vuelos['vuelos_tot'].sum() if 'vuelos_tot' in df_vuelos.columns else df_metricas['vuelos_tot'].sum():,} vuelos totales")

    tab1, tab2, tab3 = st.tabs(["📈 Aeropuertos Críticos", "🗺️ Mapa del Grafo", "💥 Simulador de Crisis"])

    # ----------------------------------------------------------
    # TAB 1: AEROPUERTOS CRÍTICOS
    # ----------------------------------------------------------
    with tab1:
        st.subheader("🏆 Aeropuertos más críticos de la red")
        st.caption("**Betweenness centrality**: cuántos caminos óptimos pasan por este aeropuerto. "
                   "Un valor alto = aeropuerto puente crítico.")

        col_a, col_b = st.columns([2, 1])

        with col_a:
            top20 = df_metricas.head(20).copy()
            top20["betweenness_%"] = (top20["betweenness"] * 100).round(2)

            fig_crit = px.bar(
                top20, x="betweenness_%", y="icao",
                orientation="h", hover_name="nombre",
                hover_data={"vuelos_tot": True, "vuelos_in": True, "vuelos_out": True},
                color="betweenness_%", color_continuous_scale="Reds",
                title="Top 20 aeropuertos por Betweenness Centrality",
                labels={"betweenness_%": "Betweenness (%)", "icao": "Aeropuerto"}
            )
            fig_crit.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=500, yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_crit, use_container_width=True)

        with col_b:
            st.dataframe(
                top20[["icao","nombre","vuelos_tot","betweenness_%"]].rename(columns={
                    "icao": "ICAO", "nombre": "Aeropuerto",
                    "vuelos_tot": "Vuelos", "betweenness_%": "Betweenness (%)"
                }),
                use_container_width=True, height=460
            )

        st.divider()
        st.subheader("📊 Distribución de conectividad (Degree)")
        fig_deg = px.histogram(
            df_metricas, x="vuelos_tot", nbins=60,
            title="Distribución del número de vuelos por aeropuerto",
            labels={"vuelos_tot": "Total vuelos (entrada + salida)"},
            color_discrete_sequence=["#1C83E1"]
        )
        fig_deg.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=300
        )
        st.plotly_chart(fig_deg, use_container_width=True)

    # ----------------------------------------------------------
    # TAB 2: MAPA DEL GRAFO
    # ----------------------------------------------------------
    with tab2:
        st.subheader("🗺️ Mapa de la red aérea")

        top_n = st.slider("Mostrar top N aeropuertos por centralidad:", 10, 100, 30, step=10)
        top_aeros = df_metricas.head(top_n)

        fig_g = go.Figure()

        # Aristas (rutas) — solo las que conectan los top aeropuertos
        icaos_top = set(top_aeros["icao"])
        for src, dst, data in G.edges(data=True):
            if src in icaos_top and dst in icaos_top:
                src_info = df_airports[df_airports["ident"] == src]
                dst_info = df_airports[df_airports["ident"] == dst]
                if src_info.empty or dst_info.empty:
                    continue
                fig_g.add_trace(go.Scattermap(
                    lat=[src_info["latitude_deg"].values[0], dst_info["latitude_deg"].values[0], None],
                    lon=[src_info["longitude_deg"].values[0], dst_info["longitude_deg"].values[0], None],
                    mode="lines",
                    line=dict(width=max(0.5, data["vuelos"]/50), color="rgba(100,150,255,0.3)"),
                    hoverinfo="none", showlegend=False
                ))

        # Nodos (aeropuertos) — tamaño proporcional a betweenness
        max_bet = top_aeros["betweenness"].max() or 1
        sizes   = (10 + 25 * top_aeros["betweenness"] / max_bet).round(0)

        fig_g.add_trace(go.Scattermap(
            lat=top_aeros["lat"],
            lon=top_aeros["lon"],
            mode="markers+text",
            text=top_aeros["icao"],
            textposition="top center",
            textfont=dict(size=9, color="white"),
            marker=go.scattermap.Marker(
                size=sizes,
                color=top_aeros["betweenness"],
                colorscale="Reds",
                showscale=True,
                colorbar=dict(title="Betweenness", thickness=12),
            ),
            hovertext=(
                "<b>" + top_aeros["nombre"] + "</b><br>" +
                "ICAO: " + top_aeros["icao"] + "<br>" +
                "Vuelos: " + top_aeros["vuelos_tot"].astype(str) + "<br>" +
                "Betweenness: " + (top_aeros["betweenness"]*100).round(2).astype(str) + "%"
            ),
            hoverinfo="text",
            name="Aeropuertos (tamaño = criticidad)"
        ))

        fig_g.update_layout(
            map_style="carto-darkmatter",
            margin={"r":0,"t":0,"l":0,"b":0},
            height=620,
            showlegend=False,
            map=dict(center=dict(lat=top_aeros["lat"].mean(), lon=top_aeros["lon"].mean()), zoom=3)
        )
        st.plotly_chart(fig_g, use_container_width=True, config={"scrollZoom": True})

    # ----------------------------------------------------------
    # TAB 3: SIMULADOR DE CRISIS
    # ----------------------------------------------------------
    with tab3:
        st.subheader("💥 Simulador de Cierre / Reducción de Capacidad")

        col_s1, col_s2 = st.columns([1, 2])

        with col_s1:
            st.markdown("**Configura el escenario:**")

            # Desplegable con aeropuertos del grafo (ordenados por criticidad)
            opciones_sim = df_metricas[["icao","nombre"]].head(100)
            opciones_sim["label"] = opciones_sim["icao"] + " — " + opciones_sim["nombre"].str[:35]
            aero_label = st.selectbox(
                "Aeropuerto afectado:",
                opciones_sim["label"].tolist()
            )
            icao_sim = aero_label.split(" — ")[0]

            capacidad_sim = st.slider(
                "Capacidad operativa restante (%):",
                min_value=0, max_value=90, value=0, step=10,
                help="0% = cierre total · 50% = reducción a la mitad · 90% = impacto mínimo"
            )

            escenario_tipo = st.radio(
                "Tipo de incidencia:",
                ["🌋 Cierre de espacio aéreo", "🌧️ Condiciones meteorológicas",
                 "🔧 Mantenimiento de emergencia", "⚔️ Conflicto geopolítico"],
                index=0
            )

            btn_simular = st.button("▶️ Ejecutar simulación", use_container_width=True, type="primary")

        with col_s2:
            if btn_simular:
                with st.spinner(f"Simulando cierre de {icao_sim}..."):
                    df_alt, resumen, n_afectados, co2_kg, km_extra = simular_cierre(
                        G, df_metricas, df_airports, icao_sim, capacidad_sim
                    )

                if resumen:
                    # KPIs de impacto
                    st.markdown(f"### Impacto del escenario: {escenario_tipo}")
                    st.markdown(f"**{icao_sim}** — capacidad operativa: **{capacidad_sim}%**")

                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("✈️ Vuelos afectados",     f"{n_afectados:,}")
                    k2.metric("📏 Km extra totales",     f"{resumen['km_extra_total']:,} km")
                    k3.metric("🌡️ CO₂ extra",            f"{resumen['co2_extra_toneladas']:,} t")
                    k4.metric("⛽ Combustible extra",    f"{resumen['fuel_extra_toneladas']:,} t")

                    st.divider()

                    col_alt1, col_alt2 = st.columns([1, 1])

                    with col_alt1:
                        st.markdown("**🛬 Aeropuertos alternativos más cercanos:**")
                        st.dataframe(
                            df_alt.rename(columns={
                                "ident": "ICAO", "name": "Aeropuerto",
                                "type": "Tipo", "dist_al_cerrado_km": "Distancia (km)"
                            })[["ICAO","Aeropuerto","Tipo","Distancia (km)"]],
                            use_container_width=True
                        )

                    with col_alt2:
                        st.markdown("**🌍 Mapa del escenario:**")
                        # Aeropuerto cerrado
                        info_c = df_airports[df_airports["ident"] == icao_sim]
                        fig_sim = go.Figure()

                        if not info_c.empty:
                            # Aeropuerto cerrado (rojo)
                            fig_sim.add_trace(go.Scattermap(
                                lat=info_c["latitude_deg"], lon=info_c["longitude_deg"],
                                mode="markers+text", name="❌ Cerrado",
                                text=[icao_sim], textposition="top right",
                                marker=go.scattermap.Marker(size=22, color="red"),
                                hovertext=[f"❌ CERRADO: {info_c['name'].values[0]}"],
                                hoverinfo="text"
                            ))

                            # Alternativas (verde)
                            fig_sim.add_trace(go.Scattermap(
                                lat=df_alt["latitude_deg"], lon=df_alt["longitude_deg"],
                                mode="markers+text", name="✅ Alternativas",
                                text=df_alt["ident"], textposition="top right",
                                marker=go.scattermap.Marker(size=16, color="lime"),
                                hovertext=df_alt["name"] + "<br>" + df_alt["dist_al_cerrado_km"].astype(str) + " km",
                                hoverinfo="text"
                            ))

                            # Líneas de desvío
                            lat_c = info_c["latitude_deg"].values[0]
                            lon_c = info_c["longitude_deg"].values[0]
                            for _, alt in df_alt.iterrows():
                                fig_sim.add_trace(go.Scattermap(
                                    lat=[lat_c, alt["latitude_deg"], None],
                                    lon=[lon_c, alt["longitude_deg"], None],
                                    mode="lines",
                                    line=dict(width=2, color="rgba(0,255,0,0.5)"),
                                    hoverinfo="none", showlegend=False
                                ))

                        fig_sim.update_layout(
                            map_style="carto-darkmatter",
                            margin={"r":0,"t":0,"l":0,"b":0}, height=350,
                            showlegend=True,
                            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                                        bgcolor="rgba(0,0,0,0.6)", font=dict(color="white")),
                            map=dict(
                                center=dict(lat=info_c["latitude_deg"].values[0] if not info_c.empty else 40,
                                            lon=info_c["longitude_deg"].values[0] if not info_c.empty else -3),
                                zoom=4
                            )
                        )
                        st.plotly_chart(fig_sim, use_container_width=True)

                    # Análisis de equivalencias del CO2
                    st.divider()
                    st.subheader("🌱 Impacto Medioambiental")
                    co2_t = resumen["co2_extra_toneladas"]
                    arboles_eq = int(co2_t * 1000 / 21.77)  # un árbol absorbe ~21.77 kg CO2/año
                    coches_km  = int(co2_t * 1000 / 0.21)   # coche medio: ~0.21 kg CO2/km

                    e1, e2, e3 = st.columns(3)
                    e1.metric("🌳 Árboles/año para compensar", f"{arboles_eq:,}")
                    e2.metric("🚗 Equivalente en km de coche", f"{coches_km:,}")
                    e3.metric("⛽ Coste combustible extra (~0.8$/kg)", f"${resumen['fuel_extra_toneladas']*800:,.0f}")

                    st.info(
                        f"El cierre de **{icao_sim}** con capacidad al **{capacidad_sim}%** "
                        f"afecta a **{n_afectados:,} vuelos**, genera **{co2_t:,} toneladas** de CO₂ extra "
                        f"equivalentes a **{arboles_eq:,} árboles** absorbiendo CO₂ durante un año completo."
                    )
                else:
                    st.warning("No se encontraron vuelos con destino a este aeropuerto en los datos del día seleccionado.")
            else:
                st.info("👈 Configura el escenario y pulsa **Ejecutar simulación**.")

else:
    st.info("👈 Selecciona un día y región en la barra lateral y pulsa **Cargar datos y construir grafo** para empezar.")

    # Explicación metodológica
    st.divider()
    with st.expander("📖 ¿Cómo funciona el simulador?"):
        st.markdown("""
        ### Metodología

        **1. Construcción del grafo**
        - Se descargan todos los vuelos del día seleccionado desde `flights_data4` (OpenSky/Trino)
        - Se filtran los registros con **origen Y destino conocidos** (sin nulos)
        - Se construye un grafo dirigido donde: **nodos = aeropuertos**, **aristas = rutas**, **peso = número de vuelos**

        **2. Análisis de centralidad**
        - **Degree centrality**: número total de vuelos que pasan por cada aeropuerto (entrada + salida)
        - **Betweenness centrality**: fracción de los caminos óptimos de la red que pasan por cada nodo.
          Un aeropuerto con alta betweenness es un "puente" crítico — si falla, muchas rutas quedan cortadas.

        **3. Simulación de crisis**
        - Se selecciona un aeropuerto y un % de capacidad operativa restante
        - Se identifican todos los vuelos con ese aeropuerto como **destino**
        - Se buscan los **5 aeropuertos alternativos más cercanos** (large/medium)
        - Se calcula el **desvío en km** y el **impacto en CO₂** usando el modelo ICAO de emisiones

        **4. Cálculo de emisiones**
        - Consumo medio: ~5 kg combustible/km (aeronave narrow-body tipo A320)
        - Factor de emisión: 3.16 kg CO₂/kg de queroseno (IPCC)
        - **CO₂ extra ≈ vuelos afectados × km de desvío × 16 kg/km**
        """)