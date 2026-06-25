"""
pages/6_Analisis_Red.py
TFG: Analisis de la red aerea - grafo, centralidad y aislamiento
Jacob Altenburger Villar - UAX 2026

Construye el grafo dirigido de la red aerea de un dia real (Trino / flights_data4),
calcula betweenness y degree para identificar los aeropuertos criticos, y analiza el
aislamiento geografico (distancia al vecino mas cercano). Los resultados quedan en
sesion para poder enviar un aeropuerto critico al Simulador.
"""
import io
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from sklearn.neighbors import BallTree
from trino.dbapi import connect
from trino.auth import OAuth2Authentication
from datetime import datetime, timezone
from pathlib import Path

# --- Rutas de datos robustas: encuentra airports.csv y los .xlsx tanto si
# --- ejecutas la app desde la raiz del proyecto como si abres la pagina sola.
_AQUI = Path(__file__).resolve().parent
def _ruta_datos(nombre):
    for _base in (_AQUI, _AQUI.parent, Path.cwd()):
        _p = _base / nombre
        if _p.exists():
            return str(_p)
    return nombre  # si no esta en ningun sitio, deja que pandas avise

st.set_page_config(page_title="Analisis de Red", page_icon="🕸️", layout="wide")

# Continentes: filtran que aeropuertos entran en el grafo
CONTINENTES = {"EU": "Europa", "NA": "Norteamerica", "SA": "Sudamerica",
               "AS": "Asia", "AF": "Africa", "OC": "Oceania"}


# ================================================================
# AEROPUERTOS + AISLAMIENTO (BallTree, vecino mas cercano)
# ================================================================
@st.cache_data(show_spinner=False)
def cargar_aeropuertos():
    df = pd.read_csv(_ruta_datos("airports.csv"))
    df = df[df["type"].isin(["small_airport", "medium_airport", "large_airport"])].copy()
    df = df.dropna(subset=["latitude_deg", "longitude_deg"])
    df["continent"] = df["continent"].fillna("NA")
    # distancia de cada aeropuerto a su vecino mas cercano (haversine sobre la esfera)
    coords = np.radians(df[["latitude_deg", "longitude_deg"]].values)
    tree = BallTree(coords, metric="haversine")
    dists, _ = tree.query(coords, k=2)              # k=2: el 1o es el propio aeropuerto
    df["distancia_vecino_km"] = (dists[:, 1] * 6371).round(1)
    return df.reset_index(drop=True)


df_ap = cargar_aeropuertos()


# ================================================================
# TRINO
# ================================================================
def get_trino(usuario):
    if "trino_conn" not in st.session_state or st.session_state.get("trino_user") != usuario:
        st.session_state.trino_conn = connect(
            host="trino.opensky-network.org", port=443,
            user=usuario, auth=OAuth2Authentication(),
            http_scheme="https", catalog="minio", schema="osky", request_timeout=120.0)
        st.session_state.trino_user = usuario
    return st.session_state.trino_conn


@st.cache_data(show_spinner=False)
def query_rutas_dia(fecha, usuario):
    """Todas las rutas (origen, destino, nº de vuelos) de un dia. Se agrega en Trino."""
    ts_day = int(datetime(fecha.year, fecha.month, fecha.day, 0, 0, 0,
                          tzinfo=timezone.utc).timestamp())
    conn = get_trino(usuario)
    q = f"""
        SELECT estdepartureairport AS origen,
               estarrivalairport   AS destino,
               COUNT(*)            AS vuelos
        FROM flights_data4
        WHERE day={ts_day}
          AND estdepartureairport IS NOT NULL
          AND estarrivalairport   IS NOT NULL
        GROUP BY estdepartureairport, estarrivalairport
    """
    cur = conn.cursor()
    cur.execute(q)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


# ================================================================
# GRAFO + CENTRALIDAD
# ================================================================
@st.cache_data(show_spinner=False)
def construir_grafo(rutas_tuple, continentes, exacto=True, k_muestra=100):
    """
    rutas_tuple: tupla de (origen, destino, vuelos) -> hashable para cachear.
    Filtra a los aeropuertos de los continentes elegidos, monta el DiGraph y
    calcula betweenness (muestreo k) y degree ponderados por nº de vuelos.
    """
    rutas = pd.DataFrame(rutas_tuple, columns=["origen", "destino", "vuelos"])
    ap = df_ap[df_ap["continent"].isin(continentes)]
    info = ap.set_index("ident")[["name", "latitude_deg", "longitude_deg", "type"]].to_dict("index")
    validos = set(info.keys())
    rutas = rutas[rutas["origen"].isin(validos) & rutas["destino"].isin(validos)]
    if rutas.empty:
        return pd.DataFrame(), [], {}

    G = nx.DiGraph()
    for _, r in rutas.iterrows():
        for n in (r["origen"], r["destino"]):
            if n not in G:
                i = info[n]
                G.add_node(n, nombre=i["name"], lat=i["latitude_deg"],
                           lon=i["longitude_deg"], tipo=i["type"])
        v = int(r["vuelos"])
        # 'inv' = 1/vuelos: en la betweenness el peso es una distancia (menor = camino
        # preferido), asi que invertimos para que las rutas con mas vuelos cuenten como
        # las "mas cortas". Si se usara 'vuelos' directamente, los hubs quedarian fuera
        # de los caminos minimos y darian betweenness practicamente cero.
        G.add_edge(r["origen"], r["destino"], vuelos=v, inv=1.0 / v)

    if G.number_of_nodes() < 2:
        return pd.DataFrame(), [], {}

    if exacto or not k_muestra:
        # betweenness exacta: todos los aeropuertos como origen (deterministica)
        bet = nx.betweenness_centrality(G, weight="inv", normalized=True)
    else:
        k = min(k_muestra, max(2, G.number_of_nodes() - 1))
        bet = nx.betweenness_centrality(G, k=k, weight="inv", normalized=True, seed=42)
    din = dict(G.in_degree(weight="vuelos"))
    dout = dict(G.out_degree(weight="vuelos"))

    rows = []
    for n in G.nodes():
        i = info.get(n, {})
        rows.append({"icao": n, "nombre": i.get("name", n),
                     "lat": i.get("latitude_deg"), "lon": i.get("longitude_deg"),
                     "tipo": i.get("type", ""), "betweenness": bet.get(n, 0.0),
                     "llegadas": int(din.get(n, 0)), "salidas": int(dout.get(n, 0)),
                     "vuelos_tot": int(din.get(n, 0) + dout.get(n, 0))})
    df_met = (pd.DataFrame(rows).dropna(subset=["lat", "lon"])
              .sort_values("betweenness", ascending=False).reset_index(drop=True))
    edges = [{"src": s, "dst": d, "vuelos": dd["vuelos"]} for s, d, dd in G.edges(data=True)]
    stats = {"nodos": G.number_of_nodes(), "aristas": G.number_of_edges(),
             "vuelos": int(rutas["vuelos"].sum())}
    return df_met, edges, stats


# ================================================================
# CABECERA
# ================================================================
st.markdown("## 🕸️ Analisis de Red - aeropuertos criticos y aislamiento")
st.caption("Construye el grafo de un dia real, identifica los nodos puente (betweenness) "
           "y los aeropuertos mas aislados. Fuente: OpenSky / Trino (flights_data4).")

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.header("🔑 Conexion Trino")
    user_trino = st.text_input("Usuario (email)", value="jaltevil@myuax.com").lower()
    st.divider()
    st.header("📅 Dia a analizar (UTC)")
    fecha_sel = st.date_input("Dia", datetime(2024, 1, 16))
    conts = st.multiselect("Continentes en el grafo:", list(CONTINENTES.keys()),
                           default=["EU"], format_func=lambda c: CONTINENTES[c])
    modo_bet = st.radio("Calculo de betweenness:", ["Exacto", "Aproximado"], index=0,
                        help="Exacto: usa todos los aeropuertos como origen, mas preciso (para "
                             "Europa tarda unos segundos). Aproximado: usa una muestra de k "
                             "aeropuertos, mas rapido para redes enormes.")
    if modo_bet == "Aproximado":
        k_muestra = st.slider("Muestreo (k)", 20, 300, 100, 10,
                              help="Nº de aeropuertos de partida para estimar la betweenness.")
    else:
        k_muestra = None
    st.divider()
    btn = st.button("🕸️ Construir red y analizar", type="primary", use_container_width=True)

if btn:
    if not conts:
        st.sidebar.error("Elige al menos un continente.")
        st.stop()
    try:
        with st.spinner("Consultando rutas del dia en Trino..."):
            rutas = query_rutas_dia(fecha_sel, user_trino)
        if rutas.empty:
            st.warning("Trino no devolvio rutas para ese dia.")
            st.stop()
        with st.spinner("Construyendo grafo y calculando centralidad..."):
            rutas_t = tuple(rutas[["origen", "destino", "vuelos"]].itertuples(index=False, name=None))
            df_met, edges, stats = construir_grafo(rutas_t, tuple(conts),
                                                   exacto=(modo_bet == "Exacto"), k_muestra=k_muestra)
        st.session_state["red_met"] = df_met
        st.session_state["red_edges"] = edges
        st.session_state["red_stats"] = stats
        st.session_state["red_fecha"] = str(fecha_sel)
        st.session_state["red_conts"] = conts
    except Exception as e:
        st.error(f"Error al consultar Trino: {e}")
        if "trino_conn" in st.session_state:
            del st.session_state["trino_conn"]
        st.stop()

df_met = st.session_state.get("red_met")
if df_met is None or df_met.empty:
    st.info("Configura el dia y los continentes en la barra lateral y pulsa "
            "**Construir red y analizar**. Tambien puedes analizar el aislamiento mas abajo "
            "sin construir el grafo.")
else:
    stats = st.session_state.get("red_stats", {})
    st.success(f"Red del {st.session_state.get('red_fecha','')} · "
               f"continentes: {', '.join(st.session_state.get('red_conts', []))}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Aeropuertos (nodos)", f"{stats.get('nodos', 0):,}")
    c2.metric("Rutas (aristas)", f"{stats.get('aristas', 0):,}")
    c3.metric("Vuelos del dia", f"{stats.get('vuelos', 0):,}")

    # ============================================================
    # 1) AEROPUERTOS CRITICOS (BETWEENNESS)
    # ============================================================
    st.divider()
    st.subheader("1 · Aeropuertos criticos (betweenness)")
    st.caption("Betweenness = fraccion de rutas optimas de la red que pasan por ese aeropuerto. "
               "Un valor alto significa que es un nodo puente: su cierre alarga o desconecta muchas "
               "rutas, aunque el aeropuerto no sea de los que mas vuelos mueve.")

    top_n = st.slider("Top N aeropuertos a mostrar", 5, 40, 20, 5)
    top = df_met.head(top_n).copy()
    top["bet_%"] = (top["betweenness"] * 100).round(2)

    col_g, col_t = st.columns([3, 2])
    with col_g:
        fig_bar = px.bar(top.sort_values("bet_%"), x="bet_%", y="icao", orientation="h",
                         color="bet_%", color_continuous_scale="YlOrRd",
                         labels={"bet_%": "Betweenness (%)", "icao": "Aeropuerto"},
                         hover_data={"nombre": True})
        fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font_color="white", height=520, coloraxis_showscale=False,
                              margin=dict(t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_t:
        st.dataframe(
            top[["icao", "nombre", "bet_%", "vuelos_tot"]].rename(
                columns={"icao": "ICAO", "nombre": "Aeropuerto",
                         "bet_%": "Betweenness (%)", "vuelos_tot": "Vuelos"}),
            use_container_width=True, height=520, hide_index=True)

    # Mapa: nodos dimensionados por betweenness (top 150 para que sea fluido)
    top_map = df_met.head(150)
    max_b = top_map["betweenness"].max() or 1
    fig_m = go.Figure()
    fig_m.add_trace(go.Scattermap(
        lat=top_map["lat"], lon=top_map["lon"], mode="markers",
        marker=go.scattermap.Marker(
            size=(8 + 30 * top_map["betweenness"] / max_b).round(0),
            color=top_map["betweenness"], colorscale="YlOrRd",
            colorbar=dict(title="Betw.", thickness=12), opacity=0.9),
        text=(top_map["nombre"] + " (" + top_map["icao"] + ")<br>Betweenness: " +
              (top_map["betweenness"] * 100).round(2).astype(str) + "%<br>Vuelos: " +
              top_map["vuelos_tot"].astype(str)),
        hoverinfo="text", name="Aeropuertos"))
    fig_m.update_layout(
        map_style="carto-darkmatter", margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=540,
        map=dict(center=dict(lat=top_map["lat"].mean(), lon=top_map["lon"].mean()), zoom=2.8),
        showlegend=False)
    st.plotly_chart(fig_m, use_container_width=True)

    # ============================================================
    # 2) VOLUMEN DE TRAFICO (DEGREE)
    # ============================================================
    st.divider()
    st.subheader("2 · Volumen de trafico (degree)")
    st.caption("Degree = nº total de vuelos que entran y salen del aeropuerto. Identifica los "
               "grandes hubs por volumen directo. Comparado con la betweenness, ayuda a separar "
               "los aeropuertos grandes (mucho trafico propio) de los aeropuertos puente.")
    top_deg = df_met.sort_values("vuelos_tot", ascending=False).head(top_n)
    fig_deg = px.bar(top_deg.sort_values("vuelos_tot"), x="vuelos_tot", y="icao",
                     orientation="h", color="vuelos_tot", color_continuous_scale="Blues",
                     labels={"vuelos_tot": "Vuelos (llegadas + salidas)", "icao": "Aeropuerto"},
                     hover_data={"nombre": True, "llegadas": True, "salidas": True})
    fig_deg.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="white", height=480, coloraxis_showscale=False,
                          margin=dict(t=10, b=10))
    st.plotly_chart(fig_deg, use_container_width=True)

    # ============================================================
    # ENVIAR UN AEROPUERTO CRITICO AL SIMULADOR
    # ============================================================
    st.divider()
    st.markdown("**Enviar al simulador**")
    st.caption("Elige un aeropuerto critico para estudiarlo en el Simulador como un cierre.")
    opts = (df_met["nombre"] + " (" + df_met["icao"] + ")").head(40).tolist()
    sel = st.selectbox("Aeropuerto:", opts)
    if st.button("➡️ Enviar al Simulador"):
        st.session_state["sim_icao_preset"] = sel.split("(")[-1].rstrip(")")
        st.success(f"Listo. Abre la pagina **Simulador**: aparecera {st.session_state['sim_icao_preset']} preseleccionado.")

# ============================================================
# 3) AISLAMIENTO GEOGRAFICO (no necesita el grafo)
# ============================================================
st.divider()
st.subheader("3 · Aislamiento geografico")
st.caption("Distancia de cada aeropuerto a su vecino mas cercano. Los mas aislados son "
           "criticos en una crisis: si hay que desviar un vuelo, el alternativo mas proximo "
           "esta a cientos de km, lo que dispara combustible y emisiones.")

cc1, cc2 = st.columns([1, 2])
with cc1:
    conts_ais = st.multiselect("Continentes:", list(CONTINENTES.keys()),
                               default=st.session_state.get("red_conts", ["EU"]),
                               format_func=lambda c: CONTINENTES[c], key="ais_conts")
    tipos_ais = st.multiselect("Tipos:", ["large_airport", "medium_airport", "small_airport"],
                               default=["large_airport", "medium_airport"], key="ais_tipos")
    dist_min = st.slider("Aislados a mas de X km del vecino:", 0, 2000, 100, 25)
with cc2:
    df_ais = df_ap[df_ap["continent"].isin(conts_ais) & df_ap["type"].isin(tipos_ais)] \
        if (conts_ais and tipos_ais) else df_ap.iloc[0:0]
    df_isol = df_ais[df_ais["distancia_vecino_km"] >= dist_min] \
        .sort_values("distancia_vecino_km", ascending=False)
    m1, m2 = st.columns(2)
    m1.metric("Aeropuertos en seleccion", f"{len(df_ais):,}")
    m2.metric(f"Aislados (>{dist_min} km)", f"{len(df_isol):,}")

if not df_ais.empty:
    fig_h = px.histogram(df_ais, x="distancia_vecino_km", nbins=50,
                         color_discrete_sequence=["#1C83E1"],
                         labels={"distancia_vecino_km": "Distancia al vecino (km)"})
    fig_h.add_vline(x=dist_min, line_dash="dash", line_color="red",
                    annotation_text=f"Umbral: {dist_min} km", annotation_position="top right")
    fig_h.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font_color="white", height=240, margin=dict(t=10, b=10))
    st.plotly_chart(fig_h, use_container_width=True)

if not df_isol.empty:
    cm, ct = st.columns([3, 2])
    with cm:
        fig_ma = px.scatter_map(df_ais, lat="latitude_deg", lon="longitude_deg",
                                color="distancia_vecino_km", size="distancia_vecino_km",
                                size_max=18, color_continuous_scale="RdYlGn_r",
                                hover_name="name",
                                hover_data={"ident": True, "municipality": True,
                                            "distancia_vecino_km": ":.1f",
                                            "latitude_deg": False, "longitude_deg": False},
                                map_style="carto-darkmatter", zoom=2)
        fig_ma.update_layout(height=400, margin=dict(r=0, t=0, l=0, b=0),
                             coloraxis_colorbar=dict(title="km", thickness=12))
        st.plotly_chart(fig_ma, use_container_width=True)
    with ct:
        st.dataframe(
            df_isol[["ident", "name", "municipality", "type", "distancia_vecino_km"]].head(20).rename(
                columns={"ident": "ICAO", "name": "Aeropuerto", "municipality": "Ciudad",
                         "type": "Tipo", "distancia_vecino_km": "Dist. vecino (km)"}),
            use_container_width=True, height=380, hide_index=True)