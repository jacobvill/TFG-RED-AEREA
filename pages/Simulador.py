"""
pages/3_Simulador.py
TFG: Simulador de Crisis Aérea con efecto cascada multi-nivel
Jacob Altenburger Villar · UAX 2026
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

# ================================================================
# CAPACIDADES AEROPORTUARIAS (llegadas/hora)
# Fuentes: AENA DGAC Resolución S20 · Eurocontrol STATFOR
# ================================================================
CAPS = {
    # España
    "LEMD":{"arr":48,"night":20,"name":"Madrid-Barajas"},
    "LEBL":{"arr":38,"night":18,"name":"Barcelona El Prat"},
    "LEPA":{"arr":33,"night":33,"name":"Palma de Mallorca"},
    "GCLP":{"arr":24,"night":24,"name":"Gran Canaria"},
    "LEMG":{"arr":25,"night":24,"name":"Málaga"},
    "LEVC":{"arr":20,"night":20,"name":"Valencia"},
    "GCTS":{"arr":21,"night":21,"name":"Tenerife Sur"},
    "LEAL":{"arr":19,"night":19,"name":"Alicante"},
    "LEIB":{"arr":16,"night":16,"name":"Ibiza"},
    "LEBB":{"arr":14,"night":14,"name":"Bilbao"},
    "GCFV":{"arr":14,"night":12,"name":"Fuerteventura"},
    "GCRR":{"arr":12,"night":12,"name":"Lanzarote"},
    "GCXO":{"arr":15,"night":15,"name":"Tenerife Norte"},
    "LEZL":{"arr":15,"night":15,"name":"Sevilla"},
    "LEMH":{"arr":12,"night":12,"name":"Menorca"},
    "LEGE":{"arr":12,"night":12,"name":"Girona"},
    "LERS": {"arr":12,"night":12,"name":"Reus"},
    "LEZG":{"arr":8, "night":8, "name":"Zaragoza"},
    "LEAS":{"arr":8, "night":8, "name":"Asturias"},
    "LEXJ":{"arr":8, "night":8, "name":"Santander"},
    "LEVT":{"arr":10,"night":10,"name":"Vitoria"},
    "LECO":{"arr":6, "night":6, "name":"A Coruña"},
    "LEVX":{"arr":7, "night":7, "name":"Vigo"},
    "LEPP":{"arr":5, "night":5, "name":"Pamplona"},
    "GCLA":{"arr":8, "night":8, "name":"La Palma"},
    "LEAM":{"arr":7, "night":7, "name":"Almería"},
    "LEGR":{"arr":7, "night":7, "name":"Granada"},
    "LEJR":{"arr":8, "night":8, "name":"Jerez"},
    "LEST":{"arr":12,"night":12,"name":"Santiago"},
    # Europa
    "EGLL":{"arr":40,"night":15,"name":"London Heathrow"},
    "EGKK":{"arr":32,"night":15,"name":"London Gatwick"},
    "LFPG":{"arr":53,"night":30,"name":"Paris CDG"},
    "LFPO":{"arr":28,"night":15,"name":"Paris Orly"},
    "EDDF":{"arr":53,"night":25,"name":"Frankfurt"},
    "EDDM":{"arr":56,"night":25,"name":"Munich"},
    "EDDL":{"arr":45,"night":15,"name":"Düsseldorf"},
    "EDDB":{"arr":32,"night":10,"name":"Berlin Brandenburg"},
    "EHAM":{"arr":50,"night":25,"name":"Amsterdam Schiphol"},
    "LIRF":{"arr":42,"night":20,"name":"Roma Fiumicino"},
    "LIMC":{"arr":36,"night":15,"name":"Milán Malpensa"},
    "LSZH":{"arr":36,"night":10,"name":"Zúrich"},
    "LOWW":{"arr":48,"night":20,"name":"Viena"},
    "EKCH":{"arr":38,"night":15,"name":"Copenhague"},
    "ENGM":{"arr":32,"night":10,"name":"Oslo"},
    "ESSA":{"arr":38,"night":15,"name":"Estocolmo"},
    "EFHK":{"arr":28,"night":10,"name":"Helsinki"},
    "LPPT":{"arr":28,"night":15,"name":"Lisboa"},
    "LGAV":{"arr":36,"night":15,"name":"Atenas"},
    "LKPR":{"arr":32,"night":10,"name":"Praga"},
    "EPWA":{"arr":28,"night":10,"name":"Varsovia"},
    "LTFM":{"arr":72,"night":40,"name":"Estambul"},
    "UUEE":{"arr":38,"night":20,"name":"Moscú Sheremetyevo"},
    "EBCI":{"arr":28,"night":10,"name":"Bruselas"},
}
CAP_DEF = {"large_airport":35,"medium_airport":18,"small_airport":6}
CO2_PER_KM  = 16.0
FUEL_PER_KM = 5.0
NIVEL_COLOR = {1:"#00FF88",2:"#FFD700",3:"#FF8C00",4:"#FF4444",5:"#AA00FF"}

@st.cache_data
def cargar_aeropuertos():
    df = pd.read_csv("airports.csv")
    df = df[df["type"].isin(["small_airport","medium_airport","large_airport"])].copy()
    df = df.dropna(subset=["latitude_deg","longitude_deg"])
    df["continent"] = df["continent"].fillna("NA")
    return df

df_ap = cargar_aeropuertos()

def get_trino(usuario):
    if "trino_conn" not in st.session_state or st.session_state.get("trino_user") != usuario:
        st.session_state.trino_conn = connect(
            host="trino.opensky-network.org", port=443,
            user=usuario, auth=OAuth2Authentication(),
            http_scheme="https", catalog="minio", schema="osky", request_timeout=120.0
        )
        st.session_state.trino_user = usuario
    return st.session_state.trino_conn

def get_cap(icao, hora):
    if icao in CAPS:
        return CAPS[icao]["night"] if hora < 5 or hora >= 23 else CAPS[icao]["arr"]
    info = df_ap[df_ap["ident"]==icao]
    return CAP_DEF.get(info["type"].values[0], 15) if not info.empty else 15

def get_nombre(icao):
    if icao in CAPS: return CAPS[icao]["name"]
    info = df_ap[df_ap["ident"]==icao]
    return info["name"].values[0][:35] if not info.empty else icao

def haversine_km(lat1,lon1,lat2,lon2):
    R=6371; phi1,phi2=np.radians(lat1),np.radians(lat2)
    a=np.sin(np.radians(lat2-lat1)/2)**2+np.cos(phi1)*np.cos(phi2)*np.sin(np.radians(lon2-lon1)/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

@st.cache_data(show_spinner=False)
def descargar_flights(_usuario, fecha_str, continente):
    """Descarga flights_data4 con origen Y destino conocidos."""
    BBOXES = {
        "EU":(-25.0,29.0,45.0,81.2),"NA":(-176.6,7.4,-52.0,83.1),
        "SA":(-109.4,-55.0,-32.4,12.4),"AS":(26.0,-12.2,180.0,80.8),
        "AF":(-25.1,-34.8,63.4,37.2),"OC":(110.0,-46.9,180.0,28.2),
    }
    fecha  = datetime.strptime(fecha_str,"%Y-%m-%d")
    ts_day = int(datetime(fecha.year,fecha.month,fecha.day,0,0,0,tzinfo=timezone.utc).timestamp())
    aeros  = df_ap[df_ap["continent"]==continente]["ident"].tolist()
    if not aeros: return pd.DataFrame()
    conn = get_trino(_usuario)
    q = f"""
        SELECT icao24, TRIM(callsign) AS callsign,
               estdepartureairport AS origen, estarrivalairport AS destino,
               firstseen, lastseen
        FROM flights_data4
        WHERE day={ts_day}
          AND estdepartureairport IS NOT NULL AND estarrivalairport IS NOT NULL
          AND estdepartureairport != estarrivalairport AND icao24 IS NOT NULL
    """
    cur=conn.cursor(); cur.execute(q)
    rows=cur.fetchall(); cols=[d[0] for d in cur.description]
    df=pd.DataFrame(rows,columns=cols)
    if not df.empty:
        df=df[df["origen"].isin(aeros)|df["destino"].isin(aeros)]
        df["callsign"]=df["callsign"].fillna("").str.strip()
    return df.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def descargar_posiciones(_usuario, fecha_str, hora, continente):
    """
    Descarga posiciones reales (state_vectors) para la hora seleccionada.
    Permite mostrar los aviones en el mapa con sus posiciones reales.
    """
    BBOXES = {
        "EU":(-25.0,29.0,45.0,81.2),"NA":(-176.6,7.4,-52.0,83.1),
        "SA":(-109.4,-55.0,-32.4,12.4),"AS":(26.0,-12.2,180.0,80.8),
        "AF":(-25.1,-34.8,63.4,37.2),
    }
    fecha  = datetime.strptime(fecha_str,"%Y-%m-%d")
    dt_utc = datetime(fecha.year,fecha.month,fecha.day,hora,0,0,tzinfo=timezone.utc)
    ts     = int(dt_utc.timestamp())
    ts_h   = ts-(ts%3600)
    bbox   = BBOXES.get(continente,(-25.0,29.0,45.0,81.2))
    lnmin,ltmin,lnmax,ltmax = bbox
    conn   = get_trino(_usuario)
    q = f"""
        SELECT icao24,
               MAX_BY(callsign,     time) AS callsign,
               MAX_BY(lat,          time) AS lat,
               MAX_BY(lon,          time) AS lon,
               MAX_BY(velocity,     time) AS velocity,
               MAX_BY(heading,      time) AS heading,
               MAX_BY(baroaltitude, time) AS baroaltitude
        FROM state_vectors_data4
        WHERE hour={ts_h} AND time BETWEEN {ts} AND {ts}+60
          AND onground=false
          AND lat BETWEEN {ltmin-2} AND {ltmax+2}
          AND lon BETWEEN {lnmin-2} AND {lnmax+2}
          AND lat IS NOT NULL AND lon IS NOT NULL
        GROUP BY icao24
    """
    cur=conn.cursor(); cur.execute(q)
    rows=cur.fetchall(); cols=[d[0] for d in cur.description]
    return pd.DataFrame(rows,columns=cols).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def calcular_grafo_y_centralidad(df_vuelos_json):
    """
    Construye el grafo y calcula betweenness centrality.
    Cacheado: solo se recalcula si cambian los datos.
    """
    df_vuelos = pd.read_json(df_vuelos_json)
    aero_dict = df_ap.set_index("ident")[["name","latitude_deg","longitude_deg","type"]].to_dict("index")

    G = nx.DiGraph()
    rutas = df_vuelos.groupby(["origen","destino"]).size().reset_index(name="vuelos")
    for _,r in rutas.iterrows():
        for n in [r["origen"],r["destino"]]:
            if n not in G.nodes and n in aero_dict:
                i=aero_dict[n]
                G.add_node(n,nombre=i["name"],lat=i["latitude_deg"],
                           lon=i["longitude_deg"],tipo=i["type"])
        if r["origen"] in G.nodes and r["destino"] in G.nodes:
            G.add_edge(r["origen"],r["destino"],vuelos=r["vuelos"])

    if len(G.nodes()) < 2:
        return pd.DataFrame(), {}

    k = min(100,max(2,len(G.nodes())-1))
    bet    = nx.betweenness_centrality(G,k=k,weight="vuelos",normalized=True)
    deg_in = dict(G.in_degree(weight="vuelos"))
    deg_out= dict(G.out_degree(weight="vuelos"))

    rows=[]
    for n in G.nodes():
        info = aero_dict.get(n,{})
        rows.append({
            "icao":n,
            "nombre": CAPS.get(n,{}).get("name", info.get("name",n)),
            "lat":    info.get("latitude_deg"),
            "lon":    info.get("longitude_deg"),
            "betweenness": bet.get(n,0),
            "vuelos_in":   deg_in.get(n,0),
            "vuelos_out":  deg_out.get(n,0),
            "vuelos_tot":  deg_in.get(n,0)+deg_out.get(n,0),
        })

    df_met = pd.DataFrame(rows).dropna(subset=["lat","lon"])
    df_met = df_met.sort_values("betweenness",ascending=False).reset_index(drop=True)

    # Serializar aristas para devolver junto a las métricas
    edges = [{"src":s,"dst":d,"vuelos":data["vuelos"]}
             for s,d,data in G.edges(data=True)]

    return df_met, edges

def simular_cascada(df_radar, icao_afectado, capacidad_pct, hora_utc, max_niveles=5):
    n_orig = len(df_radar)
    if n_orig == 0: return [],{},0

    # Inicializar capacidades con 40% libre (aeropuertos no vacíos)
    cap_rest = {}
    for icao in CAPS:
        cap_rest[icao] = int(get_cap(icao,hora_utc)*0.40)
    for _,r in df_ap[df_ap["type"].isin(["large_airport","medium_airport"])].iterrows():
        if r["ident"] not in cap_rest:
            cap_rest[r["ident"]] = int(CAP_DEF.get(r["type"],15)*0.40)

    cap_afect   = int(get_cap(icao_afectado,hora_utc)*capacidad_pct/100)
    n_overflow  = max(0, n_orig - cap_afect)
    cap_rest[icao_afectado] = 0
    saturados   = {icao_afectado}
    resultados  = []
    sin_alt     = 0
    cola        = [(icao_afectado, n_overflow, 1)]

    info_pos = df_ap.set_index("ident")[["latitude_deg","longitude_deg"]].to_dict("index")

    while cola:
        src, pendientes, nivel = cola.pop(0)
        if pendientes <= 0: continue
        if nivel > max_niveles:
            sin_alt += pendientes
            resultados.append({"nivel":nivel,"desde":src,"hasta":"Sin alternativa",
                               "n_vuelos":pendientes,"dist_km":0,
                               "lat_dest":None,"lon_dest":None,"nombre_dest":"Sin alternativa"})
            continue

        # Posición del aeropuerto fuente
        if src in info_pos:
            lat_s = info_pos[src]["latitude_deg"]
            lon_s = info_pos[src]["longitude_deg"]
        elif src in CAPS:
            lat_s = lon_s = None
        else:
            sin_alt += pendientes; continue

        # Candidatos: large/medium con capacidad, no saturados, ordenados por distancia
        candidatos = df_ap[
            df_ap["type"].isin(["large_airport","medium_airport"]) &
            (~df_ap["ident"].isin(saturados)) &
            (df_ap["ident"]!=src)
        ].copy()

        if lat_s is not None:
            candidatos["dist"] = candidatos.apply(
                lambda r: haversine_km(lat_s,lon_s,r["latitude_deg"],r["longitude_deg"]),axis=1)
            candidatos = candidatos.sort_values("dist").head(15)
        else:
            candidatos = candidatos.head(15)

        encontrado = False
        for _,alt in candidatos.iterrows():
            if pendientes <= 0: break
            icao_alt = alt["ident"]
            cap_disp = cap_rest.get(icao_alt, int(get_cap(icao_alt,hora_utc)*0.40))
            if cap_disp <= 0: continue

            n_asig = min(pendientes, cap_disp)
            pendientes -= n_asig
            cap_rest[icao_alt] = cap_disp - n_asig
            encontrado = True

            lat_d = alt["latitude_deg"]; lon_d = alt["longitude_deg"]
            dist  = haversine_km(lat_s,lon_s,lat_d,lon_d) if lat_s else 0

            resultados.append({
                "nivel":nivel,"desde":src,"hasta":icao_alt,
                "nombre_dest":get_nombre(icao_alt),
                "n_vuelos":n_asig,"dist_km":round(dist,1),
                "lat_dest":lat_d,"lon_dest":lon_d,
            })

            if cap_rest[icao_alt] <= 0:
                overflow_nuevo = abs(cap_rest[icao_alt])
                cap_rest[icao_alt] = 0
                saturados.add(icao_alt)
                if overflow_nuevo > 0:
                    cola.append((icao_alt, overflow_nuevo, nivel+1))

        if pendientes > 0:
            cola.append((src, pendientes, nivel+1))

    return resultados, cap_rest, sin_alt

# ================================================================
# SIDEBAR
# ================================================================
st.title("🔬 Simulador de Crisis Aérea")
st.caption("Efecto cascada multi-nivel · Capacidades AENA/Eurocontrol · Aviones reales en mapa")

datos_ext = st.session_state.get("datos_sim")
st.sidebar.header("📥 Fuente de datos")
if datos_ext:
    st.sidebar.success(f"📨 {datos_ext['label']}")
    usar_ext = st.sidebar.toggle("Usar datos recibidos", value=True)
else:
    usar_ext = False
    st.sidebar.info("Sin datos de otra página.")

st.sidebar.divider()
st.sidebar.header("📅 Cargar desde Trino")
user_sim  = st.sidebar.text_input("Usuario Trino", value="jaltevil@myuax.com").lower()
fecha_sim = st.sidebar.date_input("Día", datetime(2024,1,16))
hora_sim  = st.sidebar.selectbox("Hora (UTC)", list(range(24)), index=12)
CONTS     = {"EU":"Europa","NA":"Norteamérica","SA":"Sudamérica","AS":"Asia","AF":"África"}
cont_sim  = st.sidebar.selectbox("Continente", list(CONTS.keys()), format_func=lambda x:CONTS[x])
btn_trino = st.sidebar.button("📊 Cargar desde Trino", use_container_width=True, type="primary")
st.sidebar.caption("⚠️ 30-60 s · Datos cacheados.")

# ── CARGA ────────────────────────────────────────────────────────
if btn_trino:
    with st.spinner("⏳ Descargando rutas (flights_data4) y posiciones (state_vectors)..."):
        try:
            df_fl = descargar_flights(user_sim, str(fecha_sim), cont_sim)
            df_pos= descargar_posiciones(user_sim, str(fecha_sim), hora_sim, cont_sim)
            if df_fl.empty:
                st.sidebar.error("Sin vuelos con ruta completa para ese día.")
            else:
                st.session_state.update({
                    "sim_flights": df_fl,
                    "sim_pos":     df_pos,
                    "sim_label":   f"Trino · {fecha_sim} · {CONTS[cont_sim]}",
                    "sim_hora":    hora_sim,
                    "sim_fecha":   str(fecha_sim),
                    "sim_cont":    cont_sim,
                    "sim_met":     None,   # forzar recálculo
                    "sim_edges":   None,
                })
                st.sidebar.success(f"✅ {len(df_fl):,} vuelos · {len(df_pos):,} posiciones")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            if "trino_conn" in st.session_state: del st.session_state["trino_conn"]

if usar_ext and datos_ext:
    df_ext = datos_ext["df"]
    if "origen" in df_ext.columns and "destino" in df_ext.columns:
        df_ext = df_ext.dropna(subset=["origen","destino"])
        if not df_ext.empty:
            st.session_state.update({
                "sim_flights": df_ext,
                "sim_pos":     pd.DataFrame(),
                "sim_label":   datos_ext["label"],
                "sim_hora":    hora_sim,
                "sim_fecha":   datos_ext.get("fecha", str(fecha_sim)),
                "sim_cont":    datos_ext.get("continente","EU"),
                "sim_met":     None,
                "sim_edges":   None,
            })

df_flights = st.session_state.get("sim_flights", pd.DataFrame())
df_pos     = st.session_state.get("sim_pos",     pd.DataFrame())
label_sim  = st.session_state.get("sim_label","")
hora_carg  = st.session_state.get("sim_hora", hora_sim)
fecha_carg = st.session_state.get("sim_fecha", str(fecha_sim))
cont_carg  = st.session_state.get("sim_cont", cont_sim)

# ── CALCULAR GRAFO (solo si no está cacheado) ────────────────────
if not df_flights.empty and st.session_state.get("sim_met") is None:
    with st.spinner("⚙️ Calculando centralidad del grafo (solo la primera vez)..."):
        df_met, edges = calcular_grafo_y_centralidad(df_flights.to_json())
        st.session_state["sim_met"]   = df_met
        st.session_state["sim_edges"] = edges

df_met  = st.session_state.get("sim_met",  pd.DataFrame())
edges   = st.session_state.get("sim_edges", [])

# ================================================================
# INTERFAZ
# ================================================================
if not df_flights.empty:
    st.success(f"📊 **{label_sim}** · {len(df_flights):,} vuelos · "
               f"{len(df_met):,} aeropuertos en red")

    tab1, tab2, tab3 = st.tabs(["💥 Simulador de Crisis", "📈 Red & Centralidad", "🗺️ Mapa de la Red"])

    # ── TAB 1: SIMULADOR ─────────────────────────────────────────
    with tab1:
        col_cfg, col_mapa = st.columns([1,2])

        with col_cfg:
            st.markdown("### ⚙️ Escenario")
            dest_counts = df_flights["destino"].value_counts()
            dest_counts = dest_counts[dest_counts>=3]
            opc = [f"{i} — {get_nombre(i)}" for i in dest_counts.index]
            sel = st.selectbox("🏢 Aeropuerto afectado:", opc)
            icao_af = sel.split(" — ")[0]

            cap_pct = st.slider("Capacidad operativa restante (%):", 0, 90, 0, 10,
                                help="0% = cierre total")
            tipo_inc= st.selectbox("Tipo de incidencia:",[
                "🌋 Cierre de espacio aéreo",
                "🌧️ Meteorología extrema",
                "🔧 Mantenimiento emergencia",
                "⚔️ Conflicto geopolítico",
                "💻 Fallo de sistemas",
            ])
            ventana_h = st.slider("Ventana de radar (horas):", 1, 3, 1,
                help="Vuelos que aterrizarán en las próximas N horas")
            radar_km  = st.slider("Radio visual radar (km):", 100, 800, 400, 50)
            btn_sim   = st.button("▶️ Ejecutar simulación",
                                  use_container_width=True, type="primary")

            # Calcular vuelos en radar
            fecha_dt = datetime.strptime(fecha_carg, "%Y-%m-%d")
            ts_hora  = int(datetime(fecha_dt.year,fecha_dt.month,fecha_dt.day,
                                    hora_carg,0,0,tzinfo=timezone.utc).timestamp())
            ts_fin   = ts_hora + ventana_h*3600

            df_radar = df_flights[
                (df_flights["destino"]==icao_af) &
                (df_flights["lastseen"]>=ts_hora) &
                (df_flights["lastseen"]<=ts_fin)
            ].copy() if "lastseen" in df_flights.columns else df_flights[df_flights["destino"]==icao_af].copy()

            cap_hora = get_cap(icao_af, hora_carg)
            cap_ef   = int(cap_hora * cap_pct/100)

            st.markdown("---")
            m1,m2 = st.columns(2)
            m1.metric("Capacidad máx.", f"{cap_hora} arr/h")
            m2.metric("Capacidad efectiva", f"{cap_ef} arr/h", delta=f"{cap_pct-100}%")
            m3,m4 = st.columns(2)
            m3.metric("Vuelos en radar", f"{len(df_radar):,}")
            m4.metric("Overflow estimado",
                      f"{max(0,len(df_radar)-cap_ef*ventana_h):,}",
                      delta_color="inverse")

        with col_mapa:
            st.markdown("### 🗺️ Situación actual")

            info_c = df_ap[df_ap["ident"]==icao_af]
            fig_base = go.Figure()

            if not info_c.empty:
                lat_c = info_c["latitude_deg"].values[0]
                lon_c = info_c["longitude_deg"].values[0]

                # Círculo radar
                theta = np.linspace(0,2*np.pi,72)
                lat_circ = lat_c + (radar_km/111.32)*np.cos(theta)
                lon_circ = lon_c + (radar_km/(111.32*np.cos(np.radians(lat_c))))*np.sin(theta)
                fig_base.add_trace(go.Scattermap(
                    lat=np.append(lat_circ,lat_circ[0]),
                    lon=np.append(lon_circ,lon_circ[0]),
                    mode="lines",name=f"Radar {radar_km}km",showlegend=True,
                    line=dict(color="rgba(255,100,100,0.5)",width=2),hoverinfo="none"
                ))

                # Aeropuerto afectado
                fig_base.add_trace(go.Scattermap(
                    lat=[lat_c],lon=[lon_c],mode="markers+text",
                    name=f"🏢 {icao_af}",text=[icao_af],textposition="top right",
                    marker=go.scattermap.Marker(size=24,color="red"),
                    hovertext=[f"🏢 {get_nombre(icao_af)}<br>Cap: {cap_pct}%<br>Vuelos radar: {len(df_radar)}"],
                    hoverinfo="text"
                ))

                # Aviones reales en posición (si están disponibles)
                if not df_pos.empty and not df_radar.empty:
                    icaos_radar = set(df_radar["icao24"])
                    df_av_vis = df_pos[df_pos["icao24"].isin(icaos_radar)].copy()
                    if not df_av_vis.empty:
                        df_av_vis["callsign"] = df_av_vis["callsign"].fillna("").str.strip()
                        vel_k = (pd.to_numeric(df_av_vis["velocity"],errors="coerce").fillna(0)*3.6).round(0).astype(int).astype(str)
                        alt_k = (pd.to_numeric(df_av_vis["baroaltitude"],errors="coerce").fillna(0)*3.281).round(0).astype(int).astype(str)
                        # Línea recta al destino
                        for _,av in df_av_vis.iterrows():
                            fig_base.add_trace(go.Scattermap(
                                lat=[av["lat"],lat_c,None],
                                lon=[av["lon"],lon_c,None],
                                mode="lines",showlegend=False,
                                line=dict(width=1,color="rgba(255,255,0,0.3)"),
                                hoverinfo="none"
                            ))
                        hover_av=(
                            "✈️ <b>"+df_av_vis["callsign"]+"</b><br>"+
                            "ICAO24: "+df_av_vis["icao24"]+"<br>"+
                            "Vel: "+vel_k+" km/h · Alt: "+alt_k+" ft"
                        )
                        fig_base.add_trace(go.Scattermap(
                            lat=df_av_vis["lat"],lon=df_av_vis["lon"],
                            mode="markers",name=f"✈️ En ruta a {icao_af} ({len(df_av_vis)})",
                            marker=go.scattermap.Marker(size=10,color="yellow"),
                            text=hover_av,hoverinfo="text"
                        ))

                fig_base.update_layout(
                    map_style="carto-darkmatter",
                    margin={"r":0,"t":0,"l":0,"b":0},height=520,
                    showlegend=True,
                    legend=dict(yanchor="top",y=0.98,xanchor="left",x=0.02,
                                bgcolor="rgba(0,0,0,0.7)",font=dict(color="white")),
                    map=dict(center=dict(lat=lat_c,lon=lon_c),zoom=4)
                )
            st.plotly_chart(fig_base, use_container_width=True, config={"scrollZoom":True})

        # ── RESULTADOS ───────────────────────────────────────────
        if btn_sim:
            if df_radar.empty:
                st.warning(f"No hay vuelos con destino {icao_af} en la ventana temporal.")
            else:
                with st.spinner("⚙️ Ejecutando cascada multi-nivel..."):
                    res, cap_fin, sin_alt = simular_cascada(
                        df_radar, icao_af, cap_pct, hora_carg
                    )
                    st.session_state.update({
                        "sim_res":res,"sim_cap_fin":cap_fin,
                        "sim_sin_alt":sin_alt,"sim_icao_res":icao_af,
                        "sim_cap_pct_res":cap_pct,"sim_tipo_res":tipo_inc,
                        "sim_n_radar_res":len(df_radar),
                    })

        res       = st.session_state.get("sim_res")
        icao_res  = st.session_state.get("sim_icao_res","")
        cap_res   = st.session_state.get("sim_cap_pct_res",0)
        tipo_res  = st.session_state.get("sim_tipo_res","")
        n_rad_res = st.session_state.get("sim_n_radar_res",0)
        sin_alt   = st.session_state.get("sim_sin_alt",0)

        if res is not None and icao_res == icao_af:
            st.divider()
            df_res    = pd.DataFrame(res)
            df_res_ok = df_res[df_res["hasta"]!="Sin alternativa"]
            km_tot    = (df_res_ok["n_vuelos"]*df_res_ok["dist_km"]).sum()
            co2_t     = round(km_tot*CO2_PER_KM/1000,1)
            fuel_t    = round(km_tot*FUEL_PER_KM/1000,1)
            n_niv     = int(df_res["nivel"].max()) if not df_res.empty else 0
            n_red     = int(df_res_ok["n_vuelos"].sum())
            n_alt     = df_res_ok["hasta"].nunique()

            st.markdown(f"### 📊 Resultados: {tipo_res}")
            k1,k2,k3,k4,k5 = st.columns(5)
            k1.metric("✈️ Redirigidos",    f"{n_red:,}")
            k2.metric("🔁 Niveles cascada",f"{n_niv}")
            k3.metric("🏢 Aeropuertos alt.",f"{n_alt}")
            k4.metric("🌡️ CO₂ extra",      f"{co2_t:,} t")
            k5.metric("❌ Sin alternativa", f"{sin_alt:,}",delta_color="inverse")

            # Mapa de resultados
            info_c = df_ap[df_ap["ident"]==icao_af]
            if not info_c.empty:
                lat_c = info_c["latitude_deg"].values[0]
                lon_c = info_c["longitude_deg"].values[0]
                fig_r = go.Figure()

                # Aeropuerto cerrado
                fig_r.add_trace(go.Scattermap(
                    lat=[lat_c],lon=[lon_c],mode="markers+text",
                    name=f"❌ {icao_af}",text=[icao_af],textposition="top right",
                    marker=go.scattermap.Marker(size=26,color="red"),
                    hovertext=[f"❌ {get_nombre(icao_af)}<br>Cap: {cap_res}%"],
                    hoverinfo="text"
                ))

                # Aviones en mapa de resultados (redirigidos)
                if not df_pos.empty and not df_radar.empty:
                    icaos_rd = set(df_radar["icao24"])
                    df_redir_pos = df_pos[df_pos["icao24"].isin(icaos_rd)].copy()
                    if not df_redir_pos.empty:
                        fig_r.add_trace(go.Scattermap(
                            lat=df_redir_pos["lat"],lon=df_redir_pos["lon"],
                            mode="markers",name="✈️ Aviones afectados",
                            marker=go.scattermap.Marker(size=9,color="orange"),
                            hovertext=df_redir_pos["callsign"].fillna("").str.strip(),
                            hoverinfo="text"
                        ))

                # Alternativas por nivel
                mostrados = set()
                for nivel in sorted(df_res_ok["nivel"].unique()):
                    color = NIVEL_COLOR.get(nivel,"#AAAAAA")
                    df_n  = df_res_ok[(df_res_ok["nivel"]==nivel) & (df_res_ok["lat_dest"].notna())]
                    for _,row in df_n.iterrows():
                        lat_s_row = lat_c if nivel==1 else None
                        lon_s_row = lon_c if nivel==1 else None
                        if nivel>1:
                            prev = df_res_ok[(df_res_ok["hasta"]==row["desde"])&(df_res_ok["nivel"]==nivel-1)]
                            if not prev.empty and prev.iloc[0]["lat_dest"] is not None:
                                lat_s_row=prev.iloc[0]["lat_dest"]; lon_s_row=prev.iloc[0]["lon_dest"]
                            else:
                                src_i=df_ap[df_ap["ident"]==row["desde"]]
                                if not src_i.empty:
                                    lat_s_row=src_i["latitude_deg"].values[0]
                                    lon_s_row=src_i["longitude_deg"].values[0]
                        if lat_s_row:
                            grosor = max(1,int(row["n_vuelos"]//5))
                            fig_r.add_trace(go.Scattermap(
                                lat=[lat_s_row,row["lat_dest"],None],
                                lon=[lon_s_row,row["lon_dest"],None],
                                mode="lines",showlegend=False,
                                line=dict(width=grosor,color=color),hoverinfo="none"
                            ))
                        if row["hasta"] not in mostrados:
                            mostrados.add(row["hasta"])
                            n_vuel = int(df_n[df_n["hasta"]==row["hasta"]]["n_vuelos"].sum())
                            fig_r.add_trace(go.Scattermap(
                                lat=[row["lat_dest"]],lon=[row["lon_dest"]],
                                mode="markers+text",
                                name=f"Nivel {nivel}",
                                text=[row["hasta"]],textposition="top right",
                                marker=go.scattermap.Marker(size=18,color=color),
                                hovertext=[f"✅ {row['nombre_dest']}<br>Nivel: {nivel}<br>"
                                           f"Vuelos: {n_vuel}<br>Dist: {row['dist_km']} km"],
                                hoverinfo="text",showlegend=(row["hasta"]==df_n["hasta"].iloc[0])
                            ))

                fig_r.update_layout(
                    map_style="carto-darkmatter",
                    margin={"r":0,"t":0,"l":0,"b":0},height=500,showlegend=True,
                    legend=dict(yanchor="top",y=0.98,xanchor="left",x=0.02,
                                bgcolor="rgba(0,0,0,0.7)",font=dict(color="white")),
                    map=dict(center=dict(lat=lat_c,lon=lon_c),zoom=4)
                )
                st.plotly_chart(fig_r, use_container_width=True, config={"scrollZoom":True})

            # Tabla detalle
            st.markdown("#### 📋 Detalle por nivel")
            df_res["CO₂ (t)"] = (df_res["n_vuelos"]*df_res["dist_km"]*CO2_PER_KM/1000).round(1)
            st.dataframe(
                df_res[["nivel","desde","hasta","nombre_dest","n_vuelos","dist_km","CO₂ (t)"]].rename(columns={
                    "nivel":"Nivel","desde":"Desde","hasta":"Hacia",
                    "nombre_dest":"Aeropuerto alt.","n_vuelos":"Vuelos","dist_km":"Dist (km)"
                }),use_container_width=True
            )

            # Impacto ambiental
            st.divider()
            st.markdown("#### 🌱 Impacto Medioambiental")
            arboles=int(co2_t*1000/21.77); km_coche=int(co2_t*1000/0.21); coste=int(fuel_t*800)
            e1,e2,e3,e4=st.columns(4)
            e1.metric("🌳 Árboles/año",    f"{arboles:,}")
            e2.metric("🚗 Km de coche eq.",f"{km_coche:,}")
            e3.metric("💵 Coste combustible",f"${coste:,}")
            e4.metric("⛽ Fuel extra",      f"{fuel_t:,} t")
            st.info(
                f"**{tipo_res}** en **{get_nombre(icao_af)}** al **{cap_res}%** → "
                f"**{n_red:,} vuelos redirigidos** en **{n_niv} niveles** → "
                f"**{co2_t:,} t CO₂ extra** ≈ **{arboles:,} árboles** absorbiendo CO₂ durante 1 año."
            )
            if sin_alt>0:
                st.error(f"⚠️ {sin_alt} vuelos sin alternativa disponible tras {5} niveles de cascada.")

    # ── TAB 2: CENTRALIDAD ───────────────────────────────────────
    with tab2:
        if df_met.empty:
            st.info("Cargando datos...")
        else:
            st.subheader("🏆 Aeropuertos más críticos")
            st.caption("Betweenness: fracción de rutas óptimas que pasan por este aeropuerto. "
                       "Alto betweenness = nodo puente crítico.")
            top20 = df_met.head(20).copy()
            top20["bet_%"] = (top20["betweenness"]*100).round(2)
            cg,ct = st.columns([2,1])
            with cg:
                fig_b=px.bar(top20,x="bet_%",y="icao",orientation="h",hover_name="nombre",
                             color="bet_%",color_continuous_scale="Reds",
                             labels={"bet_%":"Betweenness (%)","icao":"Aeropuerto"},
                             title="Top 20 — Betweenness Centrality")
                fig_b.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                                    font_color="white",height=480,
                                    yaxis=dict(autorange="reversed"),coloraxis_showscale=False)
                st.plotly_chart(fig_b, use_container_width=True)
            with ct:
                st.dataframe(top20[["icao","nombre","vuelos_tot","bet_%"]].rename(columns={
                    "icao":"ICAO","nombre":"Aeropuerto","vuelos_tot":"Vuelos","bet_%":"Betweenness (%)"}),
                    use_container_width=True,height=450)
            st.divider()
            cd,cb2=st.columns(2)
            with cd:
                fig_d=px.histogram(df_met,x="vuelos_tot",nbins=60,
                                   title="Distribución de conectividad (Degree)",
                                   labels={"vuelos_tot":"Total vuelos"},color_discrete_sequence=["#1C83E1"])
                fig_d.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                                    font_color="white",height=270)
                st.plotly_chart(fig_d, use_container_width=True)
            with cb2:
                fig_b2=px.histogram(df_met,x="betweenness",nbins=60,
                                    title="Distribución de Betweenness",
                                    color_discrete_sequence=["#FF4B4B"])
                fig_b2.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                                     font_color="white",height=270)
                st.plotly_chart(fig_b2, use_container_width=True)

    # ── TAB 3: MAPA DE LA RED ────────────────────────────────────
    with tab3:
        if df_met.empty or not edges:
            st.info("Cargando grafo...")
        else:
            st.subheader("🗺️ Mapa de la red aérea")
            top_n = st.slider("Top N aeropuertos:", 10, 120, 40, 10)
            top_a = df_met.head(top_n)
            icaos_top = set(top_a["icao"])

            # Aristas en UNA SOLA TRAZA (mucho más rápido que una por arista)
            lats_e, lons_e = [], []
            for e in edges:
                if e["src"] in icaos_top and e["dst"] in icaos_top:
                    si = df_ap[df_ap["ident"]==e["src"]]
                    di = df_ap[df_ap["ident"]==e["dst"]]
                    if si.empty or di.empty: continue
                    lats_e += [si["latitude_deg"].values[0],di["latitude_deg"].values[0],None]
                    lons_e += [si["longitude_deg"].values[0],di["longitude_deg"].values[0],None]

            fig_g = go.Figure()
            if lats_e:
                fig_g.add_trace(go.Scattermap(
                    lat=lats_e, lon=lons_e, mode="lines",
                    line=dict(width=0.5,color="rgba(100,150,255,0.2)"),
                    hoverinfo="none", showlegend=False, name="Rutas"
                ))

            max_b = top_a["betweenness"].max() or 1
            sizes = (10+28*top_a["betweenness"]/max_b).round(0)
            fig_g.add_trace(go.Scattermap(
                lat=top_a["lat"],lon=top_a["lon"],
                mode="markers+text",text=top_a["icao"],
                textposition="top center",textfont=dict(size=9,color="white"),
                marker=go.scattermap.Marker(
                    size=sizes,color=top_a["betweenness"],
                    colorscale="Reds",showscale=True,
                    colorbar=dict(title="Betweenness",thickness=12)
                ),
                hovertext=(
                    "<b>"+top_a["nombre"]+"</b><br>"+
                    "ICAO: "+top_a["icao"]+"<br>"+
                    "Vuelos: "+top_a["vuelos_tot"].astype(str)+"<br>"+
                    "Betweenness: "+(top_a["betweenness"]*100).round(2).astype(str)+"%"
                ),
                hoverinfo="text",name="Aeropuertos"
            ))
            fig_g.update_layout(
                map_style="carto-darkmatter",
                margin={"r":0,"t":0,"l":0,"b":0},height=650,showlegend=False,
                map=dict(center=dict(lat=top_a["lat"].mean(),lon=top_a["lon"].mean()),zoom=3)
            )
            st.plotly_chart(fig_g, use_container_width=True, config={"scrollZoom":True})

else:
    st.info("**Para empezar:** carga datos desde Trino o envía datos desde otra página.")
    with st.expander("📖 Metodología"):
        st.markdown("""
        **Radar temporal**: filtra vuelos cuyo `lastseen` cae en la ventana de N horas desde la hora seleccionada.
        Equivale al tráfico que aterrizará en ese período.

        **Cascada multi-nivel**: cuando el aeropuerto afectado no puede absorber todos los vuelos,
        el overflow se redistribuye entre los aeropuertos más cercanos con capacidad disponible.
        Si alguno de ellos también se satura, su overflow genera un nuevo nivel de cascada.

        **Capacidades**: AENA DGAC Resolución S20 para aeropuertos españoles.
        Eurocontrol STATFOR para aeropuertos europeos. Franja nocturna (00:00-04:59) con reducción.

        **CO₂**: modelo ICAO · ~5 kg combustible/km (A320) · 3.16 kg CO₂/kg queroseno.
        """)