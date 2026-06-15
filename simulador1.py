"""
pages/5_Simulador.py
TFG: Simulacion y Analisis del Impacto Operativo de la Red Aerea Global
Jacob Altenburger Villar - UAX 2026

Simulador (Modelo B - prioridad al vuelo desviado) con dimension temporal:
- Red: aeropuertos medianos y grandes de TODA EUROPA. Capacidades reales declaradas
  por AENA para los espanoles; estimadas por tipo para el resto (editables).
- Incidencia: reduccion de capacidad de un aeropuerto durante H horas (100% = cierre total).
- Cada hora entra una nueva tanda de llegadas que hay que desviar. Las plazas que ocupan
  los desviados NO se liberan durante el cierre, asi que la red se satura hora a hora y la
  cascada se extiende a aeropuertos cada vez mas lejanos.
- Cascada (Modelo B): los desviados tienen prioridad; cuando un aeropuerto se llena desplaza
  su propio trafico, que pasa a ser el problema del nivel siguiente. Hasta 5 niveles.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import deque
from math import radians, sin, cos, asin, sqrt, pi
from datetime import datetime, timezone
from trino.dbapi import connect
from trino.auth import OAuth2Authentication

st.set_page_config(page_title="TFG - Simulador", page_icon="🛬", layout="wide")

NIVEL_COLOR = {1: "#00CC66", 2: "#FFD400", 3: "#FF8C00", 4: "#FF3B3B", 5: "#A020F0"}

# Estimacion de capacidad (llegadas/h) para aeropuertos FUERA del Excel europeo
# (resto del mundo). Media europea por tipo: ~27 grandes, ~13 medianos.
CAP_EST = {"large_airport": 27, "medium_airport": 13}

# Excel con la capacidad de los aeropuertos europeos.
# Debe estar en la misma carpeta que airports.csv.
EXCEL_CAP = "capacidad_aterrizaje_aeropuertos_europa.xlsx"


@st.cache_data(show_spinner=False)
def cargar_cap_excel():
    cap = pd.read_excel(EXCEL_CAP)
    return dict(zip(cap["ICAO"].astype(str), cap["Llegadas/h (modelo)"]))


@st.cache_data(show_spinner=False)
def cargar_eu():
    cap_eu = cargar_cap_excel()
    df = pd.read_csv("airports.csv")
    df = df[(df["continent"] == "EU") &
            (df["type"].isin(["large_airport", "medium_airport"]))]
    df = df.dropna(subset=["latitude_deg", "longitude_deg"]).copy()
    df = df[df["scheduled_service"] == "yes"]
    df["cap_real"] = df["ident"].isin(cap_eu)        # True = capacidad del Excel europeo
    df["cap_h"] = df["ident"].map(cap_eu)            # Europa: valor del Excel
    df["cap_h"] = df["cap_h"].fillna(                # resto del mundo: estimacion por tipo
        df["type"].map(CAP_EST)).astype(int)
    return df.reset_index(drop=True)


# ================================================================
# DATOS REALES (Trino): llegadas reales por hora a un aeropuerto
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
def llegadas_reales_por_hora(fecha, icao, usuario):
    """
    Llegadas reales a 'icao' ese dia, repartidas por hora UTC (lista de 24 enteros).
    Se usa 'lastseen' como momento aproximado de aterrizaje (ultimo contacto del vuelo),
    en linea con la ventana de deteccion temporal del modelo.
    """
    ts_day = int(datetime(fecha.year, fecha.month, fecha.day, 0, 0, 0,
                          tzinfo=timezone.utc).timestamp())
    conn = get_trino(usuario)
    q = f"""
        SELECT lastseen
        FROM flights_data4
        WHERE day={ts_day} AND estarrivalairport='{icao}' AND lastseen IS NOT NULL
    """
    cur = conn.cursor()
    cur.execute(q)
    rows = cur.fetchall()
    horas = [0] * 24
    for (ls,) in rows:
        h = int(((int(ls) - ts_day) // 3600) % 24)
        horas[h] += 1
    return horas


def hav(la1, lo1, la2, lo2):
    R = 6371.0
    p1, p2 = radians(la1), radians(la2)
    a = (sin(radians(la2 - la1) / 2) ** 2 +
         cos(p1) * cos(p2) * sin(radians(lo2 - lo1) / 2) ** 2)
    return 2 * R * asin(sqrt(max(0, a)))


def circulo(lat, lon, r_km=500, n=72):
    lats, lons = [], []
    for k in range(n + 1):
        ang = 2 * pi * k / n
        lats.append(lat + (r_km / 111.0) * cos(ang))
        lons.append(lon + (r_km / (111.0 * cos(radians(lat)))) * sin(ang))
    return lats, lons


# ================================================================
# MOTOR DE CASCADA - MODELO B CON DIMENSION TEMPORAL
# ================================================================
def simular_b(caps_df, icao_A, reduccion, N_in, horas=1, occ=0.60, radio=500, max_nivel=5,
              heavy_pct=0.0, seed=1):
    d = caps_df.set_index("ident")
    co = {i: (float(d.loc[i, "latitude_deg"]), float(d.loc[i, "longitude_deg"])) for i in d.index}
    cap = {i: int(d.loc[i, "cap_h"]) for i in d.index}
    tipo_ap = {i: str(d.loc[i, "type"]) for i in d.index}   # large/medium para compatibilidad
    # estado de capacidad PERSISTENTE durante todo el cierre (no se libera entre horas)
    free = {i: int(round(cap[i] * (1 - occ))) for i in d.index}
    sched = {i: cap[i] - free[i] for i in d.index}
    free[icao_A] = 0
    sched[icao_A] = 0

    red_cap = int(cap[icao_A] * (1 - reduccion / 100))

    # N_in puede ser un entero (mismas llegadas cada hora, modo parametrico) o una lista
    # con las llegadas reales de cada hora del cierre (modo datos reales).
    if isinstance(N_in, (list, tuple, np.ndarray)):
        llegadas_h = [int(x) for x in N_in][:horas]
        llegadas_h += [0] * (horas - len(llegadas_h))
    else:
        llegadas_h = [int(N_in)] * horas
    overflow_por_hora = [max(0, n - red_cap) for n in llegadas_h]   # a desviar en cada hora

    dist_cache = {}

    def cercanos(src):
        if src in dist_cache:
            return dist_cache[src]
        la, lo = co[src]
        out = [(hav(la, lo, co[j][0], co[j][1]), j) for j in d.index if j != src]
        out = sorted((dd, j) for dd, j in out if dd <= radio)
        dist_cache[src] = out
        return out

    vuelos, no_reub = [], []
    circulos = {icao_A: (1, 1)}                 # icao -> (nivel emitido, hora en que aparece)
    nid = 0

    def nuevos_vuelos(n, prefijo, src_tipo):
        # Crea n vuelos como (id, es_pesado). Solo los aeropuertos grandes generan
        # trafico pesado (cat. 6); en los medianos no operan, asi que ahi heavy = 0.
        nonlocal nid
        nh = int(round(n * heavy_pct)) if src_tipo == "large_airport" else 0
        out = [(f"{prefijo}{nid + k:04d}", k < nh) for k in range(n)]
        nid += n
        return out

    for hora in range(1, horas + 1):
        overflow_h = overflow_por_hora[hora - 1]
        if overflow_h <= 0:
            continue
        q = deque([(icao_A, nuevos_vuelos(overflow_h, f"{icao_A}-", tipo_ap.get(icao_A)),
                    "desviado", 1)])
        while q:
            src, lote, tipo, niv = q.popleft()
            if not lote:
                continue
            if niv > max_nivel:
                no_reub.extend({"vuelo": f, "origen": src, "tipo": tipo, "hora": hora,
                                "heavy": h} for f, h in lote)
                continue
            cands = cercanos(src)
            if not cands:
                no_reub.extend({"vuelo": f, "origen": src, "tipo": tipo, "hora": hora,
                                "heavy": h} for f, h in lote)
                continue
            pend_h = [f for f, h in lote if h]      # pesados: solo a aeropuertos grandes
            pend_n = [f for f, h in lote if not h]  # resto: a grandes o medianos

            bumped = []
            # Pass 1 = hueco libre (free); Pass 2 = desplazar trafico propio (sched)
            for capdict, es_bump in ((free, False), (sched, True)):
                for dd, j in cands:
                    if not pend_h and not pend_n:
                        break
                    libre = capdict[j]
                    if libre <= 0:
                        continue
                    j_grande = tipo_ap.get(j) == "large_airport"
                    dist_j = round(hav(*co[src], *co[j]), 1)
                    usados = 0
                    if j_grande:                    # los pesados primero, solo aqui caben
                        while libre > 0 and pend_h:
                            f = pend_h.pop(0)
                            vuelos.append({"vuelo": f, "origen": src, "destino": j, "nivel": niv,
                                           "dist": dist_j, "tipo": tipo, "bump": es_bump,
                                           "hora": hora, "heavy": True})
                            libre -= 1
                            usados += 1
                    while libre > 0 and pend_n:
                        f = pend_n.pop(0)
                        vuelos.append({"vuelo": f, "origen": src, "destino": j, "nivel": niv,
                                       "dist": dist_j, "tipo": tipo, "bump": es_bump,
                                       "hora": hora, "heavy": False})
                        libre -= 1
                        usados += 1
                    if usados > 0:
                        capdict[j] -= usados
                        if es_bump:
                            bumped.append((j, usados))

            for f in pend_h:
                no_reub.append({"vuelo": f, "origen": src, "tipo": tipo, "hora": hora, "heavy": True})
            for f in pend_n:
                no_reub.append({"vuelo": f, "origen": src, "tipo": tipo, "hora": hora, "heavy": False})
            for j, c in bumped:
                circulos.setdefault(j, (niv + 1, hora))
                q.append((j, nuevos_vuelos(c, f"{j}-", tipo_ap.get(j)), "desplazado", niv + 1))

    df_v = pd.DataFrame(vuelos)
    df_n = pd.DataFrame(no_reub)
    return {
        "overflow_por_hora": overflow_por_hora, "llegadas_h": llegadas_h,
        "red_cap": red_cap, "horas": horas,
        "circulos": circulos, "vuelos": df_v, "noreub": df_n,
        "icao": icao_A, "reduccion": reduccion, "seed": seed, "radio": radio,
    }


# ================================================================
# CABECERA + CAPACIDADES EDITABLES
# ================================================================
st.markdown("## Simulador de cascada - Europa")
st.caption("Modelo B (prioridad al desviado) · AENA real en Espana, estimada en el resto de Europa · cierre de duracion variable.")

base = cargar_eu()
# recarga la tabla si cambia el numero de aeropuertos (p. ej. al pasar de Espana a Europa)
if "sim_caps" not in st.session_state or len(st.session_state["sim_caps"]) != len(base):
    st.session_state["sim_caps"] = base[["ident", "name", "type", "cap_h",
                                         "latitude_deg", "longitude_deg", "cap_real"]].copy()
caps = st.session_state["sim_caps"]

with st.expander("Capacidades de los aeropuertos (llegadas/hora) — editable", expanded=False):
    st.caption("Europa: capacidad del Excel (cap_real = True). Resto del mundo: estimada por tipo (27 grandes / 13 medianos). Editable.")
    edit = st.data_editor(
        caps[["ident", "name", "type", "cap_h", "cap_real"]],
        use_container_width=True, height=280, hide_index=True,
        disabled=["ident", "name", "type", "cap_real"],
        column_config={"cap_h": st.column_config.NumberColumn("cap_h (lleg/h)", min_value=1, max_value=80, step=1)})
    caps["cap_h"] = edit["cap_h"].values
    st.session_state["sim_caps"] = caps

# ================================================================
# SIDEBAR
# ================================================================
opciones = caps.sort_values("name").assign(lbl=lambda x: x["name"] + " (" + x["ident"] + ")")
lbl2icao = dict(zip(opciones["lbl"], opciones["ident"]))

with st.sidebar:
    st.markdown("### Incidencia")
    idents = opciones["ident"].tolist()
    preset = st.session_state.get("sim_icao_preset")
    if preset and preset in idents:
        idx_def = idents.index(preset)
    elif "LEMD" in idents:
        idx_def = idents.index("LEMD")
    else:
        idx_def = 0
    sel_lbl = st.selectbox("Aeropuerto afectado:", opciones["lbl"].tolist(), index=idx_def)
    icao_sel = lbl2icao.get(sel_lbl, idents[0])
    cap_sel = int(caps.loc[caps["ident"] == icao_sel, "cap_h"].values[0])
    st.caption(f"Capacidad actual: {cap_sel} llegadas/h")
    if preset and preset in idents:
        st.caption("↳ aeropuerto recibido de Analisis de Red")

    modo_datos = st.radio("Llegadas por hora:", ["Datos reales (Trino)", "Parametrico (a mano)"],
                          index=1,
                          help="Datos reales: las llegadas salen del trafico real de un dia. "
                               "Parametrico: las pones tu a mano.")
    if modo_datos == "Datos reales (Trino)":
        trino_user = st.text_input("Usuario Trino (email)", value="jaltevil@myuax.com").lower()
        fecha_real = st.date_input("Dia (UTC)", datetime(2024, 1, 16))
        hora_ini = st.slider("Hora de inicio del cierre (UTC)", 0, 23, 12)
        escala = st.toggle("Compensar vuelos sin destino", value=False,
                           help="Sube las llegadas reales para estimar el trafico total incluyendo "
                                "los vuelos sin destino verificado (que se excluyen del dato). "
                                "Los resultados quedan marcados como ESTIMACION.")
        escala_pct = st.slider("% de vuelos sin destino", 0, 40, 24, 1) if escala else 24
        N_in_param = None
    else:
        trino_user, fecha_real, hora_ini = None, None, None
        escala, escala_pct = False, 24
        N_in_param = st.number_input("Llegadas por hora", 1, 600, max(cap_sel, 48), 2,
                                     help="Vuelos que llegan al aeropuerto afectado cada hora")
    reduccion = st.slider("Reduccion de capacidad (%)", 0, 100, 100, 5, help="100% = cierre total")
    horas = st.slider("Duracion del cierre (horas)", 1, 8, 1, 1,
                      help="Cada hora entra una tanda nueva; las plazas ocupadas no se liberan")
    st.divider()
    st.markdown("### Parametros del modelo")
    occ = st.slider("Ocupacion previa de los demas (%)", 0, 95, 60, 5,
                    help="Hueco libre = 100% - este valor. El resto es trafico propio desplazable.") / 100
    radio = st.slider("Radio de desvio (km)", 100, 1500, 500, 50)
    max_niv = st.slider("Niveles maximos", 1, 5, 5, 1)
    heavy_pct = st.slider("Aviones pesados (%)", 0, 40, 10, 5,
                          help="Fraccion de aeronaves de fuselaje ancho (cat. 6, tipo A380/B747) que "
                               "solo pueden aterrizar en aeropuertos grandes. El resto va a grandes o medianos.")
    st.divider()
    st.markdown("### Visualizacion")
    ver_areas = st.toggle("Areas de 500 km (focos)", value=True)
    ver_lineas = st.toggle("Lineas de desvio", value=True)
    ver_noreub = st.toggle("Vuelos sin alternativa", value=True)
    st.divider()
    btn = st.button("▶ Simular", type="primary", use_container_width=True)

if btn:
    if modo_datos == "Datos reales (Trino)":
        try:
            with st.spinner("Consultando llegadas reales en Trino..."):
                horas24 = llegadas_reales_por_hora(fecha_real, icao_sel, trino_user)
            # ventana del cierre: desde hora_ini, H horas (envuelve si pasa de medianoche)
            N_in = [horas24[(hora_ini + k) % 24] for k in range(int(horas))]
            ventana_real = list(N_in)
            if escala:
                fct = 1.0 / (1 - escala_pct / 100.0)
                N_in = [int(round(x * fct)) for x in N_in]
            st.session_state["sim_reales_info"] = {
                "fecha": str(fecha_real), "hora_ini": hora_ini, "total_dia": sum(horas24),
                "ventana": ventana_real, "escala": escala, "escala_pct": escala_pct,
                "ventana_esc": list(N_in) if escala else None}
        except Exception as e:
            st.error(f"Error al consultar Trino: {e}")
            if "trino_conn" in st.session_state:
                del st.session_state["trino_conn"]
            st.stop()
    else:
        N_in = int(N_in_param)
        st.session_state.pop("sim_reales_info", None)
    with st.spinner("Propagando cascada hora a hora..."):
        st.session_state["simb"] = simular_b(caps, icao_sel, reduccion, N_in, horas=int(horas),
                                             occ=occ, radio=radio, max_nivel=max_niv,
                                             heavy_pct=heavy_pct / 100.0)

res = st.session_state.get("simb")
if not res:
    st.info("Configura la incidencia en la barra lateral y pulsa **Simular**.")
    st.stop()

# ================================================================
# DESLIZADOR TEMPORAL (filtra lo que se muestra, no recalcula)
# ================================================================
H = res["horas"]
if H > 1:
    hsel = st.slider("Ver acumulado hasta la hora:", 1, H, H)
else:
    hsel = 1

dfv_all = res["vuelos"]
dfn_all = res["noreub"]
dfv = dfv_all[dfv_all["hora"] <= hsel] if not dfv_all.empty else dfv_all
dfn = dfn_all[dfn_all["hora"] <= hsel] if not dfn_all.empty else dfn_all
circ = {k: v for k, v in res["circulos"].items() if v[1] <= hsel}

if H > 1:
    st.caption(f"Mostrando el acumulado tras **{hsel}** de {H} horas de cierre.")

info_r = st.session_state.get("sim_reales_info")
if info_r:
    txt = (f"📡 Datos reales del {info_r['fecha']} · {info_r['total_dia']:,} llegadas ese dia a "
           f"{res['icao']} · ventana desde las {info_r['hora_ini']:02d}:00 UTC: "
           f"{info_r['ventana']} llegadas/h")
    if info_r.get("escala"):
        txt += (f"  ·  ⚠️ ESTIMACION: +{info_r['escala_pct']}% por vuelos sin destino "
                f"→ {info_r['ventana_esc']}")
    st.caption(txt)

# ================================================================
# METRICAS (acumuladas hasta la hora seleccionada)
# ================================================================
co2 = float((dfv["dist"] * 16).sum()) if not dfv.empty else 0.0
desv_origen = sum(res["overflow_por_hora"][:hsel])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Vuelos desviados (origen)", f"{desv_origen:,}",
          help="Llegadas que el afectado no puede absorber, acumuladas hasta esta hora")
c2.metric("Vuelos movidos (total)", f"{len(dfv):,}",
          help="Incluye el trafico propio desplazado en la cascada")
c3.metric("Sin alternativa", f"{len(dfn):,}",
          delta=f"{int(dfv['nivel'].max()) if not dfv.empty else 0} niveles", delta_color="off")
c4.metric("CO₂ extra", f"{co2/1000:.1f} t",
          help=f"≈ {co2/21.77:,.0f} arboles/año · {co2/0.21:,.0f} km en coche")

# ================================================================
# MAPA
# ================================================================
caps_i = caps.set_index("ident")
fig = go.Figure()

fig.add_trace(go.Scattermap(
    lat=caps["latitude_deg"], lon=caps["longitude_deg"], mode="markers",
    marker=go.scattermap.Marker(size=5, color="rgba(160,160,160,0.5)"),
    text=caps["name"] + " · cap " + caps["cap_h"].astype(str) + " lleg/h",
    hoverinfo="text", name="Aeropuertos", showlegend=False))

if ver_areas:
    for icao, (niv, _h) in circ.items():
        if icao not in caps_i.index:
            continue
        la, lo = caps_i.loc[icao, "latitude_deg"], caps_i.loc[icao, "longitude_deg"]
        clat, clon = circulo(la, lo, res["radio"])
        col = "#FF0033" if icao == res["icao"] else NIVEL_COLOR.get(niv, "#FFFFFF")
        fig.add_trace(go.Scattermap(
            lat=clat, lon=clon, mode="lines", line=dict(width=1.2, color=col),
            fill="toself", fillcolor="rgba(255,255,255,0.03)",
            hoverinfo="none", showlegend=False, name=f"Area {icao}"))

if ver_lineas and not dfv.empty:
    for niv in sorted(dfv["nivel"].unique()):
        sub = dfv[dfv["nivel"] == niv]
        lats, lons = [], []
        for _, r in sub.iterrows():
            if r["origen"] in caps_i.index and r["destino"] in caps_i.index:
                o, a = caps_i.loc[r["origen"]], caps_i.loc[r["destino"]]
                lats += [o["latitude_deg"], a["latitude_deg"], None]
                lons += [o["longitude_deg"], a["longitude_deg"], None]
        if lats:
            fig.add_trace(go.Scattermap(
                lat=lats, lon=lons, mode="lines",
                line=dict(width=1.5, color=NIVEL_COLOR.get(niv, "#FFF")),
                opacity=0.55, hoverinfo="none", name=f"Nivel {niv} ({len(sub)} vuelos)"))

if not dfv.empty:
    rng = np.random.default_rng(res["seed"])
    for niv in sorted(dfv["nivel"].unique()):
        sub = dfv[dfv["nivel"] == niv]
        jlat, jlon, txt = [], [], []
        for _, r in sub.iterrows():
            if r["destino"] not in caps_i.index:
                continue
            a = caps_i.loc[r["destino"]]
            jlat.append(a["latitude_deg"] + rng.uniform(-0.08, 0.08))
            jlon.append(a["longitude_deg"] + rng.uniform(-0.08, 0.08))
            de = ("desplazado de " + r["origen"]) if r["tipo"] == "desplazado" else ("desviado de " + res["icao"])
            txt.append(f"Vuelo {r['vuelo']}<br>{de} → {r['destino']}<br>"
                       f"Hora {r['hora']} · nivel {niv} · +{r['dist']:.0f} km")
        if jlat:
            fig.add_trace(go.Scattermap(
                lat=jlat, lon=jlon, mode="markers",
                marker=go.scattermap.Marker(size=7, color=NIVEL_COLOR.get(niv, "#FFF"), opacity=0.9),
                text=txt, hoverinfo="text", name=f"Vuelos nivel {niv}", showlegend=False))

if ver_noreub and not dfn.empty:
    rng2 = np.random.default_rng(res["seed"] + 7)
    jlat, jlon, txt = [], [], []
    for _, r in dfn.iterrows():
        if r["origen"] not in caps_i.index:
            continue
        a = caps_i.loc[r["origen"]]
        jlat.append(a["latitude_deg"] + rng2.uniform(-0.1, 0.1))
        jlon.append(a["longitude_deg"] + rng2.uniform(-0.1, 0.1))
        txt.append(f"Vuelo {r['vuelo']} · SIN ALTERNATIVA<br>atascado en {r['origen']} (hora {r['hora']})")
    if jlat:
        fig.add_trace(go.Scattermap(
            lat=jlat, lon=jlon, mode="markers",
            marker=go.scattermap.Marker(size=9, color="#FFFFFF", opacity=1.0),
            text=txt, hoverinfo="text", name=f"Sin alternativa ({len(jlat)})"))

fa = caps_i.loc[res["icao"]]
fig.add_trace(go.Scattermap(
    lat=[fa["latitude_deg"]], lon=[fa["longitude_deg"]], mode="markers+text",
    marker=go.scattermap.Marker(size=20, color="#FF0033"),
    text=[res["icao"]], textposition="top right", textfont=dict(color="white", size=13),
    hovertext=[f"<b>{fa['name']}</b><br>AFECTADO · reduccion {res['reduccion']}%"
               f"<br>cap. reducida {res['red_cap']} lleg/h · cierre {H} h"],
    hoverinfo="text", name="Afectado", showlegend=False))

fig.update_layout(
    map_style="carto-darkmatter", margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=700,
    map=dict(center=dict(lat=float(fa["latitude_deg"]), lon=float(fa["longitude_deg"])), zoom=4.5),
    legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01,
                bgcolor="rgba(0,0,0,0.65)", font=dict(color="white", size=12)))
st.plotly_chart(fig, use_container_width=True)

# ================================================================
# EVOLUCION POR HORAS (todo el cierre)
# ================================================================
if H > 1:
    movidos_h = dfv_all.groupby("hora").size() if not dfv_all.empty else pd.Series(dtype=int)
    nore_h = dfn_all.groupby("hora").size() if not dfn_all.empty else pd.Series(dtype=int)
    evo = pd.DataFrame({"hora": range(1, H + 1)}).set_index("hora")
    evo["Vuelos movidos"] = movidos_h.reindex(evo.index, fill_value=0)
    evo["Sin alternativa"] = nore_h.reindex(evo.index, fill_value=0)
    st.markdown("### Evolucion por hora de cierre")
    st.bar_chart(evo, color=["#00CC66", "#FF3B3B"])

# ================================================================
# TABLA POR VUELO + CSV
# ================================================================
if not dfv.empty:
    tab = dfv.merge(caps[["ident", "name"]], left_on="destino", right_on="ident", how="left")
    tab["tipo"] = tab["tipo"].map({"desviado": "Desviado (afectado)", "desplazado": "Desplazado"})
    tab["co2_kg"] = (tab["dist"] * 16).round(0).astype(int)
    tab["avion"] = tab["heavy"].map({True: "Pesado", False: "Normal"})
    tab = tab.rename(columns={"vuelo": "Vuelo", "hora": "Hora", "origen": "Desde",
                              "destino": "Aterriza en", "name": "Aeropuerto destino",
                              "nivel": "Nivel", "dist": "Dist. extra (km)", "co2_kg": "CO2 (kg)",
                              "tipo": "Tipo", "avion": "Avion"})
    cols = ["Hora", "Vuelo", "Tipo", "Avion", "Desde", "Aterriza en", "Aeropuerto destino",
            "Nivel", "Dist. extra (km)", "CO2 (kg)"]
    st.markdown("### Vuelos redirigidos")
    st.dataframe(tab[cols].sort_values(["Hora", "Nivel", "Vuelo"]), use_container_width=True, height=320)
    st.download_button(
        "⬇️ Descargar vuelos (CSV)",
        data=tab[cols].to_csv(index=False),
        file_name=f"cascada_{res['icao']}_{res['reduccion']}pct_{H}h.csv",
        mime="text/csv")