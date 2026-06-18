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


CONT_NOMBRES = {"EU": "Europa", "NA": "Norteamerica", "SA": "Sudamerica",
                "AS": "Asia", "AF": "Africa", "OC": "Oceania"}


@st.cache_data(show_spinner=False)
def cargar_aeropuertos(continentes):
    """
    Carga los aeropuertos (large/medium con servicio programado) de los continentes indicados.
    Europa usa la capacidad real del Excel; el resto del mundo, la estimacion por tipo (27/13).
    'continentes' es una tupla de codigos OurAirports: EU, NA, SA, AS, AF, OC.
    """
    cap_eu = cargar_cap_excel()
    df = pd.read_csv("airports.csv")
    df["continent"] = df["continent"].fillna("NA")   # pandas lee "NA" (Norteamerica) como vacio
    df = df[df["continent"].isin(continentes) &
            df["type"].isin(["large_airport", "medium_airport"])]
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


@st.cache_data(show_spinner=False)
def aviones_inbound_ventana(fecha, icao, hora_ini, horas, usuario):
    """
    Vuelos que iban a aterrizar en 'icao' durante el cierre [T, T+horas) y que ya estaban en
    el aire al empezar el cierre, con su posicion en ese instante T y la hora en la que habrian
    aterrizado. Cada avion aparece UNA sola vez (se reparte por su lastseen), asi no hay
    duplicados entre horas.
    Devuelve DataFrame: icao24, lat, lon, hora (1..horas).
    """
    ts_day = int(datetime(fecha.year, fecha.month, fecha.day, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    T = ts_day + hora_ini * 3600          # inicio del cierre
    T_fin = T + horas * 3600
    conn = get_trino(usuario)
    cur = conn.cursor()
    # 1) vuelos a 'icao' que aterrizarian dentro de la ventana y ya volaban al empezar el cierre
    cur.execute(f"""
        SELECT icao24, lastseen
        FROM flights_data4
        WHERE day={ts_day} AND estarrivalairport='{icao}'
          AND firstseen <= {T} AND lastseen >= {T} AND lastseen < {T_fin}
          AND icao24 IS NOT NULL
    """)
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["icao24", "lat", "lon", "hora"])
    last_by = {}                          # un lastseen por icao24 (el primero dentro de la ventana)
    for ic, ls in rows:
        ls = int(ls)
        if ic not in last_by or ls < last_by[ic]:
            last_by[ic] = ls
    lista = "','".join(last_by.keys())
    # 2) posicion de cada uno en el instante del cierre T
    cur.execute(f"""
        SELECT icao24, MIN_BY(lat, time) AS lat, MIN_BY(lon, time) AS lon
        FROM state_vectors_data4
        WHERE hour={T} AND time BETWEEN {T} AND {T}+300
          AND icao24 IN ('{lista}')
          AND lat IS NOT NULL AND lon IS NOT NULL AND onground=false
        GROUP BY icao24
    """)
    pos = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
    if pos.empty:
        return pd.DataFrame(columns=["icao24", "lat", "lon", "hora"])
    pos["hora"] = pos["icao24"].map(lambda ic: int((last_by[ic] - T) // 3600) + 1)
    pos["hora"] = pos["hora"].clip(1, horas)
    return pos


@st.cache_data(show_spinner=False)
def arrivals_alternativos(fecha, hora_ini, horas, usuario):
    """
    Llegadas reales por hora a CADA aeropuerto durante el cierre [T, T+horas). Una sola consulta
    agregada para todos los aeropuertos. Permite saber la ocupacion real de los alternativos:
    hueco libre = capacidad - llegadas reales de esa hora (en vez de suponer un 60% fijo).
    Devuelve dict {icao: {hora_idx (0..horas-1): n_llegadas}}.
    """
    ts_day = int(datetime(fecha.year, fecha.month, fecha.day, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    T = ts_day + hora_ini * 3600
    T_fin = T + horas * 3600
    conn = get_trino(usuario)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT estarrivalairport AS apt,
               CAST((lastseen - {T}) / 3600 AS INTEGER) AS h,
               COUNT(*) AS n
        FROM flights_data4
        WHERE day={ts_day} AND estarrivalairport IS NOT NULL
          AND lastseen >= {T} AND lastseen < {T_fin}
        GROUP BY estarrivalairport, CAST((lastseen - {T}) / 3600 AS INTEGER)
    """)
    out = {}
    for apt, h, n in cur.fetchall():
        out.setdefault(apt, {})[int(h)] = int(n)
    return out


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
              heavy_pct=0.0, liberacion=0.0, seed=1):
    d = caps_df.set_index("ident")
    co = {i: (float(d.loc[i, "latitude_deg"]), float(d.loc[i, "longitude_deg"])) for i in d.index}
    cap = {i: int(d.loc[i, "cap_h"]) for i in d.index}
    tipo_ap = {i: str(d.loc[i, "type"]) for i in d.index}   # large/medium para compatibilidad
    # estado de capacidad PERSISTENTE durante todo el cierre (no se libera entre horas)
    free = {i: int(round(cap[i] * (1 - occ))) for i in d.index}
    sched = {i: cap[i] - free[i] for i in d.index}
    free[icao_A] = 0
    sched[icao_A] = 0
    free_ini = dict(free)            # hueco original, para la liberacion por hora

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
        # liberacion: cada hora reabren algunas plazas en los alternativos (despegues),
        # hasta recuperar como mucho el hueco inicial. El afectado sigue cerrado.
        if hora > 1 and liberacion > 0:
            for j in free:
                if j == icao_A:
                    continue
                free[j] = min(free_ini[j], free[j] + int(round(cap[j] * liberacion)))
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
# MOTOR POR POSICION - INSTANTANEA (datos reales)
# ================================================================
def simular_posicion(caps_df, aviones, icao_A, reduccion, horas=1, occ=0.60, radio=500,
                     max_nivel=5, heavy_pct=0.0, arrivals_alt=None):
    """
    Datos reales por posicion. 'aviones' = DataFrame con lat/lon/hora de los vuelos que iban a
    aterrizar en icao_A durante el cierre (cada uno en la hora en que habria aterrizado, sin
    duplicados). Como todos tenian combustible para llegar a icao_A, pueden alcanzar su zona:
    se desvian al alternativo compatible MAS CERCANO al aeropuerto cerrado que tenga sitio (se
    llenan los de alrededor primero y luego se va hacia afuera). La posicion real de cada avion
    se usa para el combustible extra de verdad: lo que vuela de mas respecto a su plan,
    (avion->alternativo) - (avion->destino), nunca negativo. El trafico desplazado sigue la
    cascada desde los aeropuertos.
    Ocupacion de los alternativos: si se pasa 'arrivals_alt' (llegadas reales por hora a cada
    aeropuerto, de Trino), el hueco de cada uno en cada hora es capacidad menos sus llegadas
    reales de esa hora, y el trafico desplazable es ese trafico real. Si no, se usa la ocupacion
    plana 'occ'.
    """
    d = caps_df.set_index("ident")
    co = {i: (float(d.loc[i, "latitude_deg"]), float(d.loc[i, "longitude_deg"])) for i in d.index}
    cap = {i: int(d.loc[i, "cap_h"]) for i in d.index}
    tipo_ap = {i: str(d.loc[i, "type"]) for i in d.index}
    free = {i: 0 for i in d.index}
    sched = {i: 0 for i in d.index}
    ocup_used = {}

    def carga_hora(h_idx):
        # capacidad de cada alternativo en esta hora: hueco = capacidad - llegadas reales.
        # Si no hay datos reales (arrivals_alt vacio), se usa la ocupacion plana 'occ'.
        for j in d.index:
            if j == icao_A:
                free[j] = 0
                sched[j] = 0
                continue
            if arrivals_alt:
                R = min(int(arrivals_alt.get(j, {}).get(h_idx, 0)), cap[j])
            else:
                R = int(round(cap[j] * occ))
            free[j] = cap[j] - R
            sched[j] = R           # trafico real programado, desplazable si se le ocupa el hueco
            ocup_used.setdefault(j, {})[h_idx] = R

    la0, lo0 = co[icao_A]
    # candidatos: aeropuertos dentro del radio del AEROPUERTO CERRADO, ordenados por cercania a el
    cand = sorted((hav(la0, lo0, co[j][0], co[j][1]), j) for j in d.index if j != icao_A)
    cand = [(dd, j) for dd, j in cand if dd <= radio]

    dist_cache = {}

    def cercanos_ap(src):
        if src in dist_cache:
            return dist_cache[src]
        la, lo = co[src]
        out = sorted((dd, j) for dd, j in
                     ((hav(la, lo, co[j][0], co[j][1]), j) for j in d.index if j != src) if dd <= radio)
        dist_cache[src] = out
        return out

    vuelos, no_reub = [], []
    circulos = {icao_A: (1, 1)}
    nid = 0

    def nuevos_vuelos(n, prefijo, src_tipo):
        nonlocal nid
        nh2 = int(round(n * heavy_pct)) if src_tipo == "large_airport" else 0
        out = [(f"{prefijo}{nid + k:04d}", k < nh2) for k in range(n)]
        nid += n
        return out

    def cascada(bumped_total, hora):
        # propaga el trafico desplazado desde los aeropuertos (nivel 2+)
        q = deque()
        for j, c in bumped_total.items():
            circulos.setdefault(j, (2, hora))
            q.append((j, nuevos_vuelos(c, f"{j}-", tipo_ap.get(j)), "desplazado", 2))
        while q:
            src, lote, tipo, niv = q.popleft()
            if not lote:
                continue
            olat, olon = co[src]
            if niv > max_nivel:
                no_reub.extend({"vuelo": f, "origen": src, "tipo": tipo, "hora": hora, "heavy": h,
                                "olat": olat, "olon": olon} for f, h in lote)
                continue
            cands = cercanos_ap(src)
            if not cands:
                no_reub.extend({"vuelo": f, "origen": src, "tipo": tipo, "hora": hora, "heavy": h,
                                "olat": olat, "olon": olon} for f, h in lote)
                continue
            pend_h = [f for f, h in lote if h]
            pend_n = [f for f, h in lote if not h]
            bumped = []
            for capdict, es_bump in ((free, False), (sched, True)):
                for dd, j in cands:
                    if not pend_h and not pend_n:
                        break
                    libre = capdict[j]
                    if libre <= 0:
                        continue
                    j_grande = tipo_ap.get(j) == "large_airport"
                    dist_j = round(hav(olat, olon, co[j][0], co[j][1]), 1)
                    usados = 0
                    if j_grande:
                        while libre > 0 and pend_h:
                            f = pend_h.pop(0)
                            vuelos.append({"vuelo": f, "origen": src, "destino": j, "nivel": niv,
                                           "dist": dist_j, "tipo": tipo, "bump": es_bump,
                                           "hora": hora, "heavy": True, "olat": olat, "olon": olon})
                            libre -= 1
                            usados += 1
                    while libre > 0 and pend_n:
                        f = pend_n.pop(0)
                        vuelos.append({"vuelo": f, "origen": src, "destino": j, "nivel": niv,
                                       "dist": dist_j, "tipo": tipo, "bump": es_bump,
                                       "hora": hora, "heavy": False, "olat": olat, "olon": olon})
                        libre -= 1
                        usados += 1
                    if usados > 0:
                        capdict[j] -= usados
                        if es_bump:
                            bumped.append((j, usados))
            for f in pend_h:
                no_reub.append({"vuelo": f, "origen": src, "tipo": tipo, "hora": hora, "heavy": True,
                                "olat": olat, "olon": olon})
            for f in pend_n:
                no_reub.append({"vuelo": f, "origen": src, "tipo": tipo, "hora": hora, "heavy": False,
                                "olat": olat, "olon": olon})
            for j, c in bumped:
                circulos.setdefault(j, (niv + 1, hora))
                q.append((j, nuevos_vuelos(c, f"{j}-", tipo_ap.get(j)), "desplazado", niv + 1))

    H = int(horas)
    av = aviones.dropna(subset=["lat", "lon"]).copy() if (aviones is not None and not aviones.empty) \
        else pd.DataFrame(columns=["lat", "lon", "hora"])
    if not av.empty and "hora" not in av.columns:
        av["hora"] = 1
    overflow_por_hora = []

    for hora in range(1, H + 1):
        carga_hora(hora - 1)        # huecos reales de esta hora en cada alternativo

        sub = av[av["hora"] == hora].copy() if not av.empty else av
        if sub.empty:
            overflow_por_hora.append(0)
            continue
        sub["d_aff"] = [hav(la0, lo0, float(r.lat), float(r.lon)) for r in sub.itertuples()]
        sub = sub.sort_values("d_aff").reset_index(drop=True)   # los mas cercanos a Madrid, primero
        M = len(sub)
        n_div = M if reduccion >= 100 else int(round(M * reduccion / 100.0))
        # en cierre parcial, los mas cercanos aterrizan; el resto se desvia
        divert = sub.iloc[M - n_div:].reset_index(drop=True) if n_div > 0 else sub.iloc[0:0]
        divert = divert.sort_values("d_aff").reset_index(drop=True)
        overflow_por_hora.append(n_div)
        nh = int(round(n_div * heavy_pct))

        bumped_total = {}
        for idx, r in enumerate(divert.itertuples()):
            es_heavy = idx < nh
            la, lo = float(r.lat), float(r.lon)
            d_dest = hav(la0, lo0, la, lo)               # avion -> destino original (lo que iba a volar)
            fid = f"AV-{nid:04d}"
            nid += 1
            colocado = False
            for capdict, es_bump in ((free, False), (sched, True)):
                for dd_m, j in cand:                     # candidatos ordenados por cercania al cerrado
                    if capdict[j] <= 0:
                        continue
                    if es_heavy and tipo_ap.get(j) != "large_airport":
                        continue
                    d_alt = hav(la, lo, co[j][0], co[j][1])      # avion -> alternativo
                    extra = max(0.0, d_alt - d_dest)             # lo que vuela de mas
                    capdict[j] -= 1
                    vuelos.append({"vuelo": fid, "origen": icao_A, "destino": j, "nivel": 1,
                                   "dist": round(extra, 1), "tipo": "desviado", "bump": es_bump,
                                   "hora": hora, "heavy": es_heavy, "olat": la, "olon": lo})
                    if es_bump:
                        bumped_total[j] = bumped_total.get(j, 0) + 1
                    colocado = True
                    break
                if colocado:
                    break
            if not colocado:
                no_reub.append({"vuelo": fid, "origen": icao_A, "tipo": "desviado", "hora": hora,
                                "heavy": es_heavy, "olat": la, "olon": lo})

        cascada(bumped_total, hora)

    df_v = pd.DataFrame(vuelos)
    df_n = pd.DataFrame(no_reub)
    return {
        "overflow_por_hora": overflow_por_hora, "llegadas_h": overflow_por_hora,
        "red_cap": int(cap[icao_A] * (1 - reduccion / 100)), "horas": H,
        "circulos": circulos, "vuelos": df_v, "noreub": df_n,
        "icao": icao_A, "reduccion": reduccion, "seed": 1, "radio": radio,
        "aviones": av, "modo": "posicion", "ocupacion": ocup_used, "ocup_es_real": bool(arrivals_alt),
    }


# ================================================================
# CABECERA + CAPACIDADES EDITABLES
# ================================================================
st.markdown("## Simulador de cascada")
st.caption("Modelo B (prioridad al desviado) · AENA real en Espana, estimada en el resto del mundo · cierre de duracion variable.")

with st.sidebar:
    st.markdown("### Cobertura")
    conts_sel = st.multiselect(
        "Continentes (aeropuertos disponibles):",
        list(CONT_NOMBRES.keys()), default=["EU"],
        format_func=lambda c: CONT_NOMBRES[c],
        help="Aeropuertos que el simulador puede usar como afectado o como alternativa de desvio. "
             "Para cerrar un aeropuerto fuera de Europa, marca aqui su continente.")
    if not conts_sel:
        conts_sel = ["EU"]

base = cargar_aeropuertos(tuple(sorted(conts_sel)))
# recarga la tabla si cambia el numero de aeropuertos (p. ej. al anadir un continente)
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

    modo_datos = st.radio("Modo de datos:",
                          ["Parametrico (a mano)", "Datos reales por hora (Trino)",
                           "Instantanea por posicion (Trino)"],
                          index=0,
                          help="Parametrico: pones las llegadas a mano. Por hora: salen del trafico real "
                               "de un dia (conteos). Por posicion: foto de un instante, desvia los aviones "
                               "que en ese momento iban al aeropuerto desde su posicion real.")
    if modo_datos == "Parametrico (a mano)":
        trino_user = fecha_real = hora_ini = None
        escala, escala_pct = False, 24
        N_in_param = st.number_input("Llegadas por hora", 1, 600, max(cap_sel, 48), 2,
                                     help="Vuelos que llegan al aeropuerto afectado cada hora")
        horas = st.slider("Duracion del cierre (horas)", 1, 8, 1, 1,
                          help="Cada hora entra una tanda nueva; las plazas ocupadas no se liberan")
    elif modo_datos == "Datos reales por hora (Trino)":
        trino_user = st.text_input("Usuario Trino (email)", value="jaltevil@myuax.com").lower()
        fecha_real = st.date_input("Dia (UTC)", datetime(2024, 1, 16))
        hora_ini = st.slider("Hora de inicio del cierre (UTC)", 0, 23, 12)
        escala = st.toggle("Estimar trafico real (compensar vuelos sin destino)", value=False,
                           help="Sube las llegadas reales del afectado alrededor de un 32%, porque cerca "
                                "del 24% de los vuelos no tienen destino verificado y no se cuentan. No son "
                                "vuelos aleatorios: es la misma llegada estimada al alza. Marca ESTIMACION.")
        escala_pct = 24
        N_in_param = None
        horas = st.slider("Duracion del cierre (horas)", 1, 8, 1, 1,
                          help="Cada hora entra una tanda nueva; las plazas ocupadas no se liberan")
    else:  # Datos reales por posicion
        trino_user = st.text_input("Usuario Trino (email)", value="jaltevil@myuax.com").lower()
        fecha_real = st.date_input("Dia (UTC)", datetime(2024, 1, 16))
        hora_ini = st.slider("Hora de inicio del cierre (UTC)", 0, 23, 12)
        horas = st.slider("Duracion del cierre (horas)", 1, 6, 1, 1,
                          help="Cada hora se traen los aviones que iban al aeropuerto en ese momento y "
                               "se desvian desde su posicion. La capacidad persiste entre horas.")
        escala, escala_pct = False, 24
        N_in_param = None
        st.caption("Los aviones se desvian desde donde estan, hacia el aeropuerto alcanzable mas "
                   "cercano al cerrado.")
    reduccion = st.slider("Reduccion de capacidad (%)", 0, 100, 100, 5, help="100% = cierre total")
    st.divider()
    st.markdown("### Parametros del modelo")
    radio = st.slider("Radio de desvio (km)", 100, 1500, 500, 50)
    max_niv = st.slider("Niveles maximos", 1, 5, 5, 1)
    heavy_pct = st.slider("Aviones pesados (%)", 0, 40, 10, 5,
                          help="Fraccion de aeronaves de fuselaje ancho (cat. 6, tipo A380/B747) que "
                               "solo pueden aterrizar en aeropuertos grandes. El resto va a grandes o medianos.")
    if modo_datos == "Instantanea por posicion (Trino)":
        liberacion = 0.0
        usar_ocup_real = st.toggle("Ocupacion real de los alternativos (Trino)", value=True,
            help="ON: el hueco de cada aeropuerto sale de sus llegadas reales de ese dia y hora. "
                 "OFF: lo pones tu con un % fijo igual para todos, para estudiar escenarios "
                 "hipoteticos (p. ej. 'y si estuvieran al 80%?').")
        if usar_ocup_real:
            occ = 0.60
            st.caption("Hueco de cada alternativo = su capacidad menos sus llegadas reales de ese dia y hora.")
        else:
            occ = st.slider("Ocupacion supuesta de los alternativos (%)", 0, 95, 60, 5,
                            help="Todos los alternativos al mismo % de ocupacion. "
                                 "Hueco libre = 100% - este valor.") / 100
            st.caption("Ocupacion hipotetica e igual para todos. Para comparar escenarios "
                       "('y si en vez del trafico real estuvieran al X%?').")
    else:
        usar_ocup_real = False
        occ = st.slider("Ocupacion previa de los demas (%)", 0, 95, 60, 5,
                        help="Hueco libre = 100% - este valor. El resto es trafico propio desplazable.") / 100
        liberacion = st.slider("Liberacion de plazas por hora (%)", 0, 50, 10, 5,
                               help="Cada hora reabren algunas plazas en los alternativos (aviones que "
                                    "despegan), hasta recuperar como mucho el hueco inicial. 0 = no se libera nada.")
    st.divider()
    st.markdown("### Visualizacion")
    ver_areas = st.toggle("Areas de 500 km (focos)", value=True)
    ver_lineas = st.toggle("Lineas de desvio", value=True)
    ver_noreub = st.toggle("Vuelos sin alternativa", value=True)
    st.divider()
    btn = st.button("▶ Simular", type="primary", use_container_width=True)

if btn:
    if modo_datos == "Parametrico (a mano)":
        N_in = int(N_in_param)
        st.session_state.pop("sim_reales_info", None)
        with st.spinner("Propagando cascada hora a hora..."):
            st.session_state["simb"] = simular_b(caps, icao_sel, reduccion, N_in, horas=int(horas),
                                                 occ=occ, radio=radio, max_nivel=max_niv,
                                                 heavy_pct=heavy_pct / 100.0, liberacion=liberacion / 100.0)
    elif modo_datos == "Datos reales por hora (Trino)":
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
                "modo": "horas", "fecha": str(fecha_real), "hora_ini": hora_ini,
                "total_dia": sum(horas24), "ventana": ventana_real, "escala": escala,
                "escala_pct": escala_pct, "ventana_esc": list(N_in) if escala else None}
        except Exception as e:
            st.error(f"Error al consultar Trino: {e}")
            if "trino_conn" in st.session_state:
                del st.session_state["trino_conn"]
            st.stop()
        with st.spinner("Propagando cascada hora a hora..."):
            st.session_state["simb"] = simular_b(caps, icao_sel, reduccion, N_in, horas=int(horas),
                                                 occ=occ, radio=radio, max_nivel=max_niv,
                                                 heavy_pct=heavy_pct / 100.0, liberacion=liberacion / 100.0)
    else:  # Datos reales por posicion
        try:
            with st.spinner("Consultando vuelos y posiciones reales en Trino..."):
                aviones = aviones_inbound_ventana(fecha_real, icao_sel, hora_ini, int(horas), trino_user)
                arr_alt = (arrivals_alternativos(fecha_real, hora_ini, int(horas), trino_user)
                           if usar_ocup_real else {})
            if aviones.empty:
                st.warning("Trino no devolvio vuelos hacia ese aeropuerto en esa ventana. "
                           "Prueba otra hora, mas duracion u otro aeropuerto.")
                st.stop()
            st.session_state["sim_reales_info"] = {
                "modo": "posicion", "fecha": str(fecha_real), "hora_ini": hora_ini,
                "horas": int(horas), "n_aviones": len(aviones),
                "ocup_real": usar_ocup_real, "occ_fija": (None if usar_ocup_real else occ)}
        except Exception as e:
            st.error(f"Error al consultar Trino: {e}")
            if "trino_conn" in st.session_state:
                del st.session_state["trino_conn"]
            st.stop()
        with st.spinner("Desviando desde las posiciones reales..."):
            st.session_state["simb"] = simular_posicion(caps, aviones, icao_sel, reduccion,
                                                        horas=int(horas), occ=occ, radio=radio,
                                                        max_nivel=max_niv, heavy_pct=heavy_pct / 100.0,
                                                        arrivals_alt=(arr_alt or None))

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
if info_r and info_r.get("modo") == "posicion":
    h0 = info_r['hora_ini']
    Hp = info_r.get('horas', 1)
    rango = f"{h0:02d}:00 UTC" if Hp == 1 else f"{h0:02d}:00-{(h0 + Hp) % 24:02d}:00 UTC"
    if info_r.get("ocup_real"):
        extra = " · ocupacion real de alternativos"
    elif info_r.get("occ_fija") is not None:
        extra = f" · ocupacion hipotetica {round(info_r['occ_fija'] * 100)}% (igual para todos)"
    else:
        extra = ""
    st.caption(f"📡 Posiciones reales del {info_r['fecha']} · cierre {rango} ({Hp} h) · "
               f"{info_r['n_aviones']} aviones hacia {res['icao']}, desviados desde su posicion{extra}.")
elif info_r:
    txt = (f"📡 Datos reales del {info_r['fecha']} · {info_r['total_dia']:,} llegadas ese dia a "
           f"{res['icao']} · ventana desde las {info_r['hora_ini']:02d}:00 UTC: "
           f"{info_r['ventana']} llegadas/h")
    if info_r.get("escala"):
        txt += (f"  ·  ⚠️ ESTIMACION: llegadas al afectado aumentadas para compensar el ~"
                f"{info_r['escala_pct']}% de vuelos sin destino verificado → {info_r['ventana_esc']} lleg/h")
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
    es_pos = res.get("modo") == "posicion"
    tiene_o = "olat" in dfv.columns
    for niv in sorted(dfv["nivel"].unique()):
        sub = dfv[dfv["nivel"] == niv]
        lats, lons = [], []
        for _, r in sub.iterrows():
            if r["destino"] not in caps_i.index:
                continue
            a = caps_i.loc[r["destino"]]
            if es_pos and tiene_o:                 # primera oleada: desde la posicion del avion
                olat, olon = r["olat"], r["olon"]
            elif r["origen"] in caps_i.index:       # cascada / modo por horas: desde el aeropuerto
                o = caps_i.loc[r["origen"]]
                olat, olon = o["latitude_deg"], o["longitude_deg"]
            else:
                continue
            lats += [olat, a["latitude_deg"], None]
            lons += [olon, a["longitude_deg"], None]
        if lats:
            fig.add_trace(go.Scattermap(
                lat=lats, lon=lons, mode="lines",
                line=dict(width=1.5, color=NIVEL_COLOR.get(niv, "#FFF")),
                opacity=0.55, hoverinfo="none", name=f"Nivel {niv} ({len(sub)} vuelos)"))

if not dfv.empty:
    # Un marcador por AEROPUERTO de destino (no por vuelo), con su codigo ICAO visible,
    # coloreado por nivel de cascada y dimensionado por nº de vuelos recibidos.
    agg = (dfv.groupby("destino")
           .agg(vuelos=("vuelo", "size"), nivel=("nivel", "min"), pesados=("heavy", "sum"))
           .reset_index())
    agg = agg[agg["destino"].isin(caps_i.index)]
    for niv in sorted(agg["nivel"].unique()):
        sub = agg[agg["nivel"] == niv]
        lat = [float(caps_i.loc[i, "latitude_deg"]) for i in sub["destino"]]
        lon = [float(caps_i.loc[i, "longitude_deg"]) for i in sub["destino"]]
        sizes = [12 + min(18, int(v) // 3) for v in sub["vuelos"]]   # 12-30 segun vuelos
        ocup = res.get("ocupacion") or {}
        hov = []
        for _, r in sub.iterrows():
            ic = r["destino"]
            nom = caps_i.loc[ic, "name"]
            cap_ap = int(caps_i.loc[ic, "cap_h"])
            extra = f" ({int(r['pesados'])} pesados)" if r["pesados"] else ""
            d_here = int(r["vuelos"])
            sufijo_h = f" en {hsel}h" if hsel > 1 else ""
            linea_ocup = ""
            if ocup:
                propias = sum(int(ocup.get(ic, {}).get(k, 0)) for k in range(hsel))  # propias hasta la hora mostrada
                plazas = cap_ap * hsel
                pct = min(100, round((propias + d_here) / plazas * 100)) if plazas else 0
                linea_ocup = (f"<br>En {hsel}h caben {plazas} · {propias} propias + {d_here} desviados "
                              f"= {pct}% lleno")
            hov.append(f"<b>{nom}</b> ({ic}) · Nivel {niv}<br>"
                       f"Capacidad: {cap_ap} lleg/h · {d_here} desviados aqui{sufijo_h}{extra}{linea_ocup}")
        fig.add_trace(go.Scattermap(
            lat=lat, lon=lon, mode="markers+text",
            marker=go.scattermap.Marker(size=sizes, color=NIVEL_COLOR.get(niv, "#FFF"), opacity=0.95),
            text=list(sub["destino"]), textposition="top right",
            textfont=dict(color="white", size=11),
            hovertext=hov, hoverinfo="text", name=f"Nivel {niv} ({len(sub)} aeropuertos)"))

# Aviones en su posicion real (solo modo datos reales por posicion)
avs = res.get("aviones")
if isinstance(avs, pd.DataFrame) and not avs.empty:
    if "hora" in avs.columns:
        avs = avs[avs["hora"] <= hsel]
    if not avs.empty:
        fig.add_trace(go.Scattermap(
            lat=avs["lat"], lon=avs["lon"], mode="markers",
            marker=go.scattermap.Marker(size=6, color="#00E5FF", opacity=0.85),
            text=[f"Avion {ic} · iba a {res['icao']}" for ic in avs.get("icao24", avs.index)],
            hoverinfo="text", name=f"Aviones en ruta ({len(avs)})"))

if ver_noreub and not dfn.empty:
    rng2 = np.random.default_rng(res["seed"] + 7)
    jlat, jlon, txt = [], [], []
    tiene_o = "olat" in dfn.columns
    for _, r in dfn.iterrows():
        if tiene_o and pd.notna(r.get("olat")):
            la, lo = float(r["olat"]), float(r["olon"])
        elif r["origen"] in caps_i.index:
            a = caps_i.loc[r["origen"]]
            la, lo = float(a["latitude_deg"]), float(a["longitude_deg"])
        else:
            continue
        jlat.append(la + rng2.uniform(-0.1, 0.1))
        jlon.append(lo + rng2.uniform(-0.1, 0.1))
        txt.append(f"Vuelo {r['vuelo']} · SIN ALTERNATIVA<br>desde {r['origen']}")
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
# AEROPUERTOS QUE RECIBEN - HORA A HORA
# ================================================================
if not dfv.empty:
    ocup = res.get("ocupacion") or {}
    per = (dfv.groupby(["destino", "hora"])
           .agg(desv=("vuelo", "size"), pes=("heavy", "sum"), niv=("nivel", "min"))
           .reset_index())
    per = per[per["destino"].isin(caps_i.index)]
    filas = []
    for _, r in per.iterrows():
        ic = r["destino"]
        h = int(r["hora"])
        D = int(r["desv"])
        cap_ap = int(caps_i.loc[ic, "cap_h"])
        fila = {"Hora": h, "Aeropuerto": caps_i.loc[ic, "name"], "ICAO": ic, "Cap/h": cap_ap}
        if ocup:
            R = int(ocup.get(ic, {}).get(h - 1, 0))          # llegadas propias de ESA hora
            fila["Propias"] = R
            fila["Libres"] = max(0, cap_ap - R)              # huecos antes de los desvios, esa hora
            fila["Desviados"] = D
            fila["% lleno"] = min(100, round((R + D) / cap_ap * 100)) if cap_ap else 0
        else:
            fila["Desviados"] = D
        fila["Pesados"] = int(r["pes"])
        fila["Nivel"] = int(r["niv"])
        filas.append(fila)
    resu = pd.DataFrame(filas).sort_values(["Hora", "Nivel", "Desviados"], ascending=[True, True, False])
    st.markdown("### Aeropuertos que reciben desvios, hora a hora" if H > 1
                else "### Aeropuertos que reciben desvios")
    _fuente = "llegadas reales" if res.get("ocup_es_real", True) else "ocupacion supuesta"
    st.caption(f"Por aeropuerto y hora: capacidad, {_fuente} de esa hora, huecos libres, vuelos "
               "recibidos y como queda de lleno. El hueco se recalcula en cada hora del cierre.")
    st.dataframe(resu, use_container_width=True, height=340, hide_index=True)

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