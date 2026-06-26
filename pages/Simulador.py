"""
pages/5_Simulador.py
TFG: SIMULACIÓN Y ANÁLISIS DEL IMPACTO OPERATIVO DE LA RED AÉREA GLOBAL ANTE DISRUPCIONES SISTÉMICAS
Jacob Altenburger Villar - UAX 2026
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
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
def _ruta_datos(nombre):
    for _base in (_AQUI, _AQUI.parent, Path.cwd()):
        _p = _base / nombre
        if _p.exists():
            return str(_p)
    return nombre  # si no esta en ningun sitio, deja que pandas avise

st.set_page_config(page_title="TFG - Simulador", page_icon="🛬", layout="wide")

NIVEL_COLOR = {1: "#00CC66", 2: "#FFD400", 3: "#FF8C00", 4: "#FF3B3B", 5: "#A020F0"}


CAP_EST = {"large_airport": 27, "medium_airport": 13}


PARKING_EST = {"large_airport": 35, "medium_airport": 10, "small_airport": 4}

EXCEL_CAP = "capacidad_aterrizaje_aeropuertos_europa.xlsx"


EXCEL_PARKING = "parking_aeropuertos.xlsx"


@st.cache_data(show_spinner=False)
def cargar_cap_excel():
    cap = pd.read_excel(_ruta_datos(EXCEL_CAP))
    return dict(zip(cap["ICAO"].astype(str), cap["Llegadas/h (modelo)"]))


@st.cache_data(show_spinner=False)
def cargar_parking():
    """Puestos de parking reales por ICAO (del AIP/Planes Directores). Si no esta el fichero,
    devuelve {} y el simulador usa la estimacion por tipo."""
    try:
        pk = pd.read_excel(_ruta_datos(EXCEL_PARKING))
        return dict(zip(pk["ICAO"].astype(str), pk["Parking"].astype(int)))
    except Exception:
        return {}


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
    df = pd.read_csv(_ruta_datos("airports.csv"))
    df["continent"] = df["continent"].fillna("NA")   # pandas lee "NA" (Norteamerica) como vacio
    df = df[df["continent"].isin(continentes) &
            df["type"].isin(["large_airport", "medium_airport"])]
    df = df.dropna(subset=["latitude_deg", "longitude_deg"]).copy()
    df = df[df["scheduled_service"] == "yes"]
    df["cap_real"] = df["ident"].isin(cap_eu)        # True = capacidad del Excel europeo
    df["cap_h"] = df["ident"].map(cap_eu)            # Europa: valor del Excel
    df["cap_h"] = df["cap_h"].fillna(                # resto del mundo: estimacion por tipo
        df["type"].map(CAP_EST)).astype(int)
    park = cargar_parking()                          # puestos reales (AIP) por ICAO
    df["parking_real"] = df["ident"].isin(park)      # True = dato leido del AIP/Plan Director
    df["parking"] = df["ident"].map(park)            # Espana/Portugal leidos: valor real
    df["parking"] = df["parking"].fillna(            # resto: estimacion por tipo
        df["type"].map(PARKING_EST)).fillna(5).astype(int)
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def coords_todos_aeropuertos():
    """Coordenadas de TODOS los aeropuertos (cualquier tipo y continente) por codigo ICAO.
    Se usa para localizar el aeropuerto de SALIDA de un vuelo, que puede estar en cualquier
    parte del mundo. Devuelve {ident: (lat, lon)}."""
    d = pd.read_csv(_ruta_datos("airports.csv")).dropna(subset=["latitude_deg", "longitude_deg"])
    return {r.ident: (float(r.latitude_deg), float(r.longitude_deg)) for r in d.itertuples()}



# DATOS REALES (Trino): llegadas reales por hora a un aeropuerto

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
    Devuelve DataFrame: icao24, lat, lon, hora (1..horas), origen, orig_lat, orig_lon.
    El origen (estdepartureairport) puede ser desconocido en Trino -> orig_lat/orig_lon = NaN.
    """
    ts_day = int(datetime(fecha.year, fecha.month, fecha.day, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    T = ts_day + hora_ini * 3600          # inicio del cierre
    T_fin = T + horas * 3600
    conn = get_trino(usuario)
    cur = conn.cursor()
    # 1) vuelos a 'icao' que aterrizarian dentro de la ventana y ya volaban al empezar el cierre
    cur.execute(f"""
        SELECT icao24, lastseen, estdepartureairport AS origen
        FROM flights_data4
        WHERE day={ts_day} AND estarrivalairport='{icao}'
          AND firstseen <= {T} AND lastseen >= {T} AND lastseen < {T_fin}
          AND icao24 IS NOT NULL
    """)
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["icao24", "lat", "lon", "hora", "origen", "orig_lat", "orig_lon"])
    last_by, orig_by = {}, {}              # un lastseen y un origen por icao24 (el primero en ventana)
    for ic, ls, og in rows:
        ls = int(ls)
        if ic not in last_by or ls < last_by[ic]:
            last_by[ic] = ls
            orig_by[ic] = og
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
        return pd.DataFrame(columns=["icao24", "lat", "lon", "hora", "origen", "orig_lat", "orig_lon"])
    pos["hora"] = pos["icao24"].map(lambda ic: int((last_by[ic] - T) // 3600) + 1)
    pos["hora"] = pos["hora"].clip(1, horas)
    # origen y sus coordenadas (NaN si Trino no conoce el aeropuerto de salida)
    coords = coords_todos_aeropuertos()
    pos["origen"]   = pos["icao24"].map(orig_by)
    pos["orig_lat"] = pos["origen"].map(lambda o: coords.get(o, (float("nan"), float("nan")))[0])
    pos["orig_lon"] = pos["origen"].map(lambda o: coords.get(o, (float("nan"), float("nan")))[1])
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


# MOTOR DE CASCADA - MODELO B CON DIMENSION TEMPORAL

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
    free_ini = dict(free)

    red_cap = int(cap[icao_A] * (1 - reduccion / 100))

    if isinstance(N_in, (list, tuple, np.ndarray)):
        llegadas_h = [int(x) for x in N_in][:horas]
        llegadas_h += [0] * (horas - len(llegadas_h))
    else:
        llegadas_h = [int(N_in)] * horas
    overflow_por_hora = [max(0, n - red_cap) for n in llegadas_h]

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
    circulos = {icao_A: (1, 1)}
    nid = 0

    def nuevos_vuelos(n, prefijo, src_tipo):
        nonlocal nid
        nh = int(round(n * heavy_pct)) if src_tipo == "large_airport" else 0
        out = [(f"{prefijo}{nid + k:04d}", k < nh) for k in range(n)]
        nid += n
        return out

    for hora in range(1, horas + 1):
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
                    dist_j = round(hav(*co[src], *co[j]), 1)
                    usados = 0
                    if j_grande:
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


# MOTOR POR POSICION - INSTANTANEA (datos reales)

def simular_posicion(caps_df, aviones, icao_A, reduccion, horas=1, occ=0.60, radio=500,
                     max_nivel=5, heavy_pct=0.0, arrivals_alt=None, occ_park=0.60, frac_retorno=0.60):
    """
    Datos reales por posicion. 'aviones' = DataFrame con lat/lon/hora de los vuelos que iban a
    aterrizar en icao_A durante el cierre. Reparto en dos fases:
      - NIVEL 1: aviones reales -> aeropuertos a <= radio del cerrado. Como todos tenian
        combustible para llegar al cerrado, alcanzan su zona. Se colocan en el alternativo
        compatible MAS CERCANO al cerrado con sitio (pista de esa hora + parking). El combustible
        extra es real: (avion->alternativo) - (avion->cerrado), nunca negativo.
      - NIVELES 2+: el sobrante que el anillo 1 no puede absorber NO desaparece. Se propaga por la
        red en anchura (BFS), pero SOLO desde aeropuertos que realmente se saturaron (recibieron
        desvios y se llenaron). Cada aeropuerto saturado empuja su sobrante hacia sus vecinos a
        <= radio, que pasan a ser el nivel siguiente. El origen de un desvio de nivel 2+ es por
        tanto un aeropuerto real saturado (la "puerta"), no un punto geometrico. El avion real no
        tiene combustible para el anillo 2, asi que estos vuelos son recuentos SINTETICOS, sin
        posicion, que modelan como se propaga la DEMANDA (peor caso conservador). Reproduce el
        efecto cascada de la memoria (6.2.3): Zaragoza se llena (nivel 1) y empuja hacia Toulouse
        (nivel 2), que esta a 294 km de Zaragoza aunque a 536 del cerrado.
    Ocupacion de pista: con 'arrivals_alt' (Trino) el hueco de cada alternativo cada hora es
    capacidad - llegadas reales; si no, ocupacion plana 'occ'. Parking: limite acumulativo,
    total * (1 - occ_park) puestos libres para desvios, no se vacia durante el cierre.
    """
    d = caps_df.set_index("ident")
    co = {i: (float(d.loc[i, "latitude_deg"]), float(d.loc[i, "longitude_deg"])) for i in d.index}
    cap = {i: int(d.loc[i, "cap_h"]) for i in d.index}
    tipo_ap = {i: str(d.loc[i, "type"]) for i in d.index}
    if "parking" in d.columns:
        park_tot = {i: int(d.loc[i, "parking"]) for i in d.index}
    else:
        park_tot = {i: PARKING_EST.get(str(d.loc[i, "type"]), 5) for i in d.index}
    park_libre_ini = {i: (0 if i == icao_A else max(0, int(round(park_tot[i] * (1 - occ_park)))))
                      for i in d.index}
    park_div = {i: 0 for i in d.index}
    free = {i: 0 for i in d.index}
    ocup_used = {}

    la0, lo0 = co[icao_A]
    dist_cache = {}

    def cercanos_ap(src):
        if src in dist_cache:
            return dist_cache[src]
        la, lo = co[src]
        out = sorted((dd, j) for dd, j in
                     ((hav(la, lo, co[j][0], co[j][1]), j) for j in d.index if j != src) if dd <= radio)
        dist_cache[src] = out
        return out

    cand1 = cercanos_ap(icao_A)

    def carga_hora(h_idx):
        for j in d.index:
            if j == icao_A:
                free[j] = 0
                continue
            if arrivals_alt:
                R = min(int(arrivals_alt.get(j, {}).get(h_idx, 0)), cap[j])
            else:
                R = int(round(cap[j] * occ))
            free[j] = cap[j] - R
            ocup_used.setdefault(j, {})[h_idx] = R

    def cabe(j, es_heavy):
        if free[j] <= 0:
            return False
        if park_div[j] >= park_libre_ini[j]:
            return False
        if es_heavy and tipo_ap.get(j) != "large_airport":
            return False
        return True

    def lleno(j):
        return free[j] <= 0 or park_div[j] >= park_libre_ini[j]

    vuelos, no_reub, retornos, cascada_av = [], [], [], []
    circulos = {icao_A: (1, 1)}
    nid = 0

    H = int(horas)
    av = aviones.dropna(subset=["lat", "lon"]).copy() if (aviones is not None and not aviones.empty) \
        else pd.DataFrame(columns=["lat", "lon", "hora"])
    if not av.empty and "hora" not in av.columns:
        av["hora"] = 1
    for _c in ("icao24", "origen", "orig_lat", "orig_lon"):   # por si no traen estas columnas
        if _c not in av.columns:
            av[_c] = "" if _c in ("icao24", "origen") else float("nan")
    overflow_por_hora = []

    for hora in range(1, H + 1):
        carga_hora(hora - 1)
        sub = av[av["hora"] == hora].copy() if not av.empty else av
        if sub.empty:
            overflow_por_hora.append(0)
            continue
        sub["d_aff"] = [hav(la0, lo0, float(r.lat), float(r.lon)) for r in sub.itertuples()]
        sub = sub.sort_values("d_aff").reset_index(drop=True)
        M = len(sub)
        n_div = M if reduccion >= 100 else int(round(M * reduccion / 100.0))
        divert = sub.iloc[M - n_div:].reset_index(drop=True) if n_div > 0 else sub.iloc[0:0]
        divert = divert.sort_values("d_aff").reset_index(drop=True)
        overflow_por_hora.append(n_div)
        nh = int(round(n_div * heavy_pct))

        # NIVEL 1: aviones reales -> anillo 1 (mas cercano al cerrado con sitio)
        recibio_h = {}
        over_h, over_n = [], []
        for idx, r in enumerate(divert.itertuples()):
            es_heavy = idx < nh
            la, lo = float(r.lat), float(r.lon)
            d_dest = hav(la0, lo0, la, lo)
            ola = getattr(r, "orig_lat", float("nan"))
            olo = getattr(r, "orig_lon", float("nan"))
            o_icao = getattr(r, "origen", "") or ""
            colocado = False
            for dd_m, j in cand1:
                if not cabe(j, es_heavy):
                    continue
                extra = max(0.0, hav(la, lo, co[j][0], co[j][1]) - d_dest)
                free[j] -= 1
                park_div[j] += 1
                recibio_h[j] = recibio_h.get(j, 0) + 1
                fid = f"AV-{nid:04d}"; nid += 1
                vuelos.append({"vuelo": fid, "origen": icao_A, "destino": j, "nivel": 1,
                               "dist": round(extra, 1), "tipo": "desviado", "bump": False,
                               "hora": hora, "heavy": es_heavy, "olat": la, "olon": lo,
                               "icao24": getattr(r, "icao24", "")})
                colocado = True
                break
            if not colocado:
                avx = {"la": la, "lo": lo, "d_dest": d_dest, "ola": ola, "olo": olo,
                       "o_icao": o_icao, "heavy": es_heavy, "icao24": getattr(r, "icao24", "")}
                (over_h if es_heavy else over_n).append(avx)

        # NIVELES 2+: el sobrante se propaga desde aeropuertos SATURADOS
        # fuentes = aeropuertos del nivel anterior que recibieron desvios y se llenaron.
        # el sobrante se recoloca de mas cercano a Madrid a mas lejano; los mas lejanos son
        # los que tienen mas probabilidad de quedarse sin sitio tras la cascada
        over_h.sort(key=lambda a: a["d_dest"])
        over_n.sort(key=lambda a: a["d_dest"])
        usados = set(recibio_h.keys())
        fuentes = {j for j in recibio_h if lleno(j)}
        nivel = 2
        while (len(over_h) + len(over_n)) > 0 and nivel <= max_nivel and fuentes:
            cand_recep = {}
            for s in fuentes:
                for dd, j in cercanos_ap(s):
                    if j == icao_A or j in usados:
                        continue
                    if j not in cand_recep or dd < cand_recep[j][0]:
                        cand_recep[j] = (dd, s)
            orden = sorted(cand_recep.keys(), key=lambda j: hav(la0, lo0, co[j][0], co[j][1]))
            nuevas_fuentes = set()
            for j in orden:
                if len(over_h) + len(over_n) == 0:
                    break
                room = min(free[j], park_libre_ini[j] - park_div[j])
                if room <= 0:
                    continue
                dgw, gw = cand_recep[j]
                olat, olon = co[gw]
                dist_hop = round(dgw, 1)
                puestos = 0
                if tipo_ap.get(j) == "large_airport" and len(over_h) > 0:
                    take = min(room, len(over_h))
                    for _ in range(take):
                        avp = over_h.pop(0)
                        fid = f"CX-{nid:04d}"; nid += 1
                        vuelos.append({"vuelo": fid, "origen": gw, "destino": j, "nivel": nivel,
                                       "dist": dist_hop, "tipo": "desviado", "bump": False,
                                       "hora": hora, "heavy": True, "olat": olat, "olon": olon,
                                       "icao24": avp.get("icao24", "")})
                        cascada_av.append({"la": avp["la"], "lo": avp["lo"], "hora": hora,
                                           "nivel": nivel, "icao24": avp.get("icao24", "")})
                    room -= take; puestos += take
                if len(over_n) > 0 and room > 0:
                    take = min(room, len(over_n))
                    for _ in range(take):
                        avp = over_n.pop(0)
                        fid = f"CX-{nid:04d}"; nid += 1
                        vuelos.append({"vuelo": fid, "origen": gw, "destino": j, "nivel": nivel,
                                       "dist": dist_hop, "tipo": "desviado", "bump": False,
                                       "hora": hora, "heavy": False, "olat": olat, "olon": olon,
                                       "icao24": avp.get("icao24", "")})
                        cascada_av.append({"la": avp["la"], "lo": avp["lo"], "hora": hora,
                                           "nivel": nivel, "icao24": avp.get("icao24", "")})
                    room -= take; puestos += take
                if puestos > 0:
                    free[j] -= puestos
                    park_div[j] += puestos
                    usados.add(j)
                    recibio_h[j] = recibio_h.get(j, 0) + puestos
                    circulos.setdefault(gw, (nivel - 1, hora))
                    circulos.setdefault(j, (nivel, hora))
                    if lleno(j):
                        nuevas_fuentes.add(j)
            fuentes = nuevas_fuentes
            nivel += 1

        for avx in over_h + over_n:
            la, lo, d_dest = avx["la"], avx["lo"], avx["d_dest"]
            ola, olo, o_icao, es_heavy = avx["ola"], avx["olo"], avx["o_icao"], avx["heavy"]
            orig_ok = pd.notna(ola) and pd.notna(olo)
            es_cerrado = (o_icao == icao_A)
            if orig_ok and not es_cerrado:
                d_orig = hav(la, lo, float(ola), float(olo))
                tot = d_orig + d_dest
                pct = (d_orig / tot) if tot > 0 else 1.0
                if pct <= frac_retorno:
                    fid = f"VR-{nid:04d}"; nid += 1
                    retornos.append({"vuelo": fid, "origen": o_icao, "hora": hora, "heavy": es_heavy,
                                     "lat": la, "lon": lo, "olat": float(ola), "olon": float(olo),
                                     "d_dest_km": round(d_dest), "d_orig_km": round(d_orig),
                                     "pct_recorrido": round(pct * 100), "icao24": avx.get("icao24", "")})
                    continue
            # sin alternativa: comprometido, sin origen conocido, o el origen es el aeropuerto cerrado
            fid = f"NA-{nid:04d}"; nid += 1
            no_reub.append({"vuelo": fid, "origen": (o_icao or icao_A), "tipo": "desviado", "hora": hora,
                            "heavy": es_heavy, "olat": la, "olon": lo,
                            "d_dest_km": round(d_dest), "orig_desconocido": (not orig_ok),
                            "orig_cerrado": es_cerrado, "icao24": avx.get("icao24", "")})

    for c in cascada_av:
        if cand1:
            hub = min(cand1, key=lambda t: hav(c["la"], c["lo"], co[t[1]][0], co[t[1]][1]))[1]
            c["glat"], c["glon"], c["puerta"] = co[hub][0], co[hub][1], hub
        else:
            c["glat"], c["glon"], c["puerta"] = c["la"], c["lo"], ""

    df_v = pd.DataFrame(vuelos)
    df_n = pd.DataFrame(no_reub)
    df_r = pd.DataFrame(retornos)
    df_casc = pd.DataFrame(cascada_av)
    return {
        "overflow_por_hora": overflow_por_hora, "llegadas_h": overflow_por_hora,
        "red_cap": int(cap[icao_A] * (1 - reduccion / 100)), "horas": H,
        "circulos": circulos, "vuelos": df_v, "noreub": df_n, "retornos": df_r, "cascada_av": df_casc,
        "icao": icao_A, "reduccion": reduccion, "seed": 1, "radio": radio,
        "aviones": av, "modo": "posicion", "ocupacion": ocup_used, "ocup_es_real": bool(arrivals_alt),
        "parking_total": park_tot, "parking_libre_ini": park_libre_ini, "occ_park": occ_park,
    }


# CABECERA + CAPACIDADES EDITABLES

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
if "sim_caps" not in st.session_state or len(st.session_state["sim_caps"]) != len(base):
    st.session_state["sim_caps"] = base[["ident", "name", "type", "cap_h",
                                         "latitude_deg", "longitude_deg", "cap_real",
                                         "parking", "parking_real"]].copy()
caps = st.session_state["sim_caps"]

with st.expander("Capacidades y parking de los aeropuertos — editable", expanded=False):
    st.caption("cap_h = llegadas/hora (pista). parking = puestos para desvios (apron). "
               "Espana/Portugal: leidos del AIP (cap_real / parking_real = True). "
               "Resto del mundo: estimados por tipo. Editable.")
    edit = st.data_editor(
        caps[["ident", "name", "type", "cap_h", "cap_real", "parking", "parking_real"]],
        use_container_width=True, height=300, hide_index=True,
        disabled=["ident", "name", "type", "cap_real", "parking_real"],
        column_config={
            "cap_h": st.column_config.NumberColumn("cap_h (lleg/h)", min_value=1, max_value=80, step=1),
            "parking": st.column_config.NumberColumn("parking (puestos)", min_value=1, max_value=400, step=1)})
    caps["cap_h"] = edit["cap_h"].values
    caps["parking"] = edit["parking"].values
    st.session_state["sim_caps"] = caps


# SIDEBAR

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
        horas = st.slider("Duracion del cierre (horas)", 1, 24, 1, 1,
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
    occ_park = 0.60
    if modo_datos == "Instantanea por posicion (Trino)":
        liberacion = 0.0
        usar_ocup_real = st.toggle("Ocupacion real de los alternativos (Trino)", value=True,
            help="ON: el hueco de PISTA de cada aeropuerto sale de sus llegadas reales de ese dia y hora. "
                 "OFF: lo pones tu con un % fijo igual para todos, para estudiar escenarios "
                 "hipoteticos (p. ej. 'y si estuvieran al 80%?').")
        if usar_ocup_real:
            occ = 0.60
            st.caption("Hueco de pista de cada alternativo = su capacidad menos sus llegadas reales de ese dia y hora.")
        else:
            occ = st.slider("Ocupacion supuesta de los alternativos (%)", 0, 95, 60, 5,
                            help="Todos los alternativos al mismo % de ocupacion de PISTA. "
                                 "Hueco libre = 100% - este valor.") / 100
            st.caption("Ocupacion de pista hipotetica e igual para todos. Para comparar escenarios "
                       "('y si en vez del trafico real estuvieran al X%?').")
        occ_park = st.slider("Ocupacion previa del parking (%)", 0, 95, 60, 5,
                             help="Cuanto de lleno esta ya el apron de cada aeropuerto con sus propios "
                                  "aviones. Puestos libres para desvios = parking x (100% - este valor). "
                                  "Los desviados se acumulan y no se liberan durante el cierre.") / 100
        st.caption("La ocupacion real del parking no se puede medir con ADS-B (un avion aparcado apaga el "
                   "transpondedor), asi que es un supuesto. Muevelo para estudiar escenarios.")
    else:
        usar_ocup_real = False
        occ = st.slider("Ocupacion previa de los demas (%)", 0, 95, 60, 5,
                        help="Hueco libre = 100% - este valor. El resto es trafico propio desplazable.") / 100
        liberacion = st.slider("Liberacion de plazas por hora (%)", 0, 50, 10, 5,
                               help="Cada hora reabren algunas plazas en los alternativos (aviones que "
                                    "despegan), hasta recuperar como mucho el hueco inicial. 0 = no se libera nada.")
    frac_ret = st.slider("Punto de no retorno (%)", 40, 80, 60, 5,
                         help="Solo para el modo por posicion. Si un avion que se queda sin alternativa "
                              "ha recorrido MENOS de este % de su trayecto, le sobra combustible y vuelve "
                              "a su aeropuerto de salida (solo recuento, sin CO2 extra). Si lo ha superado, "
                              "esta comprometido y se cuenta como sin alternativa.")
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
    else:
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
                                                        arrivals_alt=(arr_alt or None), occ_park=occ_park,
                                                        frac_retorno=frac_ret / 100.0)

res = st.session_state.get("simb")
if not res:
    st.info("Configura la incidencia en la barra lateral y pulsa **Simular**.")
    st.stop()

# DESLIZADOR TEMPORAL (filtra lo que se muestra, no recalcula)

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


co2 = float((dfv["dist"] * 16).sum()) if not dfv.empty else 0.0
desv_origen = sum(res["overflow_por_hora"][:hsel])
dfr_all = res.get("retornos", pd.DataFrame())
dfr = dfr_all[dfr_all["hora"] <= hsel] if (not dfr_all.empty) else dfr_all
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Vuelos desviados (origen)", f"{desv_origen:,}",
          help="Llegadas que el afectado no puede absorber, acumuladas hasta esta hora")
c2.metric("Vuelos movidos (total)", f"{len(dfv):,}",
          help="Total recolocado: nivel 1 (aviones reales) mas los niveles siguientes de la cascada")
c3.metric("Vuelta a origen", f"{len(dfr):,}",
          help="Aviones que no encontraron alternativa pero no habian pasado su punto de no retorno: "
               "vuelven a su aeropuerto de salida. Solo recuento, sin CO2 extra (hacen los mismos km).")
c4.metric("Sin alternativa", f"{len(dfn):,}",
          delta=f"{int(dfv['nivel'].max()) if not dfv.empty else 0} niveles", delta_color="off",
          help="Aviones comprometidos (pasado el punto de no retorno) que ademas no caben en ningun "
               "aeropuerto cercano. Es el caso critico de verdad.")
c5.metric("CO₂ extra", f"{co2/1000:.1f} t",
          help=f"≈ {co2/21.77:,.0f} arboles/año · {co2/0.21:,.0f} km en coche")
if not dfn.empty and "orig_desconocido" in dfn.columns:
    n_desc = int(dfn["orig_desconocido"].sum())
    if n_desc:
        st.caption(f"⚠️ De los {len(dfn)} sin alternativa, {n_desc} no tienen aeropuerto de salida conocido "
                   f"en Trino, por lo que no se pudo evaluar si podrian volver a casa (se cuentan como sin alternativa).")

# MAPA

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
            if es_pos and tiene_o:
                olat, olon = r["olat"], r["olon"]
            elif r["origen"] in caps_i.index:
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
        pk_ini = res.get("parking_libre_ini") or {}
        pk_tot = res.get("parking_total") or {}
        hov = []
        for _, r in sub.iterrows():
            ic = r["destino"]
            nom = caps_i.loc[ic, "name"]
            cap_ap = int(caps_i.loc[ic, "cap_h"])
            extra = f" ({int(r['pesados'])} pesados)" if r["pesados"] else ""
            d_here = int(r["vuelos"])
            sufijo_h = f" en {hsel}h" if hsel > 1 else ""
            if pk_ini:
                libre_pk = int(pk_ini.get(ic, 0))
                tot_pk = int(pk_tot.get(ic, libre_pk))
                pctp = min(100, round(d_here / libre_pk * 100)) if libre_pk else 100
                hov.append(f"<b>{nom}</b> ({ic}) · Nivel {niv}<br>"
                           f"Capacidad de pista: {cap_ap} llegadas/h<br>"
                           f"Parking: {tot_pk} puestos · {libre_pk} libres para desvios<br>"
                           f"Recibe {d_here} desviado(s){sufijo_h}{extra} → {pctp}% del parking libre")
            else:
                linea_ocup = ""
                if ocup:
                    propias = sum(int(ocup.get(ic, {}).get(k, 0)) for k in range(hsel))
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

dfc = res.get("cascada_av")
if ver_lineas and isinstance(dfc, pd.DataFrame) and not dfc.empty:
    if "hora" in dfc.columns:
        dfc = dfc[dfc["hora"] <= hsel]
    clat, clon = [], []
    for _, r in dfc.iterrows():
        clat += [float(r["la"]), float(r["glat"]), None]
        clon += [float(r["lo"]), float(r["glon"]), None]
    if clat:
        fig.add_trace(go.Scattermap(
            lat=clat, lon=clon, mode="lines",
            line=dict(width=1, color="#00E5FF"), opacity=0.30,
            hoverinfo="none", showlegend=False, name="Entra en la cascada"))

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
        if r.get("orig_desconocido"):
            nota = " · origen desconocido"
            dkm = ""
        elif r.get("orig_cerrado"):
            nota = f" · su origen es {res['icao']} (cerrado), no puede volver"
            dkm = ""
        else:
            nota = f" · desde {r['origen']}"
            dkm = f"<br>comprometido, a {int(r['d_dest_km'])} km de {res['icao']}" if pd.notna(r.get("d_dest_km")) else ""
        txt.append(f"Vuelo {r['vuelo']} · SIN ALTERNATIVA{nota}{dkm}")
    if jlat:
        fig.add_trace(go.Scattermap(
            lat=jlat, lon=jlon, mode="markers",
            marker=go.scattermap.Marker(size=9, color="#FFFFFF", opacity=1.0),
            text=txt, hoverinfo="text", name=f"Sin alternativa ({len(jlat)})"))

dfr_map = res.get("retornos", pd.DataFrame())
dfr_map = dfr_map[dfr_map["hora"] <= hsel] if (not dfr_map.empty) else dfr_map
if ver_noreub and not dfr_map.empty:
    if ver_lineas:
        vlat, vlon = [], []
        for _, r in dfr_map.iterrows():
            vlat += [float(r["lat"]), float(r["olat"]), None]
            vlon += [float(r["lon"]), float(r["olon"]), None]
        if vlat:
            fig.add_trace(go.Scattermap(
                lat=vlat, lon=vlon, mode="lines",
                line=dict(width=1.5, color="#FF80AB"), opacity=0.55,
                hoverinfo="none", showlegend=False, name="Rutas de vuelta a origen"))
    rlat, rlon, rtxt = [], [], []
    olat_o, olon_o, otxt = [], [], []
    for _, r in dfr_map.iterrows():
        rlat.append(float(r["lat"])); rlon.append(float(r["lon"]))
        rtxt.append(f"Vuelo {r['vuelo']} · VUELVE A ORIGEN ({r['origen']})<br>"
                    f"{int(r['pct_recorrido'])}% del trayecto recorrido<br>"
                    f"a {int(r['d_dest_km'])} km de {res['icao']} · a {int(r['d_orig_km'])} km del origen")
        olat_o.append(float(r["olat"])); olon_o.append(float(r["olon"]))
        otxt.append(f"{r['origen']} · aqui vuelve el vuelo {r['vuelo']}")
    if rlat:
        fig.add_trace(go.Scattermap(
            lat=olat_o, lon=olon_o, mode="markers",
            marker=go.scattermap.Marker(size=8, color="#FF80AB", opacity=0.45),
            text=otxt, hoverinfo="text", showlegend=False, name="Origen (vuelta)"))
        fig.add_trace(go.Scattermap(
            lat=rlat, lon=rlon, mode="markers",
            marker=go.scattermap.Marker(size=10, color="#FF80AB", opacity=1.0),
            text=rtxt, hoverinfo="text", name=f"Vuelta a origen ({len(rlat)})"))

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


# EVOLUCION POR HORAS

if H > 1:
    movidos_h = dfv_all.groupby("hora").size() if not dfv_all.empty else pd.Series(dtype=int)
    nore_h = dfn_all.groupby("hora").size() if not dfn_all.empty else pd.Series(dtype=int)
    evo = pd.DataFrame({"hora": range(1, H + 1)}).set_index("hora")
    evo["Vuelos movidos"] = movidos_h.reindex(evo.index, fill_value=0)
    evo["Sin alternativa"] = nore_h.reindex(evo.index, fill_value=0)
    st.markdown("### Evolucion por hora de cierre")
    st.bar_chart(evo, color=["#00CC66", "#FF3B3B"])

# AEROPUERTOS QUE RECIBEN - HORA A HORA

if not dfv.empty:
    ocup = res.get("ocupacion") or {}
    pk_ini = res.get("parking_libre_ini") or {}
    per = (dfv.groupby(["destino", "hora"])
           .agg(desv=("vuelo", "size"), pes=("heavy", "sum"), niv=("nivel", "min"))
           .reset_index())
    per = per[per["destino"].isin(caps_i.index)]
    cum_div = {}
    for ic_ in per["destino"].unique():
        run = 0
        cum_div[ic_] = {}
        for _, rr in per[per["destino"] == ic_].sort_values("hora").iterrows():
            run += int(rr["desv"])
            cum_div[ic_][int(rr["hora"])] = run
    filas = []
    for _, r in per.iterrows():
        ic = r["destino"]
        h = int(r["hora"])
        D = int(r["desv"])
        cap_ap = int(caps_i.loc[ic, "cap_h"])
        fila = {"Hora": h, "Aeropuerto": caps_i.loc[ic, "name"], "ICAO": ic, "Cap/h": cap_ap}
        if ocup:
            R = int(ocup.get(ic, {}).get(h - 1, 0))
            fila["Propias"] = R
            fila["Libres"] = max(0, cap_ap - R)
            fila["Desviados"] = D
            fila["% lleno pista"] = min(100, round((R + D) / cap_ap * 100)) if cap_ap else 0
        else:
            fila["Desviados"] = D
        if pk_ini:
            libre_pk = int(pk_ini.get(ic, 0))
            aparcados = int(cum_div.get(ic, {}).get(h, D))
            fila["Parking libre"] = libre_pk
            fila["Aparcados (acum)"] = aparcados
            fila["% parking"] = min(100, round(aparcados / libre_pk * 100)) if libre_pk else 100
        fila["Pesados"] = int(r["pes"])
        fila["Nivel"] = int(r["niv"])
        filas.append(fila)
    resu = pd.DataFrame(filas).sort_values(["Hora", "Nivel", "Desviados"], ascending=[True, True, False])
    st.markdown("### Aeropuertos que reciben desvios, hora a hora" if H > 1
                else "### Aeropuertos que reciben desvios")
    if pk_ini:
        _f = "llegadas reales" if res.get("ocup_es_real", True) else "ocupacion supuesta"
        st.caption(f"Pista (por hora): capacidad, {_f} de esa hora y huecos. Parking (acumulado): puestos "
                   "libres para desvios, aparcados acumulados y % de parking ocupado. El parking NO se "
                   "vacia durante el cierre, por eso sube hora a hora hasta que el aeropuerto se topa y "
                   "los siguientes desvios saltan a otro.")
    else:
        _fuente = "llegadas reales" if res.get("ocup_es_real", True) else "ocupacion supuesta"
        st.caption(f"Por aeropuerto y hora: capacidad, {_fuente} de esa hora, huecos libres, vuelos "
                   "recibidos y como queda de lleno. El hueco se recalcula en cada hora del cierre.")
    st.dataframe(resu, use_container_width=True, height=340, hide_index=True)


# TABLA POR VUELO + CSV

if not dfv.empty:
    tab = dfv.merge(caps[["ident", "name"]], left_on="destino", right_on="ident", how="left")
    if "icao24" not in tab.columns:
        tab["icao24"] = ""
    tab["tipo"] = tab["tipo"].map({"desviado": "Desviado (afectado)", "desplazado": "Desplazado"})
    tab["co2_kg"] = (tab["dist"] * 16).round(0).astype(int)
    tab["avion"] = tab["heavy"].map({True: "Pesado", False: "Normal"})
    tab = tab.rename(columns={"vuelo": "Vuelo", "hora": "Hora", "origen": "Desde",
                              "destino": "Aterriza en", "name": "Aeropuerto destino",
                              "nivel": "Nivel", "dist": "Dist. extra (km)", "co2_kg": "CO2 (kg)",
                              "tipo": "Tipo", "avion": "Avion", "icao24": "ID avion (icao24)"})
    cols = ["Hora", "Vuelo", "ID avion (icao24)", "Tipo", "Avion", "Desde", "Aterriza en",
            "Aeropuerto destino", "Nivel", "Dist. extra (km)", "CO2 (kg)"]
    st.markdown("### Vuelos redirigidos")
    st.dataframe(tab[cols].sort_values(["Hora", "Nivel", "Vuelo"]), use_container_width=True, height=320)
    st.download_button(
        "⬇️ Descargar vuelos (CSV)",
        data=tab[cols].to_csv(index=False),
        file_name=f"cascada_{res['icao']}_{res['reduccion']}pct_{H}h.csv",
        mime="text/csv")