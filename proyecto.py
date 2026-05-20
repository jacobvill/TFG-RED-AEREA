"""
pages/4_Proyecto.py
TFG: Simulación y Análisis del Impacto Operativo de la Red Aérea Global
Jacob Altenburger Villar · UAX 2026

Página principal del proyecto:
- Login con credenciales OpenSky (OAuth2) y Trino
- Tráfico en tiempo real via REST API con trayectoria al hacer click
- Análisis histórico via Trino con líneas de destino reales
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta, timezone
from trino.dbapi import connect
from trino.auth import OAuth2Authentication



def sanitize(series):
    """Elimina caracteres que rompen la serialización JSON de Plotly."""
    return (series.fillna("N/A")
                  .astype(str)
                  .str.replace("\\", "/", regex=False)
                  .str.replace('"', "'", regex=False)
                  .str.replace('\n', ' ', regex=False)
                  .str.replace('\r', ' ', regex=False)
                  .str.replace('\t', ' ', regex=False)
                  .str.strip())



st.set_page_config(
    page_title="TFG – Proyecto Red Aérea",
    page_icon="✈",
    layout="wide"
)

# ================================================================
# CONSTANTES
# ================================================================
TOKEN_URL = ("https://auth.opensky-network.org/auth/realms/"
             "opensky-network/protocol/openid-connect/token")

REGIONES = {
    "Europa":       (29.0, 81.2, -25.0,  45.0),
    "Norteamérica": ( 7.4, 83.1,-176.6, -52.0),
    "Sudamérica":   (-55.0,12.4,-109.4, -32.4),
    "Asia":         (-12.2,80.8,  26.0, 180.0),
    "África":       (-35.0,37.2, -25.0,  63.4),
    "Mundial":      (-90.0,90.0,-180.0, 180.0),
}

POS_SOURCE = {0:"ADS-B",1:"ASTERIX",2:"MLAT",3:"FLARM"}
BBOXES_CONT = {
    "EU":(-25.0,29.0,45.0,81.2), "NA":(-176.6,7.4,-52.0,83.1),
    "SA":(-109.4,-55.0,-32.4,12.4), "AS":(26.0,-12.2,180.0,80.8),
    "AF":(-25.1,-34.8,63.4,37.2),
}

# ================================================================
# AEROPUERTOS (cacheado)
# ================================================================
@st.cache_data
def cargar_aeropuertos():
    df = pd.read_csv("airports.csv")
    df = df[df["type"].isin(["large_airport","medium_airport","small_airport"])].copy()
    df = df.dropna(subset=["latitude_deg","longitude_deg"])
    df["continent"] = df["continent"].fillna("NA")
    return df

df_airports = cargar_aeropuertos()

# ================================================================
# FUNCIONES DE AUTENTICACIÓN
# ================================================================
def obtener_token(client_id, client_secret):
    """Obtiene un Bearer token via OAuth2 client credentials."""
    try:
        r = requests.post(TOKEN_URL, data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }, timeout=15)
        if r.status_code == 200:
            data = r.json()
            token      = data["access_token"]
            expires_in = data.get("expires_in", 1800)
            expires_at = datetime.now() + timedelta(seconds=expires_in - 60)
            return token, expires_at, None
        return None, None, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, None, str(e)


def get_token_valido():
    """Devuelve el token actual, renovándolo si está caducado."""
    token      = st.session_state.get("proy_token")
    expires_at = st.session_state.get("proy_token_exp")
    if token and expires_at and datetime.now() < expires_at:
        return token, None
    # Renovar
    cid = st.session_state.get("proy_client_id","")
    csc = st.session_state.get("proy_client_secret","")
    token, exp, err = obtener_token(cid, csc)
    if err: return None, err
    st.session_state["proy_token"]     = token
    st.session_state["proy_token_exp"] = exp
    return token, None


def get_trino():
    """Conexión Trino reutilizada."""
    user = st.session_state.get("proy_trino_user","")
    if "proy_trino_conn" not in st.session_state:
        st.session_state.proy_trino_conn = connect(
            host="trino.opensky-network.org", port=443,
            user=user, auth=OAuth2Authentication(),
            http_scheme="https", catalog="minio", schema="osky",
            request_timeout=120.0
        )
    return st.session_state.proy_trino_conn

# ================================================================
# FUNCIONES DE DATOS
# ================================================================
def fetch_live(bbox, solo_comercial=False):
    """
    Consulta /states/all con extended=1.
    Devuelve DataFrame con TODOS los campos disponibles de OpenSky.
    """
    token, err = get_token_valido()
    if err: return pd.DataFrame(), err

    lat_min, lat_max, lon_min, lon_max = bbox
    try:
        r = requests.get(
            "https://opensky-network.org/api/states/all",
            headers={"Authorization": f"Bearer {token}"},
            params={"lamin":lat_min,"lamax":lat_max,
                    "lomin":lon_min,"lomax":lon_max,
                    "extended":1},
            timeout=20
        )
        cr = r.headers.get("X-Rate-Limit-Remaining","?")
        st.session_state["proy_creditos"] = cr

        if r.status_code == 200:
            states = r.json().get("states", [])
            if not states:
                return pd.DataFrame(), None
            cols = ["icao24","callsign","origin_country","time_position",
                    "last_contact","longitude","latitude","baro_altitude",
                    "on_ground","velocity","true_track","vertical_rate",
                    "sensors","geo_altitude","squawk","spi",
                    "position_source","category"]
            df = pd.DataFrame(states, columns=cols)
            df = df[df["on_ground"]==False].dropna(subset=["latitude","longitude"]).copy()
            # Limpiar y convertir
            df["callsign"]       = df["callsign"].fillna("").str.strip()
            df["velocity"]       = pd.to_numeric(df["velocity"],      errors="coerce").fillna(0)
            df["baro_altitude"]  = pd.to_numeric(df["baro_altitude"], errors="coerce").fillna(0)
            df["geo_altitude"]   = pd.to_numeric(df["geo_altitude"],  errors="coerce").fillna(0)
            df["true_track"]     = pd.to_numeric(df["true_track"],    errors="coerce").fillna(0)
            df["vertical_rate"]  = pd.to_numeric(df["vertical_rate"], errors="coerce").fillna(0)
            df["position_source"]= pd.to_numeric(df["position_source"],errors="coerce").fillna(0).astype(int)
            df["category"]       = pd.to_numeric(df["category"],      errors="coerce").fillna(0).astype(int)
            df["pos_src_str"]    = df["position_source"].map(POS_SOURCE).fillna("Desconocido")
            return df, None
        elif r.status_code == 401: return pd.DataFrame(), " Token inválido. Vuelve a iniciar sesión."
        elif r.status_code == 429:
            s = r.headers.get("X-Rate-Limit-Retry-After-Seconds","?")
            return pd.DataFrame(), f" Sin créditos. Espera {s}s."
        else: return pd.DataFrame(), f"HTTP {r.status_code}"
    except Exception as e:
        return pd.DataFrame(), str(e)


def fetch_track(icao24):
    """
    Obtiene la trayectoria completa del vuelo actual via /tracks/all.
    Solo se llama al seleccionar un avión concreto.
    Coste: 4 créditos.
    """
    token, err = get_token_valido()
    if err: return None, err
    try:
        r = requests.get(
            "https://opensky-network.org/api/tracks/all",
            headers={"Authorization": f"Bearer {token}"},
            params={"icao24": icao24, "time": 0},
            timeout=20
        )
        if r.status_code == 200:
            return r.json(), None
        elif r.status_code == 404:
            return None, "No hay trayectoria disponible para este vuelo ahora mismo."
        else:
            return None, f"HTTP {r.status_code}"
    except Exception as e:
        return None, str(e)


def fetch_historico(fecha, hora, minuto, continente):
    """
    Consulta state_vectors_data4 para la hora indicada.
    Cruza con flights_data4 para obtener origen/destino reales.
    """
    dt_utc = datetime(fecha.year,fecha.month,fecha.day,hora,minuto,0,tzinfo=timezone.utc)
    ts     = int(dt_utc.timestamp())
    ts_h   = ts - (ts % 3600)

    bbox = BBOXES_CONT.get(continente, (-25.0,29.0,45.0,81.2))
    lnmin,ltmin,lnmax,ltmax = bbox

    conn = get_trino()

    # 1. Posiciones en ese momento
    q_sv = f"""
        SELECT icao24,
               MAX_BY(callsign,     time) AS callsign,
               MAX_BY(lat,          time) AS lat,
               MAX_BY(lon,          time) AS lon,
               MAX_BY(velocity,     time) AS velocity,
               MAX_BY(heading,      time) AS heading,
               MAX_BY(baroaltitude, time) AS baroaltitude,
               MAX_BY(vertrate,     time) AS vertrate
        FROM state_vectors_data4
        WHERE hour     = {ts_h}
          AND time     BETWEEN {ts} AND {ts}+60
          AND onground = false
          AND lat      BETWEEN {ltmin-2} AND {ltmax+2}
          AND lon      BETWEEN {lnmin-2} AND {lnmax+2}
          AND lat IS NOT NULL AND lon IS NOT NULL
        GROUP BY icao24
    """
    cur=conn.cursor(); cur.execute(q_sv)
    rows=cur.fetchall(); cols_sv=[d[0] for d in cur.description]
    df_sv = pd.DataFrame(rows, columns=cols_sv)

    if df_sv.empty:
        return df_sv, 0

    # 2. Rutas del día (flights_data4) — solo los icao24 que tenemos
    dt_day = datetime(fecha.year,fecha.month,fecha.day,0,0,0,tzinfo=timezone.utc)
    ts_day = int(dt_day.timestamp())

    icaos_str = "','".join(df_sv["icao24"].tolist())
    q_fl = f"""
            SELECT icao24,
                   TRIM(callsign)      AS callsign_fl,
                   estdepartureairport AS origen,
                   estarrivalairport   AS destino,
                   firstseen,
                   lastseen
            FROM flights_data4
            WHERE day     = {ts_day}
              AND icao24  IN ('{icaos_str}')
              AND icao24  IS NOT NULL
              AND firstseen <= {ts + 3600}
              AND lastseen  >= {ts - 3600}
        """
    cur.execute(q_fl)
    rows = cur.fetchall();
    cols_fl = [d[0] for d in cur.description]
    df_fl = pd.DataFrame(rows, columns=cols_fl)

    n_con_destino = 0
    if not df_fl.empty:
        # Si un avión tiene varios vuelos activos en esa ventana,
        # quedarse con el que empezó más cerca del timestamp
        df_fl["diff"] = abs(df_fl["firstseen"] - ts)
        df_fl = df_fl.sort_values("diff").drop_duplicates("icao24", keep="first")
        df_fl["callsign_fl"] = df_fl["callsign_fl"].fillna("").str.strip()
        df_sv = df_sv.merge(
            df_fl[["icao24", "origen", "destino"]], on="icao24", how="left"
        )
        n_con_destino = df_sv["destino"].notna().sum()

    df_sv["callsign"] = df_sv["callsign"].fillna("").str.strip()
    for col in ["velocity","baroaltitude","heading","vertrate"]:
        df_sv[col] = pd.to_numeric(df_sv[col], errors="coerce").fillna(0)

    return df_sv, n_con_destino


def fetch_trayectoria_trino(icao24, fecha, hora):
    """Trayectoria histórica de un avión ±2h via Trino."""
    dt_utc  = datetime(fecha.year,fecha.month,fecha.day,hora,0,0,tzinfo=timezone.utc)
    ts_c    = int(dt_utc.timestamp())
    ts_s, ts_e = ts_c-2*3600, ts_c+2*3600
    hours   = set()
    t = ts_s-(ts_s%3600)
    while t <= ts_e: hours.add(t); t+=3600
    h_str = ",".join(str(h) for h in sorted(hours))
    conn  = get_trino()
    q = f"""
        SELECT time, lat, lon, baroaltitude, velocity, heading
        FROM state_vectors_data4
        WHERE hour IN ({h_str})
          AND time BETWEEN {ts_s} AND {ts_e}
          AND icao24='{icao24}'
          AND lat IS NOT NULL AND lon IS NOT NULL
        ORDER BY time
    """
    cur=conn.cursor(); cur.execute(q)
    rows=cur.fetchall(); cols=[d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


# ================================================================
# PANTALLA DE LOGIN
# ================================================================
if not st.session_state.get("proy_logged_in"):
    st.markdown("""
    <style>
    .login-box {
        max-width: 460px;
        margin: 80px auto;
        padding: 40px;
        background: rgba(255,255,255,0.04);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)
        st.markdown("## ✈️ TFG — Red Aérea Global")
        st.markdown("**Jacob Altenburger Villar · UAX 2026**")
        st.divider()
        st.markdown("### 🔑 Iniciar sesión")
        st.caption("Introduce tus credenciales de OpenSky Network")

        client_id     = st.text_input("clientId",     value="jaltevil@myuax.com-api-client")
        client_secret = st.text_input("clientSecret", type="password", placeholder="••••••••")
        trino_user    = st.text_input("Email Trino",  value="jaltevil@myuax.com",
                                       help="El mismo email de tu cuenta OpenSky")
        btn_login = st.button("Entrar →", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    if btn_login:
        if not client_id or not client_secret or not trino_user:
            st.error("Rellena todos los campos.")
        else:
            with st.spinner(" Autenticando..."):
                token, exp, err = obtener_token(client_id, client_secret)
            if err:
                st.error(f"Error de autenticación: {err}")
            else:
                # Guardar credenciales y token
                st.session_state.update({
                    "proy_logged_in":    True,
                    "proy_client_id":    client_id,
                    "proy_client_secret": client_secret,
                    "proy_trino_user":   trino_user.lower(),
                    "proy_token":        token,
                    "proy_token_exp":    exp,
                    "proy_modo":         "live",
                })
                # Cargar datos live inmediatamente
                with st.spinner("📡 Cargando tráfico aéreo en tiempo real..."):
                    bbox = REGIONES["Europa"]
                    df_live, err2 = fetch_live(bbox)
                    if err2:
                        st.error(err2)
                    else:
                        st.session_state.update({
                            "proy_df":       df_live,
                            "proy_region":   "Europa",
                            "proy_ts":       datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                        })
                st.rerun()
    st.stop()

# ================================================================
# INTERFAZ PRINCIPAL (usuario autenticado)
# ================================================================
modo     = st.session_state.get("proy_modo", "live")
df       = st.session_state.get("proy_df",   pd.DataFrame())
region   = st.session_state.get("proy_region","Europa")
ts_label = st.session_state.get("proy_ts","")
creditos = st.session_state.get("proy_creditos","?")

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"**✈ TFG Red Aérea** · {st.session_state.get('proy_trino_user','')}")
    st.caption(f" Créditos restantes: {creditos}")

    if st.button(" Cerrar sesión", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("proy_"):
                del st.session_state[k]
        st.rerun()

    st.divider()

    # Modo
    nuevo_modo = st.radio("📡 Modo:", ["live","historico"],
                          index=0 if modo=="live" else 1,
                          format_func=lambda x: " En vivo (API)" if x=="live" else " Día concreto (Trino)")
    if nuevo_modo != modo:
        st.session_state["proy_modo"] = nuevo_modo
        st.session_state["proy_df"]   = pd.DataFrame()
        st.session_state["proy_track"]= None
        st.session_state["proy_sel_icao"] = None
        st.rerun()

    st.divider()

    if modo == "live":
        region = st.selectbox(" Región:", list(REGIONES.keys()),
                               index=list(REGIONES.keys()).index(region))
        solo_com = st.toggle("Solo vuelos comerciales", value=False)
        btn_act  = st.button(" Actualizar", use_container_width=True, type="primary")

    else:  # histórico
        fecha_h  = st.date_input(" Día:", datetime(2024,1,16))
        hora_h   = st.selectbox(" Hora (UTC):", list(range(24)), index=12)
        min_h    = st.selectbox("Minuto:", list(range(0,60,5)), index=0)
        cont_h   = st.selectbox("Continente:", list(BBOXES_CONT.keys()),
                                 format_func=lambda x: {"EU":"Europa","NA":"Norteamérica",
                                 "SA":"Sudamérica","AS":"Asia","AF":"África"}[x])

        # Filtro destino opcional
        df_dest_opts = df_airports[
            df_airports["type"].isin(["large_airport","medium_airport"])
        ].sort_values("name")
        dest_opts = ["— Sin filtro (todos) —"] + df_dest_opts["name"].tolist()
        dest_sel  = st.selectbox(" Filtrar por destino:", dest_opts)
        dest_icao = None
        if dest_sel != "— Sin filtro (todos) —":
            dest_icao = df_dest_opts[df_dest_opts["name"]==dest_sel]["ident"].values[0]
            st.caption(f" Solo se muestran vuelos con destino confirmado `{dest_icao}`. "
                       f"Los que no tienen destino en la BD no aparecerán.")

        btn_hist = st.button(" Consultar Trino", use_container_width=True, type="primary")

    st.divider()
    # Selector de avión para trayectoria
    st.markdown("** Trayectoria**")
    st.caption("Selecciona un avión para ver su ruta")

    sel_icao = st.session_state.get("proy_sel_icao","")
    if not df.empty:
        opciones = ["— Ninguno —"]
        if "callsign" in df.columns:
            df["_label"] = df["callsign"].str.strip().replace("","?") + " (" + df["icao24"] + ")"
        else:
            df["_label"] = df["icao24"]
        opciones += df["_label"].tolist()
        sel_label = st.selectbox("Avión:", opciones, key="sel_avion_proy")
        if sel_label != "— Ninguno —":
            nuevo_icao = sel_label.split("(")[-1].rstrip(")")
            if nuevo_icao != sel_icao:
                st.session_state["proy_sel_icao"] = nuevo_icao
                st.session_state["proy_track"]    = None
        else:
            st.session_state["proy_sel_icao"] = None
            st.session_state["proy_track"]    = None

        if st.session_state.get("proy_sel_icao"):
            btn_tray = st.button(" Cargar trayectoria", use_container_width=True)
            if btn_tray:
                with st.spinner("Cargando trayectoria..."):
                    if modo == "live":
                        track, err_t = fetch_track(st.session_state["proy_sel_icao"])
                        if err_t:
                            st.error(err_t)
                        else:
                            st.session_state["proy_track"] = {"tipo":"live","data":track}
                    else:
                        df_tray = fetch_trayectoria_trino(
                            st.session_state["proy_sel_icao"], fecha_h, hora_h
                        )
                        st.session_state["proy_track"] = {"tipo":"historico","data":df_tray}

# ================================================================
# ACCIONES DE CARGA DE DATOS
# ================================================================
if modo=="live" and "btn_act" in dir() and btn_act:
    with st.spinner(" Actualizando..."):
        df_new, err = fetch_live(REGIONES[region])
    if err: st.error(err)
    else:
        st.session_state.update({
            "proy_df":     df_new,
            "proy_region": region,
            "proy_ts":     datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            "proy_track":  None,
            "proy_sel_icao": None,
        })
        df = df_new
        st.rerun()

if modo=="historico" and "btn_hist" in dir() and btn_hist:
    with st.spinner(" Consultando Trino (puede tardar 30s)..."):
        try:
            df_hist, n_dest = fetch_historico(fecha_h, hora_h, min_h, cont_h)
            if dest_icao:
                # Filtrar solo los que tienen ese destino confirmado
                if "destino" in df_hist.columns:
                    df_hist = df_hist[df_hist["destino"]==dest_icao].reset_index(drop=True)
                    n_dest  = len(df_hist)
            st.session_state.update({
                "proy_df":       df_hist,
                "proy_ts":       f"{fecha_h} {hora_h:02d}:{min_h:02d} UTC",
                "proy_n_dest":   n_dest,
                "proy_track":    None,
                "proy_sel_icao": None,
            })
            df = df_hist
        except Exception as e:
            st.error(f"Error Trino: {e}")
            if "proy_trino_conn" in st.session_state:
                del st.session_state["proy_trino_conn"]

# ================================================================
# CONSTRUCCIÓN DEL MAPA
# ================================================================
n_dest_hist = st.session_state.get("proy_n_dest", 0)

# Banner mínimo
if not df.empty:
    if modo=="live":
        st.caption(f" **{len(df):,}** aviones · Región: **{region}** · {ts_label} · 💳 {creditos} créditos")
    else:
        pct = round(n_dest_hist/len(df)*100) if len(df)>0 else 0
        st.caption(f" **{len(df):,}** aviones · {ts_label} · "
                   f" {n_dest_hist} con destino conocido ({pct}%) · sin destino: sin línea")

fig = go.Figure()

# ── CAPA 1: AEROPUERTOS ──────────────────────────────────────────
if modo=="live":
    bb = REGIONES[region]
    df_av = df_airports[
        (df_airports["latitude_deg"]  >= bb[0]) & (df_airports["latitude_deg"]  <= bb[1]) &
        (df_airports["longitude_deg"] >= bb[2]) & (df_airports["longitude_deg"] <= bb[3])
    ]
else:
    lnmin,ltmin,lnmax,ltmax = BBOXES_CONT.get(cont_h if "cont_h" in dir() else "EU",
                                               (-25.0,29.0,45.0,81.2))
    df_av = df_airports[
        (df_airports["latitude_deg"]  >= ltmin) & (df_airports["latitude_deg"]  <= ltmax) &
        (df_airports["longitude_deg"] >= lnmin) & (df_airports["longitude_deg"] <= lnmax)
    ]

for tipo, color, size in [("large_airport","#FF4B4B",14),
                            ("medium_airport","#1C83E1",9),
                            ("small_airport","#00AA44",5)]:
    df_t = df_av[df_av["type"]==tipo]
    if df_t.empty: continue

    hover_ap = (
        "<b>" + df_t["name"].fillna("") + "</b><br>"
        + "ICAO: "      + df_t["ident"].fillna("") + "<br>"
        + "IATA: "      + df_t["iata_code"].fillna("N/A") + "<br>"
        + "Ciudad: "    + df_t["municipality"].fillna("N/A") + "<br>"
        + "Pais: "      + df_t["iso_country"].fillna("N/A") + "<br>"
        + "Elevacion: " + df_t["elevation_ft"].fillna(0).astype(int).astype(str) + " ft<br>"
        + "Tipo: "      + df_t["type"].str.replace("_"," ")
    )

    fig.add_trace(go.Scattermap(
        lat=df_t["latitude_deg"], lon=df_t["longitude_deg"],
        mode="markers",
        name=tipo.replace("_"," ").title(),
        marker=go.scattermap.Marker(size=size, color=color, opacity=0.7),
        text=hover_ap, hoverinfo="text"
    ))
# ── CAPA 2: LÍNEAS DE DESTINO / HEADING ─────────────────────────
if not df.empty:
    aero_pos = df_airports.set_index("ident")[["latitude_deg","longitude_deg"]].to_dict("index")
    lats_l, lons_l = [], []

    if modo=="historico" and "destino" in df.columns:
        # Líneas reales origen → destino para los que tienen destino en BD
        df_con_dest = df[df["destino"].notna()].copy()
        for _, row in df_con_dest.iterrows():
            dest_info = aero_pos.get(row["destino"])
            if dest_info:
                lats_l += [row["lat"], dest_info["latitude_deg"], None]
                lons_l += [row["lon"], dest_info["longitude_deg"], None]
    if lats_l:
        fig.add_trace(go.Scattermap(
            lat=lats_l, lon=lons_l, mode="lines",
            line=dict(width=1, color="rgba(255,255,255,0.25)"),
            hoverinfo="none", showlegend=False, name="Rutas"
        ))
        # Línea destacada del avión seleccionado
        sel_icao_highlight = st.session_state.get("proy_sel_icao")
        if sel_icao_highlight and modo == "historico" and "destino" in df.columns:
            fila = df[df["icao24"] == sel_icao_highlight]
            if not fila.empty and pd.notna(fila.iloc[0].get("destino")):
                dest_info = aero_pos.get(fila.iloc[0]["destino"])
                if dest_info:
                    fig.add_trace(go.Scattermap(
                        lat=[fila.iloc[0]["lat"], dest_info["latitude_deg"]],
                        lon=[fila.iloc[0]["lon"], dest_info["longitude_deg"]],
                        mode="lines",
                        line=dict(width=4, color="cyan"),
                        hoverinfo="none", showlegend=False,
                        name="Ruta seleccionada"
                    ))
# ── CAPA 3: AVIONES ──────────────────────────────────────────────
if not df.empty:
    if modo=="live":
        lat_col, lon_col = "latitude","longitude"
        vel_col, alt_col = "velocity","baro_altitude"
        hdg_col          = "true_track"

        vel_kmh  = (df[vel_col]*3.6).round(1).astype(str)
        alt_ft   = (df[alt_col]*3.281).round(0).astype(int).astype(str)
        geo_ft   = (df["geo_altitude"]*3.281).round(0).astype(int).astype(str)
        vrate_ms = df["vertical_rate"].round(1).astype(str)
        vrate_fpm= (df["vertical_rate"]*196.85).round(0).astype(int).astype(str)

        hover = (
            "✈ <b>" + df["callsign"].replace("","N/A") + "</b>"
            + " · " + df["icao24"] + "<br>"
            + " País de origen: " + df["origin_country"] + "<br>"
            + " Fuente señal: "   + df["pos_src_str"] + "<br>"
            + "⚡ Squawk: "         + df["squawk"].fillna("N/A") + "<br>"
            + "─────────────────<br>"
            + " Lat: " + df["latitude"].round(4).astype(str)
            + " · Lon: " + df["longitude"].round(4).astype(str) + "<br>"
            + " Velocidad: "    + vel_kmh  + " km/h<br>"
            + " Alt. baro: "    + alt_ft   + " ft<br>"
            + " Alt. geo: "     + geo_ft   + " ft<br>"
            + " V. vertical: "  + vrate_ms + " m/s (" + vrate_fpm + " fpm)<br>"
            + " Rumbo: "        + df[hdg_col].round(1).astype(str) + "°"
        )
    else:
        lat_col, lon_col = "lat", "lon"
        vel_col, alt_col = "velocity", "baroaltitude"
        hdg_col = "heading"

        vel_kmh = (df[vel_col] * 3.6).round(1).astype(str)
        alt_ft = (df[alt_col] * 3.281).round(0).astype(int).astype(str)
        vrate = df["vertrate"].round(1).astype(str) if "vertrate" in df.columns else pd.Series(["N/A"] * len(df))

        callsign_s = sanitize(df["callsign"])
        icao_s = sanitize(df["icao24"])
        origen_s = sanitize(df["origen"]) if "origen" in df.columns else pd.Series(["N/A"] * len(df))
        destino_s = sanitize(df["destino"]) if "destino" in df.columns else pd.Series(["N/A"] * len(df))

        if modo == "live":
            lat_col, lon_col = "latitude", "longitude"
            vel_col, alt_col = "velocity", "baro_altitude"
            hdg_col = "true_track"

            vel_kmh = (df[vel_col] * 3.6).round(1).astype(str)
            alt_ft = (df[alt_col] * 3.281).round(0).astype(int).astype(str)
            geo_ft = (df["geo_altitude"] * 3.281).round(0).astype(int).astype(str)
            vrate_ms = df["vertical_rate"].round(1).astype(str)
            vrate_fpm = (df["vertical_rate"] * 196.85).round(0).astype(int).astype(str)

            # ← Añade estas tres líneas aquí
            callsign_s = sanitize(df["callsign"])
            country_s = sanitize(df["origin_country"])
            squawk_s = sanitize(df["squawk"])

        hover = (
                "<b>" + callsign_s + "</b> (" + icao_s + ")<br>"
                + "Origen: " + origen_s + "<br>"
                + "Destino: " + destino_s + "<br>"
                + "---<br>"
                + "Lat: " + df["lat"].round(4).astype(str)
                + " | Lon: " + df["lon"].round(4).astype(str) + "<br>"
                + "Velocidad: " + vel_kmh + " km/h<br>"
                + "Altitud: " + alt_ft + " ft<br>"
                + "V.vertical: " + vrate + " m/s<br>"
                + "Rumbo: " + df[hdg_col].round(1).astype(str) + " deg"
        )

    fig.add_trace(go.Scattermap(
        lat=df[lat_col], lon=df[lon_col],
        mode="markers",
        name=f"✈ Aviones ({len(df):,})",
        marker=go.scattermap.Marker(size=11, color="yellow", opacity=0.9),
        text=hover, hoverinfo="text",
        customdata=df["icao24"]
    ))

# ── CAPA 4: TRAYECTORIA DEL AVIÓN SELECCIONADO ───────────────────
track_data = st.session_state.get("proy_track")
sel_icao   = st.session_state.get("proy_sel_icao","")

if track_data and sel_icao:
    if track_data["tipo"]=="live" and track_data["data"]:
        track = track_data["data"]
        path  = track.get("path",[])
        if path:
            df_path = pd.DataFrame(path, columns=["time","lat","lon","baro_alt","heading","on_ground"])
            df_path = df_path.dropna(subset=["lat","lon"])
            alt_ft_t= (pd.to_numeric(df_path["baro_alt"],errors="coerce").fillna(0)*3.281).round(0).astype(int)
            hover_t = ("🛤 "+sel_icao+"<br>Alt: "+alt_ft_t.astype(str)+" ft")
            fig.add_trace(go.Scattermap(
                lat=df_path["lat"], lon=df_path["lon"],
                mode="lines+markers", name=f"Trayectoria {sel_icao}",
                line=dict(width=3, color="cyan"),
                marker=go.scattermap.Marker(size=4, color="white"),
                text=hover_t, hoverinfo="text"
            ))
            # Marcar inicio y fin
            fig.add_trace(go.Scattermap(
                lat=[df_path["lat"].iloc[0], df_path["lat"].iloc[-1]],
                lon=[df_path["lon"].iloc[0], df_path["lon"].iloc[-1]],
                mode="markers+text", name="Inicio/Fin",
                text=["▶ Inicio","■ Ahora"], textposition="top right",
                marker=go.scattermap.Marker(size=13, color=["lime","red"]),
                hoverinfo="text"
            ))

    elif track_data["tipo"]=="historico":
        df_tray = track_data["data"]
        if not df_tray.empty:
            alt_ft_t= (pd.to_numeric(df_tray["baroaltitude"],errors="coerce").fillna(0)*3.281).round(0).astype(int).astype(str)
            vel_t   = (pd.to_numeric(df_tray["velocity"],errors="coerce").fillna(0)*3.6).round(0).astype(int).astype(str)
            hover_t = ""+sel_icao+"<br>Alt: "+alt_ft_t+" ft · Vel: "+vel_t+" km/h"
            fig.add_trace(go.Scattermap(
                lat=df_tray["lat"], lon=df_tray["lon"],
                mode="lines+markers", name=f"Trayectoria {sel_icao}",
                line=dict(width=3, color="cyan"),
                marker=go.scattermap.Marker(size=4, color="white"),
                text=hover_t, hoverinfo="text"
            ))
            fig.add_trace(go.Scattermap(
                lat=[df_tray["lat"].iloc[0], df_tray["lat"].iloc[-1]],
                lon=[df_tray["lon"].iloc[0], df_tray["lon"].iloc[-1]],
                mode="markers+text", name="Inicio/Fin",
                text=["▶ Inicio","■ Fin"], textposition="top right",
                marker=go.scattermap.Marker(size=13, color=["lime","red"]),
                hoverinfo="text"
            ))

# ── LAYOUT DEL MAPA ──────────────────────────────────────────────
if not df.empty:
    if modo=="live":
        bb = REGIONES[region]
        mc = dict(lat=(bb[0]+bb[1])/2, lon=(bb[2]+bb[3])/2)
        mz = 3.0 if region!="Mundial" else 1.5
    else:
        bbox_h = BBOXES_CONT.get(cont_h if "cont_h" in dir() else "EU", (-25.0,29.0,45.0,81.2))
        mc = dict(lat=(bbox_h[1]+bbox_h[0])/2, lon=(bbox_h[2]+bbox_h[3])/2)
        mz = 3.5
else:
    mc = dict(lat=45, lon=10); mz = 3.0

fig.update_layout(
    map_style="carto-darkmatter",
    margin={"r":0,"t":0,"l":0,"b":0},
    height=820,
    showlegend=True,
    legend=dict(
        yanchor="top", y=0.98, xanchor="left", x=0.01,
        bgcolor="rgba(0,0,0,0.65)", font=dict(color="white", size=11),
        itemsizing="constant"
    ),
    map=dict(center=mc, zoom=mz)
)

event = st.plotly_chart(
    fig, use_container_width=True,
    config={"scrollZoom": True},
    on_select="rerun",
    selection_mode="points"
)

# Si el usuario hizo click en un avión, guardar su icao24
if event and event.selection and event.selection.points:
    pt = event.selection.points[0]
    icao_clicked = pt.get("customdata")
    if icao_clicked and icao_clicked != st.session_state.get("proy_sel_icao"):
        st.session_state["proy_sel_icao"] = icao_clicked
        st.session_state["proy_track"]    = None
        st.rerun()

# ── STATS RÁPIDAS (bajo el mapa) ─────────────────────────────────
if not df.empty:
    st.divider()
    if modo=="live":
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("✈ Aviones",         f"{len(df):,}")
        c2.metric(" Países",           f"{df['origin_country'].nunique()}")
        c3.metric(" Vel. media",       f"{(df['velocity']*3.6).mean():.0f} km/h")
        c4.metric(" Alt. media",       f"{(df['baro_altitude']*3.281).mean():.0f} ft")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("✈ Aviones",          f"{len(df):,}")
        c2.metric(" Con destino",       f"{n_dest_hist:,}")
        c3.metric(" Vel. media",        f"{(df['velocity']*3.6).mean():.0f} km/h")
        c4.metric(" Alt. media",        f"{(df['baroaltitude']*3.281).mean():.0f} ft")

# Mensaje de bienvenida si no hay datos
if df.empty and not st.session_state.get("proy_df", pd.DataFrame()).empty is False:
    st.info("Usa el menú lateral para cargar datos o actualizar el tráfico en vivo.")