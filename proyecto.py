"""
pages/4_Proyecto.py
TFG: Simulacion y Analisis del Impacto Operativo de la Red Aerea Global
Jacob Altenburger Villar · UAX 2026
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta, timezone
from trino.dbapi import connect
from trino.auth import OAuth2Authentication

st.set_page_config(page_title="TFG - Proyecto Red Aerea", page_icon="✈", layout="wide")

TOKEN_URL = ("https://auth.opensky-network.org/auth/realms/"
             "opensky-network/protocol/openid-connect/token")
REGIONES = {
    "Europa":       (29.0, 81.2, -25.0,  45.0),
    "Norteamerica": ( 7.4, 83.1,-176.6, -52.0),
    "Sudamerica":   (-55.0,12.4,-109.4, -32.4),
    "Asia":         (-12.2,80.8,  26.0, 180.0),
    "Africa":       (-35.0,37.2, -25.0,  63.4),
    "Mundial":      (-90.0,90.0,-180.0, 180.0),
}
BBOXES_CONT = {
    "EU":(-25.0,29.0,45.0,81.2),"NA":(-176.6,7.4,-52.0,83.1),
    "SA":(-109.4,-55.0,-32.4,12.4),"AS":(26.0,-12.2,180.0,80.8),
    "AF":(-25.1,-34.8,63.4,37.2),
}
POS_SOURCE = {0:"ADS-B",1:"ASTERIX",2:"MLAT",3:"FLARM"}
COLOR_PAS  = "rgba(255,165,0,0.9)"
COLOR_FUT  = "rgba(0,220,255,0.9)"

LEG_COLORS = [
    "rgba(255,165,0,0.9)",   # naranja  — leg 1
    "rgba(0,220,255,0.9)",   # cian     — leg 2
    "rgba(0,255,130,0.9)",   # verde    — leg 3
    "rgba(220,80,255,0.9)",  # magenta  — leg 4
    "rgba(255,230,0,0.9)",   # amarillo — leg 5
    "rgba(255,100,100,0.9)", # rojo     — leg 6+
]

def sanitize(series):
    return (series.fillna("N/A").astype(str)
            .str.replace("\\","/",regex=False).str.replace('"',"'",regex=False)
            .str.replace('\n',' ',regex=False).str.replace('\r',' ',regex=False).str.strip())

@st.cache_data
def cargar_aeropuertos():
    df=pd.read_csv("airports.csv")
    df=df[df["type"].isin(["large_airport","medium_airport","small_airport"])].copy()
    df=df.dropna(subset=["latitude_deg","longitude_deg"])
    df["continent"]=df["continent"].fillna("NA")
    return df

df_airports=cargar_aeropuertos()

def obtener_token(cid,csc):
    try:
        r=requests.post(TOKEN_URL,data={"grant_type":"client_credentials","client_id":cid,"client_secret":csc},timeout=15)
        if r.status_code==200:
            d=r.json(); exp=datetime.now()+timedelta(seconds=d.get("expires_in",1800)-60)
            return d["access_token"],exp,None
        return None,None,f"HTTP {r.status_code}"
    except Exception as e: return None,None,str(e)

def get_token_valido():
    t=st.session_state.get("proy_token"); exp=st.session_state.get("proy_token_exp")
    if t and exp and datetime.now()<exp: return t,None
    cid=st.session_state.get("proy_client_id",""); csc=st.session_state.get("proy_client_secret","")
    t,exp,err=obtener_token(cid,csc)
    if err: return None,err
    st.session_state["proy_token"]=t; st.session_state["proy_token_exp"]=exp
    return t,None

def get_trino():
    user=st.session_state.get("proy_trino_user","")
    if "proy_trino_conn" not in st.session_state:
        st.session_state.proy_trino_conn=connect(
            host="trino.opensky-network.org",port=443,user=user,auth=OAuth2Authentication(),
            http_scheme="https",catalog="minio",schema="osky",request_timeout=120.0)
    return st.session_state.proy_trino_conn

def fetch_live(bbox):
    token,err=get_token_valido()
    if err: return pd.DataFrame(),err
    lat_min,lat_max,lon_min,lon_max=bbox
    try:
        r=requests.get("https://opensky-network.org/api/states/all",
            headers={"Authorization":f"Bearer {token}"},
            params={"lamin":lat_min,"lamax":lat_max,"lomin":lon_min,"lomax":lon_max,"extended":1},timeout=20)
        st.session_state["proy_creditos"]=r.headers.get("X-Rate-Limit-Remaining","?")
        if r.status_code==200:
            states=r.json().get("states",[])
            if not states: return pd.DataFrame(),None
            cols=["icao24","callsign","origin_country","time_position","last_contact","longitude","latitude",
                  "baro_altitude","on_ground","velocity","true_track","vertical_rate","sensors","geo_altitude",
                  "squawk","spi","position_source","category"]
            df=pd.DataFrame(states,columns=cols)
            df=df[df["on_ground"]==False].dropna(subset=["latitude","longitude"]).copy()
            for col in ["velocity","baro_altitude","geo_altitude","true_track","vertical_rate"]:
                df[col]=pd.to_numeric(df[col],errors="coerce").fillna(0)
            df["position_source"]=pd.to_numeric(df["position_source"],errors="coerce").fillna(0).astype(int)
            df["callsign"]=df["callsign"].fillna("").str.strip()
            df["pos_src_str"]=df["position_source"].map(POS_SOURCE).fillna("Desconocido")
            return df,None
        elif r.status_code==401: return pd.DataFrame(),"Token invalido."
        elif r.status_code==429: return pd.DataFrame(),f"Sin creditos. Espera {r.headers.get('X-Rate-Limit-Retry-After-Seconds','?')}s."
        else: return pd.DataFrame(),f"HTTP {r.status_code}"
    except Exception as e: return pd.DataFrame(),str(e)

def fetch_track_live(icao24):
    """Trayectoria del vuelo actual via /tracks/all. Coste: 4 creditos."""
    token,err=get_token_valido()
    if err: return None,err
    try:
        r=requests.get("https://opensky-network.org/api/tracks/all",
            headers={"Authorization":f"Bearer {token}"},params={"icao24":icao24,"time":0},timeout=20)
        if r.status_code==200: return r.json(),None
        elif r.status_code==404: return None,"Sin trayectoria disponible."
        else: return None,f"HTTP {r.status_code}"
    except Exception as e: return None,str(e)

def fetch_historico(fecha,hora,minuto,continente):
    dt_utc=datetime(fecha.year,fecha.month,fecha.day,hora,minuto,0,tzinfo=timezone.utc)
    ts=int(dt_utc.timestamp()); ts_h=ts-(ts%3600)
    ts_day=int(datetime(fecha.year,fecha.month,fecha.day,0,0,0,tzinfo=timezone.utc).timestamp())
    if isinstance(continente, list):
        lnmin = min(BBOXES_CONT[c][0] for c in continente if c in BBOXES_CONT)
        ltmin = min(BBOXES_CONT[c][1] for c in continente if c in BBOXES_CONT)
        lnmax = max(BBOXES_CONT[c][2] for c in continente if c in BBOXES_CONT)
        ltmax = max(BBOXES_CONT[c][3] for c in continente if c in BBOXES_CONT)
    else:
        lnmin, ltmin, lnmax, ltmax = BBOXES_CONT.get(continente, (-25.0, 29.0, 45.0, 81.2))
    conn=get_trino()
    q_sv=f"""SELECT icao24,MAX_BY(callsign,time) AS callsign,MAX_BY(lat,time) AS lat,
               MAX_BY(lon,time) AS lon,MAX_BY(velocity,time) AS velocity,
               MAX_BY(heading,time) AS heading,MAX_BY(baroaltitude,time) AS baroaltitude,
               MAX_BY(vertrate,time) AS vertrate
        FROM state_vectors_data4
        WHERE hour={ts_h} AND time BETWEEN {ts} AND {ts}+60 AND onground=false
          AND lat BETWEEN {ltmin-2} AND {ltmax+2} AND lon BETWEEN {lnmin-2} AND {lnmax+2}
          AND lat IS NOT NULL AND lon IS NOT NULL GROUP BY icao24"""
    cur=conn.cursor(); cur.execute(q_sv)
    rows=cur.fetchall(); cols=[d[0] for d in cur.description]
    df_sv=pd.DataFrame(rows,columns=cols)
    if df_sv.empty: return df_sv,0,ts
    icaos_str="','".join(df_sv["icao24"].tolist())
    q_fl=f"""SELECT icao24,TRIM(callsign) AS callsign_fl,estdepartureairport AS origen,
               estarrivalairport AS destino,firstseen,lastseen
        FROM flights_data4 WHERE day={ts_day} AND icao24 IN ('{icaos_str}')
          AND icao24 IS NOT NULL AND firstseen<={ts+3600} AND lastseen>={ts-3600}"""
    cur.execute(q_fl); rows=cur.fetchall(); cols=[d[0] for d in cur.description]
    df_fl=pd.DataFrame(rows,columns=cols)
    n_dest=0
    if not df_fl.empty:
        df_fl["diff"]=abs(df_fl["firstseen"]-ts)
        df_fl=df_fl.sort_values("diff").drop_duplicates("icao24",keep="first")
        df_sv=df_sv.merge(df_fl[["icao24","origen","destino"]],on="icao24",how="left")
        n_dest=int(df_sv["destino"].notna().sum())
    df_sv["callsign"]=df_sv["callsign"].fillna("").str.strip()
    for col in ["velocity","baroaltitude","heading","vertrate"]:
        df_sv[col]=pd.to_numeric(df_sv[col],errors="coerce").fillna(0)
    return df_sv,n_dest,ts

def fetch_trayectoria_dia_completo(icao24, fecha):
    """
    Trayectoria del dia completo para un icao24.
    Consulta las 24 particiones de hora del dia.
    Divide en legs cuando el gap entre puntos supera 30 min.
    Devuelve lista de DataFrames (uno por leg).
    """
    dt_day = datetime(fecha.year, fecha.month, fecha.day, 0, 0, 0, tzinfo=timezone.utc)
    ts_s   = int(dt_day.timestamp())
    ts_e   = ts_s + 86400  # 24 horas
    hours  = list(range(ts_s, ts_e, 3600))  # las 24 particiones
    h_str  = ",".join(str(h) for h in hours)
    conn   = get_trino()
    q = f"""
        SELECT time, lat, lon, baroaltitude, velocity, heading
        FROM state_vectors_data4
        WHERE hour IN ({h_str})
          AND time BETWEEN {ts_s} AND {ts_e}
          AND icao24 = '{icao24}'
          AND lat IS NOT NULL AND lon IS NOT NULL
        ORDER BY time
    """
    cur = conn.cursor(); cur.execute(q)
    rows = cur.fetchall(); cols = [d[0] for d in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty: return []

    # Detectar gaps > 30 min → nuevo leg
    df["gap"]    = df["time"].diff().fillna(0) > 1800
    df["leg_id"] = df["gap"].cumsum()

    legs = []
    for _, leg_df in df.groupby("leg_id"):
        if len(leg_df) >= 2:
            legs.append(leg_df.reset_index(drop=True))
    return legs


# Conectar con api adsbdb

@st.cache_data(show_spinner=False, ttl=3600)
def consultar_adsbdb(callsigns_tuple):
    """
    Consulta api.adsbdb.com para enriquecer origen/destino por callsign.
    Gratuito, sin autenticacion. Cachea 1 hora.
    callsigns_tuple: tuple de strings (para que sea hashable y cacheable)
    """
    resultados = {}
    for cs in callsigns_tuple:
        if not cs or len(cs.strip()) < 3: continue
        try:
            r = requests.get(
                f"https://api.adsbdb.com/v0/callsign/{cs.strip()}",
                timeout=5
            )
            if r.status_code == 200:
                fr = r.json().get("response",{}).get("flightroute",{})
                if fr:
                    orig = fr.get("origin",{})
                    dest = fr.get("destination",{})
                    resultados[cs.strip()] = {
                        "adb_origen_icao":  orig.get("icao_code",""),
                        "adb_origen_name":  orig.get("airport_name",""),
                        "adb_destino_icao": dest.get("icao_code",""),
                        "adb_destino_name": dest.get("airport_name",""),
                        "adb_aerolinea":    fr.get("airline",{}).get("name",""),
                    }
        except: pass
    return resultados


def consultar_adsbdb_uno(callsign):
    """Consulta adsbdb para UN callsign concreto. Rapido, sin cache."""
    if not callsign or len(callsign.strip()) < 3:
        return {}
    try:
        r = requests.get(
            f"https://api.adsbdb.com/v0/callsign/{callsign.strip()}",
            timeout=5
        )
        if r.status_code == 200:
            fr = r.json().get("response", {}).get("flightroute", {})
            if fr:
                orig = fr.get("origin", {})
                dest = fr.get("destination", {})
                return {
                    "adb_origen_icao":  orig.get("icao_code", ""),
                    "adb_origen_name":  orig.get("airport_name", ""),
                    "adb_destino_icao": dest.get("icao_code", ""),
                    "adb_destino_name": dest.get("airport_name", ""),
                    "adb_aerolinea":    fr.get("airline", {}).get("name", ""),
                }
    except:
        pass
    return {}


def consultar_adsbdb_aircraft(icao24):
    """Info del avion por icao24: tipo, matricula, operador, foto."""
    try:
        r = requests.get(
            f"https://api.adsbdb.com/v0/aircraft/{icao24.lower().strip()}",
            timeout=5
        )
        if r.status_code == 200:
            ac = r.json().get("response", {}).get("aircraft", {})
            if ac:
                return {
                    "tipo":      ac.get("type", ""),
                    "icao_type": ac.get("icao_type", ""),
                    "matricula": ac.get("registration", ""),
                    "operador":  ac.get("registered_owner", ""),
                    "pais_op":   ac.get("registered_owner_country_name", ""),
                    "foto":      ac.get("url_photo_thumbnail", ""),
                    "foto_full": ac.get("url_photo", ""),
                }
    except: pass
    return {}


# ================================================================
# LOGIN
# ================================================================
if not st.session_state.get("proy_logged_in"):
    _,col_c,_ = st.columns([1,1.2,1])
    with col_c:
        st.markdown("## TFG - Red Aerea Global")
        st.markdown("**Jacob Altenburger Villar · UAX 2026**")
        st.divider()
        st.markdown("### Iniciar sesion")
        client_id     = st.text_input("clientId", value="jaltevil@myuax.com-api-client")
        client_secret = st.text_input("clientSecret", type="password")
        trino_user    = st.text_input("Email Trino", value="jaltevil@myuax.com")
        btn_login     = st.button("Entrar", use_container_width=True, type="primary")
    if btn_login:
        if not client_id or not client_secret or not trino_user:
            st.error("Rellena todos los campos.")
        else:
            with st.spinner("Autenticando..."):
                token,exp,err=obtener_token(client_id,client_secret)
            if err: st.error(f"Error: {err}")
            else:
                st.session_state.update({
                    "proy_logged_in":True,"proy_client_id":client_id,
                    "proy_client_secret":client_secret,"proy_trino_user":trino_user.lower(),
                    "proy_token":token,"proy_token_exp":exp,"proy_modo":"live",
                    "proy_selected":[],"proy_tracks":{},
                })
                with st.spinner("Cargando trafico en vivo..."):
                    df_live,err2=fetch_live(REGIONES["Europa"])
                if not err2 and not df_live.empty:
                    st.session_state.update({
                        "proy_df":df_live,"proy_region":"Europa",
                        "proy_ts":datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                        "proy_ts_snap":None,
                    })
                st.rerun()
    st.stop()

# ================================================================
# ESTADO
# ================================================================
modo     = st.session_state.get("proy_modo","live")
df       = st.session_state.get("proy_df", pd.DataFrame())
region   = st.session_state.get("proy_region","Europa")
ts_label = st.session_state.get("proy_ts","")
creditos = st.session_state.get("proy_creditos","?")
selected = st.session_state.get("proy_selected",[])
tracks   = st.session_state.get("proy_tracks",{})
ts_snap  = st.session_state.get("proy_ts_snap",None)
n_dest   = st.session_state.get("proy_n_dest",0)
lat_col  = "latitude" if modo=="live" else "lat"
lon_col  = "longitude" if modo=="live" else "lon"
# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown(f"**TFG Red Aerea** · {st.session_state.get('proy_trino_user','')}")
    st.caption(f"Creditos API: {creditos}")
    if st.button("Cerrar sesion", use_container_width=True):
        for k in [k for k in st.session_state if k.startswith("proy_")]: del st.session_state[k]
        st.rerun()
    st.divider()
    nuevo_modo=st.radio("Modo:",["live","historico"],index=0 if modo=="live" else 1,
        format_func=lambda x:"En vivo (API)" if x=="live" else "Dia concreto (Trino)")
    if nuevo_modo!=modo:
        st.session_state.update({"proy_modo":nuevo_modo,"proy_df":pd.DataFrame(),
            "proy_selected":[],"proy_tracks":{},"proy_ts_snap":None})
        st.rerun()
    st.divider()
    if modo=="live":
        region_sel=st.selectbox("Region:",list(REGIONES.keys()),index=list(REGIONES.keys()).index(region))
        btn_act=st.button("Actualizar",use_container_width=True,type="primary")
    else:
        fecha_h=st.date_input("Dia:",datetime(2024,1,16))
        hora_h=st.selectbox("Hora (UTC):",list(range(24)),index=12)
        min_h=st.selectbox("Minuto:",list(range(0,60,5)),index=0)
        conts_h = st.multiselect(
            "Continentes:",
            list(BBOXES_CONT.keys()),
            default=["EU"],
            format_func=lambda x: {"EU": "Europa", "NA": "Norteamerica",
                                   "SA": "Sudamerica", "AS": "Asia", "AF": "Africa"}[x]
        )
        cont_h = conts_h[0] if conts_h else "EU"  # para compatibilidad
        df_dest_opts=df_airports[df_airports["type"].isin(["large_airport","medium_airport"])].sort_values("name")
        dest_sel=st.selectbox("Filtrar por destino:",["Sin filtro (todos)"]+df_dest_opts["name"].tolist())
        dest_icao=None
        if dest_sel!="Sin filtro (todos)":
            dest_icao=df_dest_opts[df_dest_opts["name"]==dest_sel]["ident"].values[0]
            st.caption(f"Solo vuelos con destino {dest_icao}.")
        btn_hist=st.button("Consultar Trino",use_container_width=True,type="primary")
    st.divider()
    st.markdown("**Opciones de visualizacion**")
    mostrar_lineas = st.toggle("Mostrar lineas de destino", value=True,
                               help="Muestra lineas desde cada avion hasta su aeropuerto destino")
    conts_aero = st.multiselect(
        "Aeropuertos visibles (continente):",
        ["EU", "NA", "SA", "AS", "AF", "OC"],
        default=["EU"],
        format_func=lambda x: {"EU": "Europa", "NA": "Norteamerica", "SA": "Sudamerica",
                               "AS": "Asia", "AF": "Africa", "OC": "Oceania"}[x]
    )
    st.divider()
    st.markdown("**Aviones seleccionados**")
    if selected:
        st.caption(f"{len(selected)} seleccionado(s). Click en mapa para añadir/quitar.")
        for icao in selected:
            tinfo = tracks.get(icao, {})
            adb   = tinfo.get("adsbdb", {})
            c1, c2 = st.columns([4,1])
            # Mostrar aerolinea si adsbdb la tiene
            label = adb.get("adb_aerolinea", icao) or icao
            c1.caption(f"**{label}** · `{icao}`")
            if adb.get("adb_origen_name"):
                st.caption(f"De: {adb['adb_origen_name']} ({adb['adb_origen_icao']})")
            if adb.get("adb_destino_name"):
                st.caption(f"A:  {adb['adb_destino_name']} ({adb['adb_destino_icao']})")
            if c2.button("x", key=f"rm_{icao}"):
                sel=list(st.session_state.get("proy_selected",[])); sel.remove(icao)
                tr=dict(st.session_state.get("proy_tracks",{})); tr.pop(icao,None)
                st.session_state["proy_selected"]=sel; st.session_state["proy_tracks"]=tr
                st.rerun()
        if st.button("Limpiar seleccion", use_container_width=True):
            st.session_state["proy_selected"]=[]; st.session_state["proy_tracks"]={}; st.rerun()
    else:
        st.caption("Haz click en un avion del mapa.\nSe cargara su trayectoria y datos de ruta automaticamente.")

# ================================================================
# ACCIONES
# ================================================================
if modo=="live" and "btn_act" in dir() and btn_act:
    with st.spinner("Actualizando..."):
        df_new,err=fetch_live(REGIONES[region_sel])
    if err: st.error(err)
    else:
        st.session_state.update({"proy_df":df_new,"proy_region":region_sel,
            "proy_ts":datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            "proy_selected":[],"proy_tracks":{},"proy_ts_snap":None})
        st.rerun()

if modo=="historico" and "btn_hist" in dir() and btn_hist:
    with st.spinner("Consultando Trino..."):
        try:
            df_h,n_d,ts_s=fetch_historico(fecha_h,hora_h,min_h,conts_h)
            if dest_icao and "destino" in df_h.columns:
                df_h=df_h[df_h["destino"]==dest_icao].reset_index(drop=True); n_d=len(df_h)
            st.session_state.update({"proy_df":df_h,"proy_ts":f"{fecha_h} {hora_h:02d}:{min_h:02d} UTC",
                "proy_n_dest":n_d,"proy_selected":[],"proy_tracks":{},"proy_ts_snap":ts_s,
                "proy_fecha_hist":fecha_h,"proy_hora_hist":hora_h,"proy_cont":cont_h})
            st.rerun()
        except Exception as e:
            st.error(f"Error Trino: {e}")
            if "proy_trino_conn" in st.session_state: del st.session_state["proy_trino_conn"]

# Resincronizar
df=st.session_state.get("proy_df",pd.DataFrame())
selected=st.session_state.get("proy_selected",[])
tracks=st.session_state.get("proy_tracks",{})
ts_snap=st.session_state.get("proy_ts_snap",None)
n_dest=st.session_state.get("proy_n_dest",0)
region=st.session_state.get("proy_region","Europa")
lat_col="latitude" if modo=="live" else "lat"
lon_col="longitude" if modo=="live" else "lon"
hidden_icaos = st.session_state.get("proy_hidden", set())
if not df.empty and "icao24" in df.columns:
    df["_lbl"]=(df["callsign"].fillna("?").replace("","?")+" ("+df["icao24"]+")")

# Banner
if not df.empty:
    if modo=="live":
        st.caption(f"En vivo: **{len(df):,}** aviones · Region: **{region}** · {ts_label} · {creditos} creditos")
    else:
        pct=round(n_dest/len(df)*100) if len(df)>0 else 0
        st.caption(f"Historico: **{len(df):,}** aviones · {ts_label} · {n_dest} con destino ({pct}%) · {len(selected)} seleccionados")
# Botón descarga CSV
if not df.empty:
    col_dl, _ = st.columns([1, 4])
    with col_dl:
        st.download_button(
            "⬇️ Descargar CSV",
            data=df.drop(columns=["_lbl"], errors="ignore").to_csv(index=False),
            file_name=f"vuelos_{modo}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
# ================================================================
# MAPA
# ================================================================
fig=go.Figure()
aero_pos=df_airports.set_index("ident")[["latitude_deg","longitude_deg"]].to_dict("index")

# Aeropuertos
if not df.empty:
    bbox_aero = (-180, -90, 180, 90)  # por defecto mundial
    if conts_aero:
        lats_min, lons_min, lats_max, lons_max = [], [], [], []
        for c in conts_aero:
            lnmn,ltmn,lnmx,ltmx = BBOXES_CONT.get(c, (-180,-90,180,90))
            lons_min.append(lnmn); lats_min.append(ltmn)
            lons_max.append(lnmx); lats_max.append(ltmx)
        bbox_aero = (min(lats_min)-2, max(lats_max)+2, min(lons_min)-2, max(lons_max)+2)

    ltmin_a, ltmax_a, lnmin_a, lnmax_a = bbox_aero
    df_av_filt = df_airports[
        (df_airports["latitude_deg"]  >= ltmin_a) & (df_airports["latitude_deg"]  <= ltmax_a) &
        (df_airports["longitude_deg"] >= lnmin_a) & (df_airports["longitude_deg"] <= lnmax_a)
    ]

    df_large  = df_av_filt[df_av_filt["type"] == "large_airport"]
    df_medium = df_av_filt[df_av_filt["type"] == "medium_airport"]
    df_small  = df_av_filt[df_av_filt["type"] == "small_airport"]

    for df_t, color, size, visible, nombre in [
        (df_large,  "#FF4B4B", 12, True,        "Large airport"),
        (df_medium, "#1C83E1",  8, True,         "Medium airport"),
        (df_small,  "#00AA44",  5, "legendonly", "Small airport"),
    ]:
        if df_t.empty: continue
        hover_ap = (
            "<b>" + df_t["name"].fillna("") + "</b><br>"
            + "ICAO: "   + df_t["ident"].fillna("") + "<br>"
            + "IATA: "   + df_t["iata_code"].fillna("N/A") + "<br>"
            + "Ciudad: " + df_t["municipality"].fillna("N/A") + "<br>"
            + "Pais: "   + df_t["iso_country"].fillna("N/A") + "<br>"
            + "Elevacion: " + df_t["elevation_ft"].fillna(0).astype(int).astype(str) + " ft"
        )
        fig.add_trace(go.Scattermap(
            lat=df_t["latitude_deg"], lon=df_t["longitude_deg"],
            mode="markers", name=nombre,
            marker=go.scattermap.Marker(size=size, color=color, opacity=0.65),
            text=hover_ap, hoverinfo="text", visible=visible
        ))
# Lineas destino (historico)
if not df.empty and modo=="historico" and "destino" in df.columns and mostrar_lineas:
    sel_set=set(selected)
    lats_d,lons_d=[],[]
    for _,row in df[df["destino"].notna()&~df["icao24"].isin(sel_set)].iterrows():
        di=aero_pos.get(row["destino"])
        if di: lats_d+=[row[lat_col],di["latitude_deg"],None]; lons_d+=[row[lon_col],di["longitude_deg"],None]
    if lats_d:
        fig.add_trace(go.Scattermap(lat=lats_d,lon=lons_d,mode="lines",
            line=dict(width=1,color="rgba(255,255,255,0.12)"),hoverinfo="none",showlegend=False,name="Rutas"))
    for icao in sel_set:
        fila=df[df["icao24"]==icao]
        if fila.empty: continue
        dest=fila.iloc[0].get("destino")
        if pd.isna(dest) or not dest: continue
        di=aero_pos.get(dest)
        if not di: continue
        fig.add_trace(go.Scattermap(lat=[fila.iloc[0][lat_col],di["latitude_deg"]],
            lon=[fila.iloc[0][lon_col],di["longitude_deg"]],mode="lines",
            line=dict(width=4,color="cyan"),hoverinfo="none",showlegend=False,name=f"Ruta {icao}"))

# Trayectorias
for icao,tinfo in tracks.items():
    if icao in hidden_icaos:
        continue
    if tinfo["tipo"] == "hist":
        legs = tinfo.get("legs", [])
        for i, leg_df in enumerate(legs):
            color = LEG_COLORS[i % len(LEG_COLORS)]
            alt_l = (pd.to_numeric(leg_df["baroaltitude"], errors="coerce").fillna(0) * 3.281).round(0).astype(
                int).astype(str)
            vel_l = (pd.to_numeric(leg_df["velocity"], errors="coerce").fillna(0) * 3.6).round(0).astype(int).astype(
                str)
            hora_inicio = datetime.utcfromtimestamp(int(leg_df["time"].iloc[0])).strftime("%H:%M")
            hora_fin = datetime.utcfromtimestamp(int(leg_df["time"].iloc[-1])).strftime("%H:%M")

            # Linea completa del trayecto en el color del leg
            fig.add_trace(go.Scattermap(
                lat=leg_df["lat"], lon=leg_df["lon"],
                mode="lines",
                name=f"Leg {i + 1} ({hora_inicio}-{hora_fin})",
                line=dict(width=4, color=color),
                text="Alt: " + alt_l + " ft | Vel: " + vel_l + " km/h",
                hoverinfo="text"
            ))

            # Solo dos marcadores: inicio y fin del leg
            fig.add_trace(go.Scattermap(
                lat=[leg_df["lat"].iloc[0], leg_df["lat"].iloc[-1]],
                lon=[leg_df["lon"].iloc[0], leg_df["lon"].iloc[-1]],
                mode="markers+text",
                text=[f"Inicio {hora_inicio}", f"Fin {hora_fin}"],
                textposition="top right",
                marker=go.scattermap.Marker(size=12, color=color, opacity=1.0),
                hoverinfo="text", showlegend=False
            ))
            # Linea punteada desde el ultimo punto conocido hasta el destino
            if legs:
                ultimo_punto = legs[-1].iloc[-1]
                fila_av = df[df["icao24"] == icao]
                destino_icao = None
                if not fila_av.empty and "destino" in fila_av.columns:
                    destino_icao = fila_av.iloc[0].get("destino")
                # Tambien intentar con adsbdb
                if not destino_icao:
                    adb_d = tinfo.get("adsbdb", {}).get("adb_destino_icao", "")
                    if adb_d: destino_icao = adb_d

                if destino_icao:
                    di = aero_pos.get(destino_icao)
                    if di:
                        fig.add_trace(go.Scattermap(
                            lat=[ultimo_punto["lat"], di["latitude_deg"]],
                            lon=[ultimo_punto["lon"], di["longitude_deg"]],
                            mode="lines",
                            name=f"Al destino ({destino_icao})",
                            line=dict(width=2, color="rgba(255,255,255,0.5)"),
                            hoverinfo="none", showlegend=False
                        ))
                        # Marcador del aeropuerto destino
                        info_dest = df_airports[df_airports["ident"] == destino_icao]
                        if not info_dest.empty:
                            fig.add_trace(go.Scattermap(
                                lat=[info_dest["latitude_deg"].values[0]],
                                lon=[info_dest["longitude_deg"].values[0]],
                                mode="markers+text",
                                text=[destino_icao],
                                textposition="top right",
                                marker=go.scattermap.Marker(size=14, color="white"),
                                hovertext=[f"Destino: {info_dest['name'].values[0]}"],
                                hoverinfo="text", showlegend=False
                            ))
    elif tinfo["tipo"]=="live":
        track=tinfo.get("data",{}); path=track.get("path",[]) if track else []
        if path:
            df_p=pd.DataFrame(path,columns=["time","lat","lon","baro_alt","heading","on_ground"]).dropna(subset=["lat","lon"])
            alt_p=(pd.to_numeric(df_p["baro_alt"],errors="coerce").fillna(0)*3.281).round(0).astype(int).astype(str)
            fig.add_trace(go.Scattermap(lat=df_p["lat"],lon=df_p["lon"],mode="lines+markers",
                name=f"Trayectoria {icao}",line=dict(width=3,color=COLOR_PAS),
                marker=go.scattermap.Marker(size=4,color="white"),text="Alt: "+alt_p+" ft",hoverinfo="text"))
            fig.add_trace(go.Scattermap(lat=[df_p["lat"].iloc[0],df_p["lat"].iloc[-1]],
                lon=[df_p["lon"].iloc[0],df_p["lon"].iloc[-1]],mode="markers+text",
                text=["Salida","Ahora"],textposition="top right",
                marker=go.scattermap.Marker(size=12,color=["lime","yellow"]),hoverinfo="text",showlegend=False))
            # Linea al destino si adsbdb lo conoce
            adb_d = tinfo.get("adsbdb", {}).get("adb_destino_icao", "")
            fila_live = df[df["icao24"] == icao]
            if adb_d and not fila_live.empty:
                di = aero_pos.get(adb_d)
                if di:
                    fig.add_trace(go.Scattermap(
                        lat=[fila_live.iloc[0]["latitude"], di["latitude_deg"]],
                        lon=[fila_live.iloc[0]["longitude"], di["longitude_deg"]],
                        mode="lines",
                        line=dict(width=2, color="rgba(255,255,255,0.4)"),
                        hoverinfo="none", showlegend=False, name=f"Al destino {adb_d}"
                    ))

# Aviones
if not df.empty:
    sel_set=set(selected)
    if modo=="live":
        vel_kmh=(df["velocity"]*3.6).round(1).astype(str)
        alt_ft=(df["baro_altitude"]*3.281).round(0).astype(int).astype(str)
        geo_ft=(df["geo_altitude"]*3.281).round(0).astype(int).astype(str)
        vrate_ms=df["vertical_rate"].round(1).astype(str)
        vrate_fpm=(df["vertical_rate"]*196.85).round(0).astype(int).astype(str)
        cs=sanitize(df["callsign"]); co=sanitize(df["origin_country"]); sq=sanitize(df["squawk"])
        hover=("<b>"+cs+"</b> ("+df["icao24"]+")<br>"+"Pais: "+co+"<br>"+"Fuente: "+df["pos_src_str"]+"<br>"
            +"Squawk: "+sq+"<br>"+"---<br>"+"Lat: "+df["latitude"].round(4).astype(str)
            +" | Lon: "+df["longitude"].round(4).astype(str)+"<br>"+"Vel: "+vel_kmh+" km/h<br>"
            +"Alt baro: "+alt_ft+" ft | Alt geo: "+geo_ft+" ft<br>"
            +"V.vert: "+vrate_ms+" m/s ("+vrate_fpm+" fpm)<br>"+"Rumbo: "+df["true_track"].round(1).astype(str)+" deg")
    else:
        vel_kmh=(df["velocity"]*3.6).round(1).astype(str)
        alt_ft=(df["baroaltitude"]*3.281).round(0).astype(int).astype(str)
        vrate=df["vertrate"].round(1).astype(str) if "vertrate" in df.columns else pd.Series(["N/A"]*len(df))
        cs=sanitize(df["callsign"])
        orig=sanitize(df["origen"]) if "origen" in df.columns else pd.Series(["N/A"]*len(df))
        dest=sanitize(df["destino"]) if "destino" in df.columns else pd.Series(["N/A"]*len(df))
        hover=("<b>"+cs+"</b> ("+df["icao24"]+")<br>"+"Origen: "+orig+"<br>"+"Destino: "+dest+"<br>"
            +"---<br>"+"Lat: "+df["lat"].round(4).astype(str)+" | Lon: "+df["lon"].round(4).astype(str)+"<br>"
            +"Vel: "+vel_kmh+" km/h<br>"+"Alt: "+alt_ft+" ft<br>"+"V.vert: "+vrate+" m/s<br>"
            +"Rumbo: "+df["heading"].round(1).astype(str)+" deg")

    df_ns=df[~df["icao24"].isin(sel_set)]
    if not df_ns.empty:
        fig.add_trace(go.Scattermap(lat=df_ns[lat_col],lon=df_ns[lon_col],mode="markers",
            name=f"Aviones ({len(df_ns):,})",marker=go.scattermap.Marker(size=7,color="yellow",opacity=0.8),
            text=hover[df_ns.index],hoverinfo="text",customdata=df_ns["icao24"]))
    df_s=df[df["icao24"].isin(sel_set)]
    if not df_s.empty:
        fig.add_trace(go.Scattermap(lat=df_s[lat_col],lon=df_s[lon_col],mode="markers",
            name=f"Seleccionados ({len(df_s)})",marker=go.scattermap.Marker(size=15,color="orange",opacity=1.0),
            text=hover[df_s.index],hoverinfo="text",customdata=df_s["icao24"]))

# Layout
if not df.empty:
    if modo=="live":
        bb=REGIONES[region]; mc=dict(lat=(bb[0]+bb[1])/2,lon=(bb[2]+bb[3])/2); mz=3.0 if region!="Mundial" else 1.5
    else:
        cont_a=st.session_state.get("proy_cont","EU"); lnmin,ltmin,lnmax,ltmax=BBOXES_CONT.get(cont_a,(-25.0,29.0,45.0,81.2))
        mc=dict(lat=(ltmin+ltmax)/2,lon=(lnmin+lnmax)/2); mz=3.5
else:
    mc=dict(lat=40.0,lon=-3.0); mz=4.0

fig.update_layout(map_style="carto-darkmatter",margin={"r":0,"t":0,"l":0,"b":0},height=820,
    uirevision="mapa_proyecto",showlegend=True,
    legend=dict(yanchor="top",y=0.98,xanchor="left",x=0.01,bgcolor="rgba(0,0,0,0.65)",font=dict(color="white",size=11)),
    map=dict(center=mc,zoom=mz))

# Renderizar y detectar click
event=st.plotly_chart(fig,use_container_width=True,config={"scrollZoom":True},
    on_select="rerun",selection_mode="points")

if event and event.selection and event.selection.points:
    sel_actual = list(st.session_state.get("proy_selected", []))
    tracks_new = dict(st.session_state.get("proy_tracks", {}))
    nuevos = []
    for pt in event.selection.points:
        icao_click = pt.get("customdata")
        if not icao_click: continue
        if icao_click in sel_actual:
            sel_actual.remove(icao_click)
            tracks_new.pop(icao_click, None)
        else:
            sel_actual.append(icao_click)
            nuevos.append(icao_click)

    if nuevos:
        with st.spinner("Cargando trayectoria e informacion..."):
            for icao in nuevos:
                if icao in tracks_new: continue

                # Trayectoria según modo
                if modo == "live":
                    data, err_t = fetch_track_live(icao)
                    if data:
                        tracks_new[icao] = {"tipo": "live", "data": data}
                    else:
                        tracks_new[icao] = {"tipo": "live", "data": None}
                        if err_t: st.warning(f"{icao}: {err_t}")
                else:
                    fecha_h_s = st.session_state.get("proy_fecha_hist")
                    legs = fetch_trayectoria_dia_completo(icao, fecha_h_s) if fecha_h_s else []
                    tracks_new[icao] = {"tipo": "hist", "legs": legs}

                # adsbdb para AMBOS modos — ruta y datos del avión
                fila_av = df[df["icao24"] == icao]
                if not fila_av.empty:
                    cs_av = str(fila_av.iloc[0].get("callsign", "")).strip()
                    if cs_av and len(cs_av) >= 3:
                        adb_ruta = consultar_adsbdb_uno(cs_av)
                        if adb_ruta:
                            tracks_new[icao]["adsbdb"] = adb_ruta
                adb_ac = consultar_adsbdb_aircraft(icao)
                if adb_ac:
                    tracks_new[icao]["aircraft"] = adb_ac

    st.session_state["proy_selected"] = sel_actual
    st.session_state["proy_tracks"]   = tracks_new
    st.rerun()
# Panel de info de aviones seleccionados
if selected and tracks:
    st.divider()
    st.markdown("### Aviones seleccionados")
    cols_info = st.columns(min(len(selected), 3))
    for i, icao in enumerate(selected):
        tinfo = tracks.get(icao, {})
        adb   = tinfo.get("adsbdb", {})
        ac    = tinfo.get("aircraft", {})
        fila  = df[df["icao24"] == icao]
        cs    = fila.iloc[0]["callsign"] if not fila.empty and "callsign" in fila.columns else icao

        with cols_info[i % 3]:
            st.markdown(f"**{cs}** · `{icao}`")

            # Toggle visibilidad
            visible_av = icao not in hidden_icaos
            nuevo_vis  = st.toggle("Mostrar en mapa", value=visible_av, key=f"vis_{icao}")
            if nuevo_vis != visible_av:
                h = set(st.session_state.get("proy_hidden", set()))
                if nuevo_vis: h.discard(icao)
                else:         h.add(icao)
                st.session_state["proy_hidden"] = h
                st.rerun()

            # Info aeronave (adsbdb aircraft)
            if ac:
                if ac.get("operador"):  st.markdown(f"**Aerolínea:** {ac['operador']}")
                if ac.get("tipo"):      st.markdown(f"**Modelo:** {ac['tipo']} `{ac.get('icao_type','')}`")
                if ac.get("matricula"): st.markdown(f"**Matrícula:** `{ac['matricula']}`")
                if ac.get("pais_op"):   st.markdown(f"**País operador:** {ac['pais_op']}")

            # Info de ruta (adsbdb route)
            if adb:
                orig = f"{adb.get('adb_origen_name','?')} ({adb.get('adb_origen_icao','?')})"
                dest = f"{adb.get('adb_destino_name','?')} ({adb.get('adb_destino_icao','?')})"
                st.markdown(f"**Ruta:** {orig} → {dest}")

            # Info del snapshot (datos de OpenSky)
            if not fila.empty:
                row = fila.iloc[0]
                st.markdown("---")
                if modo == "live":
                    datos = {
                        "País origen":    row.get("origin_country",""),
                        "Fuente señal":   row.get("pos_src_str",""),
                        "Squawk":         row.get("squawk","N/A"),
                        "Velocidad":      f"{row.get('velocity',0)*3.6:.0f} km/h",
                        "Alt. barom.":    f"{row.get('baro_altitude',0)*3.281:.0f} ft",
                        "Alt. geom.":     f"{row.get('geo_altitude',0)*3.281:.0f} ft",
                        "V. vertical":    f"{row.get('vertical_rate',0):.1f} m/s",
                        "Rumbo":          f"{row.get('true_track',0):.1f}°",
                        "Lat / Lon":      f"{row.get('latitude',0):.4f} / {row.get('longitude',0):.4f}",
                    }
                else:
                    datos = {
                        "Origen ICAO":   row.get("origen","N/A"),
                        "Destino ICAO":  row.get("destino","N/A"),
                        "Velocidad":     f"{row.get('velocity',0)*3.6:.0f} km/h",
                        "Altitud":       f"{row.get('baroaltitude',0)*3.281:.0f} ft",
                        "V. vertical":   f"{row.get('vertrate',0):.1f} m/s",
                        "Rumbo":         f"{row.get('heading',0):.1f}°",
                        "Lat / Lon":     f"{row.get('lat',0):.4f} / {row.get('lon',0):.4f}",
                    }
                for k, v in datos.items():
                    st.caption(f"**{k}:** {v}")

            # Foto
            if ac.get("foto"):
                st.image(ac["foto"], use_container_width=True)

            # Tabla trayectoria
            legs = tinfo.get("legs", [])
            if legs:
                st.markdown(f"**{len(legs)} tramo(s) ese día:**")
                for j, leg_df in enumerate(legs):
                    hi = datetime.utcfromtimestamp(int(leg_df["time"].iloc[0])).strftime("%H:%M")
                    hf = datetime.utcfromtimestamp(int(leg_df["time"].iloc[-1])).strftime("%H:%M")
                    with st.expander(f"Tramo {j+1} · {hi} → {hf}"):
                        df_tabla = pd.DataFrame({
                            "Hora UTC":  leg_df["time"].apply(lambda t: datetime.utcfromtimestamp(int(t)).strftime("%H:%M:%S")),
                            "Lat":       leg_df["lat"].round(4),
                            "Lon":       leg_df["lon"].round(4),
                            "Alt (ft)":  (pd.to_numeric(leg_df["baroaltitude"], errors="coerce").fillna(0)*3.281).round(0).astype(int),
                            "Vel (km/h)":(pd.to_numeric(leg_df["velocity"],     errors="coerce").fillna(0)*3.6).round(0).astype(int),
                            "Rumbo":     pd.to_numeric(leg_df["heading"],        errors="coerce").fillna(0).round(0).astype(int),
                        })
                        st.dataframe(df_tabla, use_container_width=True, height=200)
            st.divider()


# Stats
if not df.empty:
    st.divider()
    c1,c2,c3,c4=st.columns(4)
    if modo=="live":
        c1.metric("Aviones",f"{len(df):,}"); c2.metric("Paises",f"{df['origin_country'].nunique()}")
        c3.metric("Vel. media",f"{(df['velocity']*3.6).mean():.0f} km/h")
        c4.metric("Alt. media",f"{(df['baro_altitude']*3.281).mean():.0f} ft")
    else:
        c1.metric("Aviones",f"{len(df):,}"); c2.metric("Con destino",f"{n_dest:,}")
        c3.metric("Vel. media",f"{(df['velocity']*3.6).mean():.0f} km/h")
        c4.metric("Alt. media",f"{(df['baroaltitude']*3.281).mean():.0f} ft")