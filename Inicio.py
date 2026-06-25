"""
Inicio.py  ·  PUNTO DE ENTRADA de la app multipagina (PORTADA A PANTALLA COMPLETA).
TFG: Simulacion y Analisis del Impacto Operativo de la Red Aerea Global
Jacob Altenburger Villar - UAX 2026

Ejecuta SIEMPRE este archivo:   streamlit run Inicio.py
Portada: globo 3D que gira solo + red de rutas + campo de estrellas de fondo.
El boton para abrir el menu lateral (paginas) sigue visible arriba a la izquierda.
"""
import random
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="TFG - Red Aerea Global", page_icon="✈",
                   layout="wide", initial_sidebar_state="collapsed")

# --- CSS: cabecera transparente (NO oculta) para conservar el boton del menu ---
st.markdown("""
<style>
[data-testid="stHeader"]{background:transparent;}
[data-testid="collapsedControl"]{display:block!important; z-index:1000;}
.block-container{padding:0!important; max-width:100%!important;}
.stApp{background:#0b0f1c;}
.stApp iframe{height:100vh!important;}
[data-testid="stSidebar"]{background:#0b0f1c;}
</style>
""", unsafe_allow_html=True)

# ---- ~50 aeropuertos importantes del mundo (ICAO, lat, lon) ----
AEROPUERTOS = [
    ("LEMD",40.49,-3.57),("LEBL",41.30,2.08),("LEPA",39.55,2.74),("LEMG",36.67,-4.49),
    ("EGLL",51.47,-0.45),("EGKK",51.15,-0.19),("EHAM",52.31,4.76),("EDDF",50.03,8.57),
    ("EDDM",48.35,11.79),("LFPG",49.01,2.55),("LFMN",43.66,7.22),("LIRF",41.80,12.25),
    ("LSZH",47.46,8.55),("EKCH",55.62,12.65),("ENGM",60.19,11.10),("EIDW",53.42,-6.27),
    ("LOWW",48.11,16.57),("LTFM",41.26,28.74),("UUEE",55.97,37.41),("OMDB",25.25,55.36),
    ("OTHH",25.27,51.61),("OERK",24.96,46.70),("HECA",30.12,31.41),("FAOR",-26.13,28.24),
    ("DNMM",6.58,3.32),("GMMN",33.37,-7.59),("VIDP",28.57,77.10),("VABB",19.09,72.87),
    ("VHHH",22.31,113.91),("ZBAA",40.07,116.60),("ZSPD",31.14,121.81),("RJTT",35.55,139.78),
    ("RJAA",35.76,140.39),("RKSI",37.46,126.44),("WSSS",1.36,103.99),("VTBS",13.69,100.75),
    ("WMKK",2.74,101.71),("YSSY",-33.95,151.18),("NZAA",-37.01,174.79),("KJFK",40.64,-73.78),
    ("KLAX",33.94,-118.40),("KORD",41.97,-87.90),("KATL",33.64,-84.43),("KMIA",25.79,-80.29),
    ("KSFO",37.62,-122.38),("CYYZ",43.68,-79.63),("MMMX",19.44,-99.07),("SBGR",-23.43,-46.47),
    ("SAEZ",-34.82,-58.54),("SCEL",-33.39,-70.79),("SPJC",-12.02,-77.11),("FACT",-33.97,18.60),
]
HUBS = ["LEMD","EGLL","EDDF","LFPG","EHAM"]
DEST = ["KJFK","KLAX","SBGR","SAEZ","OMDB","WSSS","RJTT","ZBAA","VIDP","FAOR",
        "YSSY","CYYZ","MMMX","SCEL","HECA","VHHH","KMIA","SPJC","KATL","KORD"]
COORD = {i:(la,lo) for i,la,lo in AEROPUERTOS}


def _great_circle(lat1, lon1, lat2, lon2, n=70):
    la1,lo1,la2,lo2 = map(np.radians,[lat1,lon1,lat2,lon2])
    def xyz(la,lo):
        return np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)])
    p1,p2 = xyz(la1,lo1), xyz(la2,lo2)
    w = np.arccos(np.clip(np.dot(p1,p2),-1,1))
    if w == 0:
        return [lat1],[lon1]
    t = np.linspace(0,1,n)
    pts = (np.sin((1-t)*w)[:,None]*p1 + np.sin(t*w)[:,None]*p2)/np.sin(w)
    lat = np.degrees(np.arcsin(np.clip(pts[:,2],-1,1)))
    lon = np.degrees(np.arctan2(pts[:,1],pts[:,0]))
    return lat.tolist(), lon.tolist()


@st.cache_data(show_spinner=False)
def construir_globo():
    rlat, rlon = [], []
    for h in HUBS:
        for d in DEST:
            if h in COORD and d in COORD:
                la,lo = _great_circle(*COORD[h], *COORD[d])
                rlat += la + [None]; rlon += lo + [None]
    lats = [a[1] for a in AEROPUERTOS]; lons = [a[2] for a in AEROPUERTOS]
    names = [a[0] for a in AEROPUERTOS]

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(lat=rlat, lon=rlon, mode="lines",
        line=dict(width=0.8, color="rgba(0,200,255,0.35)"), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scattergeo(lat=lats, lon=lons, mode="markers",
        marker=dict(size=16, color="rgba(0,224,255,0.18)"), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scattergeo(lat=lats, lon=lons, mode="markers", text=names,
        marker=dict(size=5, color="#00e0ff", line=dict(width=0.5, color="rgba(255,255,255,0.7)")),
        hovertemplate="%{text}<extra></extra>", showlegend=False))
    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lat=22, lon=6, roll=0),
        showland=True, landcolor="#202a40",
        showocean=True, oceancolor="#070b16",
        showcountries=True, countrycolor="rgba(120,140,180,0.25)",
        showcoastlines=True, coastlinecolor="rgba(120,160,210,0.45)",
        showframe=False, bgcolor="rgba(0,0,0,0)",
        lataxis_showgrid=True, lonaxis_showgrid=True,
        lataxis_gridcolor="rgba(80,100,140,0.15)", lonaxis_gridcolor="rgba(80,100,140,0.15)",
    )
    # Fondo TRANSPARENTE para que se vean las estrellas alrededor del globo
    fig.update_layout(autosize=True, margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", dragmode="orbit")
    return fig


def campo_estrellas(n=170, seed=7):
    """Genera un cielo estrellado (SVG) con parpadeo y algunas lineas de constelacion."""
    rnd = random.Random(seed)
    estrellas = []
    coords = []
    for _ in range(n):
        x = rnd.uniform(0, 1280); y = rnd.uniform(0, 720)
        r = round(rnd.uniform(0.4, 1.8), 2)
        op = round(rnd.uniform(0.25, 1.0), 2)
        col = "#9fd8ff" if rnd.random() < 0.16 else "#ffffff"
        dur = round(rnd.uniform(2.5, 5.5), 1); beg = round(rnd.uniform(0, 5), 1)
        coords.append((x, y))
        estrellas.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{col}" opacity="{op}">'
            f'<animate attributeName="opacity" values="{op};{max(0.08, op*0.3):.2f};{op}" '
            f'dur="{dur}s" begin="{beg}s" repeatCount="indefinite"/></circle>')
    # lineas de constelacion: une pares de estrellas cercanas (hasta 12)
    lineas = []
    intentos = 0
    while len(lineas) < 12 and intentos < 400:
        intentos += 1
        a = rnd.randrange(n); b = rnd.randrange(n)
        if a == b:
            continue
        (x1, y1), (x2, y2) = coords[a], coords[b]
        if 40 < ((x1-x2)**2 + (y1-y2)**2) ** 0.5 < 200:
            lineas.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'stroke="rgba(180,210,255,0.14)" stroke-width="0.6"/>')
    return ('<svg style="position:absolute;inset:0;width:100%;height:100%;z-index:0" '
            'viewBox="0 0 1280 720" preserveAspectRatio="xMidYMid slice">'
            + "".join(lineas) + "".join(estrellas) + '</svg>')


# ---------------- PORTADA A PANTALLA COMPLETA ----------------
ALTURA = 900   # altura de respaldo en px (si el navegador no aplica 100vh, ajusta esto)

fig = construir_globo()
globo_html = fig.to_html(include_plotlyjs="cdn", full_html=False,
                         div_id="globo", default_width="100%", default_height="100%")
estrellas = campo_estrellas()

titulo = (
    "<div style=\"position:absolute;top:26px;left:0;right:0;text-align:center;"
    "z-index:5;pointer-events:none;font-family:'Segoe UI',Arial,sans-serif\">"
    "<div style=\"font-size:28px;font-weight:600;color:#eaf1ff;letter-spacing:.3px\">"
    "Simulaci&oacute;n y An&aacute;lisis del Impacto Operativo de la Red A&eacute;rea Global</div>"
    "<div style=\"font-size:15px;color:#93a7cc;margin-top:8px\">"
    "Jacob Altenburger Villar &middot; Ingenier&iacute;a Inform&aacute;tica &middot; UAX 2026</div>"
    "</div>"
)
rotacion = (
    "<script>(function(){var lon=6;setInterval(function(){lon=(lon+0.25)%360;"
    "var gd=document.getElementById('globo');"
    "if(gd&&gd.layout){Plotly.relayout(gd,{'geo.projection.rotation.lon':lon});}},50);})();</script>"
)

components.html(
    "<div style='position:relative;width:100%;height:100vh;background:#0b0f1c;overflow:hidden'>"
    + estrellas
    + titulo
    # globo un poco hacia abajo (translateY) para dejar aire bajo el titulo
    + "<div style='position:absolute;inset:0;z-index:1;transform:translateY(4vh)'>"
    + globo_html
    + "</div>"
    + rotacion
    + "</div>",
    height=ALTURA, scrolling=False,
)