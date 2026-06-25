"""
Inicio.py  ·  PUNTO DE ENTRADA de la app multipagina.
TFG: Simulacion y Analisis del Impacto Operativo de la Red Aerea Global
Jacob Altenburger Villar - UAX 2026

Ejecuta SIEMPRE este archivo:   streamlit run Inicio.py
Es solo el arranque (portada). Las paginas reales estan en pages/.

Portada: globo terraqueo 3D que gira solo con la red de rutas aereas dibujada.
No depende de airports.csv. Cero librerias nuevas (numpy + plotly + streamlit).
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="TFG - Red Aerea Global", page_icon="✈", layout="wide")

# ---- ~50 aeropuertos importantes del mundo (ICAO, lat, lon) ----
AEROPUERTOS = [
    ("LEMD", 40.49, -3.57), ("LEBL", 41.30, 2.08), ("LEPA", 39.55, 2.74), ("LEMG", 36.67, -4.49),
    ("EGLL", 51.47, -0.45), ("EGKK", 51.15, -0.19), ("EHAM", 52.31, 4.76), ("EDDF", 50.03, 8.57),
    ("EDDM", 48.35, 11.79), ("LFPG", 49.01, 2.55), ("LFMN", 43.66, 7.22), ("LIRF", 41.80, 12.25),
    ("LSZH", 47.46, 8.55), ("EKCH", 55.62, 12.65), ("ENGM", 60.19, 11.10), ("EIDW", 53.42, -6.27),
    ("LOWW", 48.11, 16.57), ("LTFM", 41.26, 28.74), ("UUEE", 55.97, 37.41), ("OMDB", 25.25, 55.36),
    ("OTHH", 25.27, 51.61), ("OERK", 24.96, 46.70), ("HECA", 30.12, 31.41), ("FAOR", -26.13, 28.24),
    ("DNMM", 6.58, 3.32), ("GMMN", 33.37, -7.59), ("VIDP", 28.57, 77.10), ("VABB", 19.09, 72.87),
    ("VHHH", 22.31, 113.91), ("ZBAA", 40.07, 116.60), ("ZSPD", 31.14, 121.81), ("RJTT", 35.55, 139.78),
    ("RJAA", 35.76, 140.39), ("RKSI", 37.46, 126.44), ("WSSS", 1.36, 103.99), ("VTBS", 13.69, 100.75),
    ("WMKK", 2.74, 101.71), ("YSSY", -33.95, 151.18), ("NZAA", -37.01, 174.79), ("KJFK", 40.64, -73.78),
    ("KLAX", 33.94, -118.40), ("KORD", 41.97, -87.90), ("KATL", 33.64, -84.43), ("KMIA", 25.79, -80.29),
    ("KSFO", 37.62, -122.38), ("CYYZ", 43.68, -79.63), ("MMMX", 19.44, -99.07), ("SBGR", -23.43, -46.47),
    ("SAEZ", -34.82, -58.54), ("SCEL", -33.39, -70.79), ("SPJC", -12.02, -77.11), ("FACT", -33.97, 18.60),
]

# ---- rutas (hubs europeos -> destinos del mundo) para el "tejido" de red ----
HUBS = ["LEMD", "EGLL", "EDDF", "LFPG", "EHAM"]
DEST = ["KJFK", "KLAX", "SBGR", "SAEZ", "OMDB", "WSSS", "RJTT", "ZBAA", "VIDP", "FAOR",
        "YSSY", "CYYZ", "MMMX", "SCEL", "HECA", "VHHH", "KMIA", "SPJC", "KATL", "KORD"]
COORD = {i: (la, lo) for i, la, lo in AEROPUERTOS}


def _great_circle(lat1, lon1, lat2, lon2, n=70):
    """Arco de circulo maximo (la ruta real mas corta sobre la esfera)."""
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    def xyz(la, lo):
        return np.array([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)])
    p1, p2 = xyz(la1, lo1), xyz(la2, lo2)
    w = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
    if w == 0:
        return [lat1], [lon1]
    t = np.linspace(0, 1, n)
    pts = (np.sin((1-t)*w)[:, None]*p1 + np.sin(t*w)[:, None]*p2) / np.sin(w)
    lat = np.degrees(np.arcsin(np.clip(pts[:, 2], -1, 1)))
    lon = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    return lat.tolist(), lon.tolist()


@st.cache_data(show_spinner=False)
def construir_globo():
    rlat, rlon = [], []
    for h in HUBS:
        for d in DEST:
            if h in COORD and d in COORD:
                la, lo = _great_circle(*COORD[h], *COORD[d])
                rlat += la + [None]
                rlon += lo + [None]

    lats = [a[1] for a in AEROPUERTOS]
    lons = [a[2] for a in AEROPUERTOS]
    names = [a[0] for a in AEROPUERTOS]

    fig = go.Figure()
    # tejido de rutas (cian translucido)
    fig.add_trace(go.Scattergeo(lat=rlat, lon=rlon, mode="lines",
        line=dict(width=0.8, color="rgba(0,200,255,0.35)"), hoverinfo="skip", showlegend=False))
    # halo (glow) de los aeropuertos
    fig.add_trace(go.Scattergeo(lat=lats, lon=lons, mode="markers",
        marker=dict(size=14, color="rgba(0,224,255,0.18)"), hoverinfo="skip", showlegend=False))
    # punto brillante de cada aeropuerto
    fig.add_trace(go.Scattergeo(lat=lats, lon=lons, mode="markers", text=names,
        marker=dict(size=4.5, color="#00e0ff", line=dict(width=0.5, color="rgba(255,255,255,0.7)")),
        hovertemplate="%{text}<extra></extra>", showlegend=False))

    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lat=25, lon=6, roll=0),
        showland=True, landcolor="#202a40",
        showocean=True, oceancolor="#070b16",
        showcountries=True, countrycolor="rgba(120,140,180,0.25)",
        showcoastlines=True, coastlinecolor="rgba(120,160,210,0.45)",
        showframe=False, bgcolor="rgba(0,0,0,0)",
        lataxis_showgrid=True, lonaxis_showgrid=True,
        lataxis_gridcolor="rgba(80,100,140,0.15)", lonaxis_gridcolor="rgba(80,100,140,0.15)",
    )
    fig.update_layout(height=640, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0b0f1c", plot_bgcolor="#0b0f1c", dragmode="orbit")
    return fig


# ---------- PORTADA ----------
st.title("Simulacion y Analisis del Impacto Operativo de la Red Aerea Global")
st.caption("Jacob Altenburger Villar - Grado en Ingenieria Informatica - UAX 2026")

fig = construir_globo()

# Lo incrustamos como HTML para poder hacerlo girar solo con un poco de JS.
html = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="globo")
rotacion = """
<script>
(function(){
  var lon = 6;
  setInterval(function(){
    lon = (lon + 0.25) % 360;
    var gd = document.getElementById('globo');
    if (gd && gd.layout) { Plotly.relayout(gd, {'geo.projection.rotation.lon': lon}); }
  }, 50);
})();
</script>
"""
components.html("<div style='background:#0b0f1c'>" + html + rotacion + "</div>",
                height=660, scrolling=False)