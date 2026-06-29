"""
GridMind — Streamlit Dashboard
Autonomous Power Grid Fault Monitoring & Dispatch System
Mianwali 132kV Grid Station
"""

import os
import sys
import time
import json
import re
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ─── Path Setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.gridmind_agent import (
    run_gridmind_agent,
    FAULT_PRIORITY,
    AI_RESOLVABLE,
    HUMAN_REQUIRED,
    classify_response,
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GridMind — Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Constants ────────────────────────────────────────────────────────────────
ZONES = ["Zone_1", "Zone_2", "Zone_3", "Zone_4", "Zone_5", "Zone_6"]

CREW_BASE = {
    "Zone_1": "Team Alpha",
    "Zone_2": "Team Bravo",
    "Zone_3": "Team Charlie",
    "Zone_4": "Team Delta",
    "Zone_5": "Team Echo",
    "Zone_6": "Team Rescue",
}

CREW_COLORS = {
    "Team Alpha":  "#7c3aed",
    "Team Bravo":  "#0ea5e9",
    "Team Charlie": "#10b981",
    "Team Delta":  "#f59e0b",
    "Team Echo":   "#ef4444",
    "Team Rescue": "#ec4899",
}

SVG_POS = {
    "Zone_1": (55,  70),
    "Zone_2": (150, 32),
    "Zone_3": (245, 70),
    "Zone_4": (245, 168),
    "Zone_5": (150, 206),
    "Zone_6": (55,  168),
}

ZONE_EDGES = [
    ("Zone_1", "Zone_2"),
    ("Zone_2", "Zone_3"),
    ("Zone_3", "Zone_4"),
    ("Zone_4", "Zone_5"),
    ("Zone_5", "Zone_6"),
]

# ─── CSS ──────────────────────────────────────────────────────────────────────
def inject_css(theme: str):
    if theme == "Cyberpunk Neon":
        bg_main = "#000000"
        bg_panel = "#050505"
        border_col = "rgba(0, 255, 65, 0.3)"
        text_main = "#00ff41"
        text_muted = "#008f11"
        accent = "#ff00ff"
        panel_glow = "0 0 10px rgba(0, 255, 65, 0.1)"
    else:
        bg_main = "#06091a"
        bg_panel = "linear-gradient(145deg, #0e1829 0%, #0c1720 100%)"
        border_col = "rgba(0,160,255,0.11)"
        text_main = "#b0ccee"
        text_muted = "#364f68"
        accent = "#00d4ff"
        panel_glow = "none"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body, .stApp, [class*="css"], [data-testid] {{
    font-family: 'Inter', sans-serif !important;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.stApp {{
    background: {bg_main} !important;
    background-image:
        radial-gradient(ellipse 65% 45% at 12% 5%,  rgba(0,80,200,0.05)  0%, transparent 70%),
        radial-gradient(ellipse 55% 35% at 88% 95%, rgba(0,50,150,0.05)  0%, transparent 70%) !important;
}}
.block-container {{ padding: 1.1rem 2rem 2rem !important; max-width: 100% !important; }}

/* Critical Alert Pulse */
.critical-pulse {{
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    pointer-events: none; z-index: 9999;
    background: rgba(255, 0, 0, 0.1);
    animation: flash 1s infinite alternate;
}}
@keyframes flash {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

/* ── Header ──────────────────────────── */
.gm-header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.5rem 0 1rem;
    border-bottom: 1px solid {border_col};
    margin-bottom: 1rem;
}}
.gm-logo {{ display: flex; align-items: center; gap: 0.65rem; }}
.gm-logo-icon {{
    font-size: 1.75rem;
    filter: drop-shadow(0 0 12px {accent});
}}
.gm-logo-title {{
    font-size: 1.6rem; font-weight: 800; letter-spacing: -0.6px;
    color: {accent};
}}
.gm-logo-sub {{
    font-size: 0.75rem; color: {text_muted}; margin-left: 0.3rem; letter-spacing: 0.2px;
}}
.gm-status {{
    display: flex; align-items: center; gap: 0.5rem;
    background: rgba(0,255,136,0.07); border: 1px solid rgba(0,255,136,0.22);
    border-radius: 20px; padding: 0.28rem 0.9rem;
    font-size: 0.75rem; font-weight: 600; color: #00ff88;
}}
.gm-dot {{
    width: 7px; height: 7px; border-radius: 50%; background: #00ff88;
    animation: dot-pulse 2s infinite;
}}
@keyframes dot-pulse {{
    0%,100% {{ box-shadow: 0 0 0 0 rgba(0,255,136,0.55); }}
    50%      {{ box-shadow: 0 0 0 5px rgba(0,255,136,0); }}
}}

/* ── Stats ───────────────────────────── */
.gm-stats {{
    display: grid; grid-template-columns: repeat(4,1fr); gap: 0.85rem;
    margin-bottom: 0.95rem;
}}
.gm-stat {{
    background: {bg_panel};
    border: 1px solid {border_col}; border-radius: 12px;
    padding: 0.95rem 1.25rem; position: relative; overflow: hidden;
    box-shadow: {panel_glow};
}}
.gm-stat::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--c, {accent}); opacity: 0.65;
}}
.gm-stat-lbl {{
    font-size: 0.69rem; font-weight: 600; color: {text_muted};
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem;
}}
.gm-stat-val {{
    font-size: 2rem; font-weight: 800; color: var(--c, {text_main});
    line-height: 1; font-variant-numeric: tabular-nums;
}}

/* ── Panel ───────────────────────────── */
.gm-panel {{
    background: {bg_panel};
    border: 1px solid {border_col}; border-radius: 14px;
    padding: 1.05rem 1.25rem; height: 100%;
    box-shadow: {panel_glow};
}}
.gm-panel-hdr {{
    display: flex; align-items: center; gap: 0.5rem;
    font-size: 0.71rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.1px; color: {text_muted};
    padding-bottom: 0.65rem; margin-bottom: 0.85rem;
    border-bottom: 1px solid {border_col};
}}

/* ── Log Feed ────────────────────────── */
.gm-log {{ width: 100%; border-collapse: collapse; }}
.gm-log th {{
    font-size: 0.61rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.65px; color: {text_muted}; padding: 0.22rem 0.5rem;
    border-bottom: 1px solid {border_col};
    text-align: left; font-family: 'JetBrains Mono', monospace;
}}
.gm-lr {{ border-bottom: 1px solid rgba(0,160,255,0.04); }}
.gm-lr td {{
    padding: 0.28rem 0.5rem; font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace; color: {text_main}; opacity: 0.8;
}}
.gm-lr.is-fault td  {{ color: #ff6080; background: rgba(255,60,80,0.05); opacity: 1; }}
.gm-lr.is-active td {{
    color: #ff3355 !important; background: rgba(255,30,60,0.13) !important;
    font-weight: 700 !important; opacity: 1;
}}
.gm-lr.is-ai td {{ color: #00cc70; background: rgba(0,255,120,0.04); opacity: 1; }}
.badge-fault {{
    display: inline-block; background: rgba(255,60,80,0.14);
    border: 1px solid rgba(255,60,80,0.28); color: #ff7090;
    padding: 0.07rem 0.42rem; border-radius: 4px; font-size: 0.61rem; font-weight: 700;
}}
.badge-ai {{
    display: inline-block; background: rgba(0,255,120,0.10);
    border: 1px solid rgba(0,255,120,0.25); color: #00dd88;
    padding: 0.07rem 0.42rem; border-radius: 4px; font-size: 0.61rem; font-weight: 700;
}}
.badge-scan {{
    display: inline-block; background: rgba(0,180,255,0.12);
    border: 1px solid rgba(0,180,255,0.3); color: #00d4ff;
    padding: 0.07rem 0.42rem; border-radius: 4px; font-size: 0.61rem; font-weight: 700;
    animation: blink 0.9s infinite;
}}
@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0.45}} }}
.badge-normal {{ color: {text_muted}; font-size: 0.64rem; }}

.gm-log-wrap {{ max-height: 252px; overflow-y: auto; scrollbar-width: thin; scrollbar-color: rgba(0,160,255,0.14) transparent; }}

/* ── Stats Table ─────────────────────── */
.gm-stats-table {{ width: 100%; border-collapse: collapse; font-size: 0.75rem; color: {text_main}; }}
.gm-stats-table th {{ text-align: left; padding: 0.4rem 0.5rem; font-weight: 600; color: {text_muted}; border-bottom: 1px solid {border_col}; text-transform: uppercase; font-size: 0.65rem; }}
.gm-stats-table td {{ padding: 0.4rem 0.5rem; border-bottom: 1px solid rgba(0,160,255,0.04); font-family: 'JetBrains Mono', monospace; }}
.gm-stats-table tr:last-child td {{ border-bottom: none; }}

/* ── Reasoning Trace ─────────────────── */
.gm-trace {{
    max-height: 252px; overflow-y: auto; scrollbar-width: thin;
    scrollbar-color: rgba(0,160,255,0.14) transparent;
}}
.gm-tline {{
    display: flex; gap: 0.5rem; padding: 0.26rem 0;
    border-bottom: 1px solid rgba(0,160,255,0.04);
    font-size: 0.75rem; color: {text_main}; line-height: 1.45;
    font-family: 'JetBrains Mono', monospace;
}}
.gm-tnum {{
    font-size: 0.59rem; color: {text_muted}; min-width: 1.25rem; padding-top: 0.22rem;
    font-family: 'JetBrains Mono', monospace; flex-shrink: 0;
}}
.gm-tline.t-diag {{ color: {accent}; font-weight: 600; }}
.gm-tline.t-ok   {{ color: #00cc70; }}
.gm-tline.t-warn {{ color: #ffaa00; }}
.gm-tline.t-err  {{ color: #ff4466; }}
.gm-tline.t-ai   {{ color: #9f7aea; font-style: italic; }}
.gm-tline.t-head {{
    color: {accent}; font-size: 0.66rem; text-transform: uppercase;
    letter-spacing: 0.55px; font-weight: 600;
}}

/* ── Grid Map ────────────────────────── */
.gm-map-wrap {{ display: flex; flex-direction: column; align-items: center; gap: 0.55rem; }}
.gm-legend {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 0.28rem; width: 100%; margin-top: 0.2rem;
}}
.gm-leg-item {{ display: flex; align-items: center; gap: 0.32rem; font-size: 0.64rem; color: {text_muted}; }}
.gm-leg-dot  {{ width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }}

/* ── Dispatch ────────────────────────── */
.gm-disp {{
    background: rgba(0,160,255,0.04); border: 1px solid rgba(0,160,255,0.12);
    border-radius: 10px; padding: 0.85rem 1rem;
}}
.gm-disp-hdr {{ margin-bottom: 0.65rem; }}
.gm-dbadge {{
    display: inline-block; font-size: 0.62rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.85px;
    padding: 0.16rem 0.55rem; border-radius: 4px;
}}
.gm-dbadge.crew {{
    background: rgba(255,60,80,0.14); border: 1px solid rgba(255,60,80,0.28); color: #ff7090;
}}
.gm-dbadge.ai {{
    background: rgba(0,255,120,0.10); border: 1px solid rgba(0,255,120,0.25); color: #00dd88;
}}
.gm-drow {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.32rem 0; border-bottom: 1px solid rgba(0,160,255,0.06); font-size: 0.77rem;
}}
.gm-drow:last-child {{ border-bottom: none; }}
.gm-dk {{ color: {text_muted}; font-weight: 500; }}
.gm-dv {{ color: {text_main}; font-weight: 600; text-align: right; max-width: 62%; }}
.sev-critical {{
    display: inline-block; padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.67rem;
    font-weight: 700; text-transform: uppercase;
    background: rgba(255,60,80,0.14); border: 1px solid rgba(255,60,80,0.28); color: #ff4466;
}}
.sev-major {{
    display: inline-block; padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.67rem;
    font-weight: 700; text-transform: uppercase;
    background: rgba(255,170,0,0.12); border: 1px solid rgba(255,170,0,0.28); color: #ffaa00;
}}
.sev-minor {{
    display: inline-block; padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.67rem;
    font-weight: 700; text-transform: uppercase;
    background: rgba(255,215,0,0.10); border: 1px solid rgba(255,215,0,0.22); color: #ffd700;
}}

/* ── Buttons ─────────────────────────── */
div[data-testid="stButton"] > button {{
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 0.79rem !important; transition: all 0.2s !important; border: 1px solid !important;
}}
div[data-testid="stButton"] > button[kind="primary"] {{
    background: linear-gradient(135deg, #004f8a, #0066bb) !important;
    border-color: rgba(0,160,255,0.4) !important; color: #fff !important;
    box-shadow: 0 0 22px rgba(0,100,220,0.28) !important;
}}
div[data-testid="stButton"] > button[kind="secondary"] {{
    background: rgba(0,160,255,0.06) !important;
    border-color: rgba(0,160,255,0.18) !important; color: #5090b8 !important;
}}
div[data-testid="stButton"] > button:hover:not(:disabled) {{
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0,100,220,0.32) !important;
}}

/* ── Tabs ────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent !important;
    border-bottom: 1px solid {border_col} !important; gap: 0.35rem;
}}
.stTabs [data-baseweb="tab"] {{
    color: {text_muted} !important; font-weight: 600 !important; font-size: 0.77rem !important;
    border-radius: 6px 6px 0 0 !important; padding: 0.42rem 0.85rem !important;
    background: transparent !important;
}}
.stTabs [aria-selected="true"] {{
    color: {accent} !important; background: rgba(0,160,255,0.08) !important;
    border-bottom: 2px solid {accent} !important;
}}
div[data-baseweb="select"] > div:first-child {{
    background: rgba(0,160,255,0.06) !important;
    border: 1px solid {border_col} !important;
    border-radius: 8px !important; color: {text_main} !important;
}}
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────
def init_state():
    defaults: dict = {
        "theme":           "Default Dark",
        "scanning":        False,
        "scan_complete":   False,
        "log_entries":     [],
        "trace_lines":     [],
        "historical_traces": {},
        "dispatch_raw":    None,
        "dispatch_rtype":  None,
        "active_zone":     None,
        "fault_zones":     {},
        "critical_alert":  False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Data Helpers ─────────────────────────────────────────────────────────────
@st.cache_data
def load_csv() -> pd.DataFrame:
    return pd.read_csv("data/grid_logs.csv")


def load_history() -> dict:
    try:
        with open("reports/resolved_sites.json") as f:
            return json.load(f)
    except Exception:
        return {"resolved": [], "history": []}


# ─── Rendering Helpers ────────────────────────────────────────────────────────

def wrap_panel(icon: str, title: str, body: str) -> str:
    return f"""<div class="gm-panel">
  <div class="gm-panel-hdr">{icon}&nbsp;{title}</div>
  {body}
</div>"""


def render_stats_table(df: pd.DataFrame) -> str:
    faults = df[df["FaultLabel"] != 0].copy()
    faults["Resolved_By"] = faults["FaultLabel"].apply(lambda x: "AI" if x == 5 else "Crew")
    pt = pd.crosstab(faults["ZoneID"], faults["Resolved_By"])
    for col in ["AI", "Crew"]:
        if col not in pt.columns: pt[col] = 0
    pt["Total Faults"] = pt["AI"] + pt["Crew"]
    pt = pt.reset_index()
    
    rows = ""
    # Add a global totals row at the top
    tot_f = pt["Total Faults"].sum()
    tot_ai = pt["AI"].sum()
    tot_cr = pt["Crew"].sum()
    rows += f'<tr style="background:rgba(0,160,255,0.06); font-weight:700"><td>ALL ZONES</td><td style="color:#ff6080">{tot_f}</td><td style="color:#00cc70">{tot_ai}</td><td style="color:#ffaa00">{tot_cr}</td></tr>'
    
    for _, r in pt.iterrows():
        zone = r["ZoneID"].replace("Zone_", "Z")
        tot = r["Total Faults"]
        ai = r["AI"]
        crew = r["Crew"]
        rows += f'<tr><td>{zone}</td><td style="color:#ff6080">{tot}</td><td style="color:#00cc70">{ai}</td><td style="color:#ffaa00">{crew}</td></tr>'
        
    return f'<table class="gm-stats-table"><thead><tr><th>Zone</th><th>Total Faults</th><th>AI Resolved</th><th>Crew Dispatched</th></tr></thead><tbody>{rows}</tbody></table>'


def _trace_cls(line: str) -> str:
    if "💡" in line: return "t-ai"
    if "🎯" in line: return "t-diag"
    if "✅" in line: return "t-ok"
    if "❌" in line: return "t-err"
    if any(x in line for x in ["⚠️","🚨"]): return "t-warn"
    if "━━" in line: return "t-head"
    return ""


def render_trace(lines: list, is_scanning: bool) -> str:
    if not lines:
        return '<div class="gm-empty">Agent idle — start scan to see reasoning</div>'
    items = ""
    for i, l in enumerate(lines):
        is_last = (i == len(lines) - 1)
        cursor = '<span style="animation: blink 1s infinite">█</span>' if is_last and is_scanning else ''
        items += f'<div class="gm-tline {_trace_cls(l)}"><span class="gm-tnum">{i+1:02d}</span><span>{l} {cursor}</span></div>'
    return f'<div class="gm-trace">{items}</div>'


def render_map(active_zone: str | None, fault_zones: dict) -> str:
    parts: list[str] = []

    # Topology edges with SVG energy flow animation
    for z1, z2 in ZONE_EDGES:
        x1, y1 = SVG_POS[z1]; x2, y2 = SVG_POS[z2]
        is_active_edge = active_zone in (z1, z2)
        stroke = "#ff3355" if is_active_edge else "rgba(0,160,255,0.3)"
        dur = "0.5s" if is_active_edge else "2s"
        width = "2.5" if is_active_edge else "1.5"
        
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{width}" stroke-dasharray="6,4">'
            f'<animate attributeName="stroke-dashoffset" from="10" to="0" dur="{dur}" repeatCount="indefinite" />'
            f'</line>'
        )

    # Zones
    for zone in ZONES:
        cx, cy = SVG_POS[zone]
        label = zone.replace("Zone_", "Z")
        color = CREW_COLORS[CREW_BASE[zone]]

        if zone == active_zone:
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="22" fill="none" stroke="#ff3355" stroke-width="1.5"><animate attributeName="r" values="17;25;17" dur="1s" repeatCount="indefinite"/><animate attributeName="opacity" values="1;0;1" dur="1s" repeatCount="indefinite"/></circle>')
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="17" fill="#280010" stroke="#ff3355" stroke-width="2.5" filter="drop-shadow(0 0 8px #ff3355)"/>')
            txt = "#ffffff"
        elif zone in fault_zones:
            ftype = fault_zones[zone]
            if ftype == "ai":
                parts.append(f'<circle cx="{cx}" cy="{cy}" r="17" fill="#001a0d" stroke="#00cc70" stroke-width="2" filter="drop-shadow(0 0 6px #00cc70)"/>')
                txt = "#00cc70"
            elif ftype == "crew":
                parts.append(f'<circle cx="{cx}" cy="{cy}" r="17" fill="#1a0a02" stroke="#ff8844" stroke-width="2" filter="drop-shadow(0 0 6px #ff8844)"/>')
                txt = "#ff9966"
            else:
                parts.append(f'<circle cx="{cx}" cy="{cy}" r="17" fill="#1a1200" stroke="#ffaa00" stroke-width="2"/>')
                txt = "#ffaa00"
        else:
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="17" fill="#0a1425" stroke="{color}" stroke-width="1.5" opacity="0.8"/>')
            txt = color

        parts.append(f'<text x="{cx}" y="{cy+1}" text-anchor="middle" dominant-baseline="middle" fill="{txt}" font-size="9.5" font-weight="700" font-family="Inter, sans-serif">{label}</text>')

    svg = f'<svg viewBox="0 0 300 228" style="width:100%;max-width:310px;display:block;margin:auto"><rect width="300" height="228" fill="transparent"/>{"".join(parts)}</svg>'
    
    items = "".join(f'<div class="gm-leg-item"><div class="gm-leg-dot" style="background:{CREW_COLORS[CREW_BASE[z]]}"></div><span>Z{z[-1]} — {CREW_BASE[z]}</span></div>' for z in ZONES)
    return f'<div class="gm-map-wrap">{svg}<div class="gm-legend">{items}</div></div>'


def _parse_dispatch(s: str) -> dict:
    d = {}
    for key, pat in [("fault_type", r"FAULT TYPE\s*:\s*(.+?)[\r\n]"), ("zone", r"(?<!\w)ZONE\s*:\s*(.+?)[\r\n]"), ("severity", r"SEVERITY\s*:\s*(.+?)[\r\n]"), ("confidence", r"CONFIDENCE\s*:\s*(.+?)[\r\n]"), ("crew", r"ASSIGNED CREW\s*:\s*(.+?)[\r\n]"), ("breaker", r"BREAKER ID\s*:\s*(.+?)[\r\n]"), ("resp_time", r"RESPONSE TIME\s*:\s*(.+?)[\r\n]")]:
        m = re.search(pat, s, re.IGNORECASE)
        if m: d[key] = m.group(1).strip()
    return d

def render_dispatch(raw, rtype: str | None) -> str:
    if raw is None: return '<div class="gm-empty">No dispatch generated yet</div>'
    if raw == "HUMAN_REVIEW": return '<div class="gm-disp"><div style="color:#ffaa00;font-weight:700;">⚠️ HUMAN REVIEW REQUIRED</div></div>'
    p = _parse_dispatch(raw)
    sev = p.get("severity", "").strip().lower()
    sev_html = f'<span class="sev-{sev if sev in ("critical", "major", "minor") else "major"}">{p.get("severity","—")}</span>'
    badge_cls, badge_txt = ("ai", "🤖 AI Auto-Resolved") if rtype == "AI_RESOLVABLE" else ("crew", "🚒 Crew Dispatched")
    
    rows = "".join(f'<div class="gm-drow"><span class="gm-dk">{k}</span><span class="gm-dv">{v}</span></div>' for k, v in [("Fault", p.get("fault_type", "—")), ("Zone", p.get("zone", "—")), ("Severity", sev_html), ("Confidence", p.get("confidence", "—")), ("Crew", p.get("crew", "—")), ("Breaker", p.get("breaker", "—"))])
    return f'<div class="gm-disp"><div class="gm-disp-hdr"><span class="gm-dbadge {badge_cls}">{badge_txt}</span></div>{rows}</div>'


def create_telemetry_fig(df, zone_id, current_ts, theme):
    zone_data = df[df["ZoneID"] == zone_id].copy()
    zone_data = zone_data[zone_data["Timestamp"] <= current_ts].tail(10)
    
    bg_col = "#000000" if theme == "Cyberpunk Neon" else "#06091a"
    grid_col = "#00ff4133" if theme == "Cyberpunk Neon" else "rgba(0,160,255,0.1)"
    font_col = "#00ff41" if theme == "Cyberpunk Neon" else "#6ea4c8"
    v_col = "#ff00ff" if theme == "Cyberpunk Neon" else "#00d4ff"
    i_col = "#00ff41" if theme == "Cyberpunk Neon" else "#ffaa00"

    fig = go.Figure()
    if zone_data.empty:
        fig.update_layout(title="Waiting for telemetry...", template="plotly_dark", plot_bgcolor=bg_col, paper_bgcolor=bg_col)
        return fig

    fig.add_trace(go.Scatter(x=zone_data["Timestamp"], y=zone_data["Voltage"], mode='lines+markers', name='Voltage (kV)', line=dict(color=v_col, width=3)))
    fig.add_trace(go.Scatter(x=zone_data["Timestamp"], y=zone_data["Current"], mode='lines+markers', name='Current (A)', line=dict(color=i_col, width=3), yaxis="y2"))
    
    fig.update_layout(
        height=220, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor=bg_col, paper_bgcolor=bg_col, font=dict(color=font_col, family="Inter"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor=grid_col, showticklabels=False),
        yaxis=dict(title="Voltage (kV)", showgrid=True, gridcolor=grid_col),
        yaxis2=dict(title="Current (A)", overlaying="y", side="right", showgrid=False)
    )
    return fig


# ─── Main Application ─────────────────────────────────────────────────────────
def main():
    init_state()
    inject_css(st.session_state.theme)

    df = load_csv()
    faulty_cnt = len(df[df["FaultLabel"] != 0])
    
    # Global visual drama pulse
    if st.session_state.critical_alert:
        st.markdown('<div class="critical-pulse"></div>', unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="gm-header">
  <div class="gm-logo">
    <span class="gm-logo-icon">⚡</span>
    <div><span class="gm-logo-title">GridMind</span><span class="gm-logo-sub">Command Center</span></div>
  </div>
  <div class="gm-status"><div class="gm-dot"></div>Monitoring active</div>
</div>""", unsafe_allow_html=True)

    # ── Controls & Stats ──────────────────────────────────────────────────────
    st.markdown(wrap_panel("📊", "Fault Resolution Breakdown", render_stats_table(df)), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1.2, 1, 3, 2])
    with c1: start = st.button("▶ Start Scan" if not st.session_state.scanning else "⏳ Scanning…", type="primary", use_container_width=True, disabled=st.session_state.scanning)
    with c2: reset = st.button("🔄 Reset", type="secondary", use_container_width=True)
    with c3: sel_zone = st.selectbox("Zone filter", ["All Zones"] + ZONES, label_visibility="collapsed")
    with c4:
        theme = st.selectbox("Theme", ["Default Dark", "Cyberpunk Neon"], index=0 if st.session_state.theme == "Default Dark" else 1, label_visibility="collapsed")
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.rerun()

    if reset:
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()
    st.write("")

    # ── Layout: 1. Map & Dispatch (Top), 2. Telemetry, 3. Log & Trace ─────────
    r1c1, r1c2 = st.columns([1, 1])
    with r1c1: map_ph = st.empty()
    with r1c2: disp_ph = st.empty()
    
    st.write("")
    telemetry_ph = st.empty()
    st.write("")

    r3c1, r3c2 = st.columns([1, 1])
    with r3c1: log_ph = st.empty()
    with r3c2: trace_ph = st.empty()

    def refresh_ui(df_context=None, active_ts=None):
        map_ph.markdown(wrap_panel("⊞", "Grid Map", render_map(st.session_state.active_zone, st.session_state.fault_zones)), unsafe_allow_html=True)
        disp_ph.markdown(wrap_panel("📄", "Dispatch Output", render_dispatch(st.session_state.dispatch_raw, st.session_state.dispatch_rtype)), unsafe_allow_html=True)
        
        # Interactive Log Feed Dataframe
        log_df = pd.DataFrame(st.session_state.log_entries)
        if log_df.empty:
            log_ph.markdown(wrap_panel("≡", "Live Log Feed", '<div class="gm-empty">Awaiting scan — no readings yet</div>'), unsafe_allow_html=True)
            active_trace = st.session_state.trace_lines
        else:
            display_df = log_df.copy()
            def get_status(r):
                fl = r.get("fault_label", 0)
                if r.get("is_active"): return "● Scanning"
                if fl == 5: return "🟢 AI Resolved"
                if fl == 0: return "⚪ Normal"
                return "🔴 Fault Detected"
            
            display_df["Status"] = display_df.apply(get_status, axis=1)
            display_df = display_df[["time", "zone", "voltage", "current", "temp", "Status"]]
            display_df.columns = ["Time", "Zone", "V(kV)", "I(A)", "T(°C)", "Status"]
            
            with log_ph.container():
                st.markdown('<div class="gm-panel"><div class="gm-panel-hdr">≡&nbsp;Live Log Feed</div>', unsafe_allow_html=True)
                event = st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    selection_mode="single-row",
                    on_select="rerun",
                    key="log_table",
                    height=240
                )
                st.markdown('</div>', unsafe_allow_html=True)

            active_trace = st.session_state.trace_lines
            if not st.session_state.scanning and event.selection and event.selection["rows"]:
                selected_idx = event.selection["rows"][0]
                sel_time = log_df.iloc[selected_idx]["time"]
                sel_zone = log_df.iloc[selected_idx]["zone"]
                if (sel_time, sel_zone) in st.session_state.historical_traces:
                    active_trace = st.session_state.historical_traces[(sel_time, sel_zone)]

        trace_ph.markdown(wrap_panel("⊕", "Agent Reasoning Trace", render_trace(active_trace, st.session_state.scanning)), unsafe_allow_html=True)

        
        if df_context is not None and active_ts is not None and st.session_state.active_zone:
            fig = create_telemetry_fig(df_context, st.session_state.active_zone, active_ts, st.session_state.theme)
            telemetry_ph.plotly_chart(fig, use_container_width=True, key=f"plot_active_{time.time()}")
        else:
            fig = go.Figure().update_layout(title="Awaiting scan...", template="plotly_dark", height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            telemetry_ph.plotly_chart(fig, use_container_width=True, key=f"plot_idle_{time.time()}")

    refresh_ui()

    # ── Scan Loop ─────────────────────────────────────────────────────────────
    if start:
        st.session_state.scanning = True
        faulty_df = df[df["FaultLabel"] != 0].copy()
        faulty_df["_pri"] = faulty_df["FaultLabel"].astype(str).map(FAULT_PRIORITY)
        faulty_df = faulty_df.sort_values(["_pri", "Timestamp"])
        if sel_zone != "All Zones": faulty_df = faulty_df[faulty_df["ZoneID"] == sel_zone]

        queue = []
        seen = set()
        for _, row in faulty_df.iterrows():
            if row["ZoneID"] not in seen:
                seen.add(row["ZoneID"])
                queue.append(row)

        for row in queue:
            zone_id = row["ZoneID"]
            ts = str(row["Timestamp"])
            fl = int(row["FaultLabel"])
            
            st.session_state.active_zone = zone_id
            st.session_state.critical_alert = (fl == 3) # Visual drama for Lethal Conductor

            # Feed log
            ts_idx = df[(df["Timestamp"] == ts) & (df["ZoneID"] == zone_id)].index
            if not ts_idx.empty:
                surrounding = df.loc[max(0, ts_idx[0] - 2) : ts_idx[0] + 1]
                for _, sr in surrounding.iterrows():
                    st.session_state.log_entries.append({
                        "time": str(sr["Timestamp"]), "zone": sr["ZoneID"],
                        "voltage": f"{sr['Voltage']:.2f}", "current": f"{sr['Current']:.2f}",
                        "temp": f"{sr['Temperature']:.1f}", "fault_label": int(sr["FaultLabel"]),
                        "is_active": (str(sr["Timestamp"]) == ts and sr["ZoneID"] == zone_id)
                    })

            st.session_state.trace_lines = [f"━━ Processing {zone_id} @ {ts} ━━"]
            refresh_ui(df, ts)
            time.sleep(0.5)

            steps, result, _ = run_gridmind_agent(zone_id, ts)
            for step in steps:
                st.session_state.trace_lines.append(step)
                refresh_ui(df, ts)
                time.sleep(0.3)

            st.session_state.dispatch_raw = result
            st.session_state.dispatch_rtype = classify_response(fl)
            st.session_state.fault_zones[zone_id] = "ai" if classify_response(fl) == "AI_RESOLVABLE" else "crew"
            st.session_state.historical_traces[(ts, zone_id)] = st.session_state.trace_lines.copy()
            st.session_state.active_zone = None
            st.session_state.critical_alert = False
            for e in st.session_state.log_entries: e["is_active"] = False

            refresh_ui(df, ts)
            time.sleep(1)

        st.session_state.scanning = False
        st.session_state.trace_lines.append(f"━━ Scan complete ━━")
        refresh_ui()

if __name__ == "__main__":
    main()
