"""
round_report_streamlit.py
=========================
Streamlit viewer predykcji kolejki.

Pokazuje:
- mecz + sędzia
- 1X2
- BTTS
- overy goli
- dla każdego rynku statystycznego:
    * średnia modelu
    * linia centralna
    * preferowany typ
    * pełny rozkład wszystkich liczonych linii

Uruchom:
python -m streamlit run scripts/predict/round_report_streamlit.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


PRED_DIR = Path("data/processed/predictions")

MARKETS = [
    {"key": "fouls",    "label": "Faule",         "mu_col": "mu_fouls"},
    {"key": "yc",       "label": "Żółte kartki",  "mu_col": "mu_yc"},
    {"key": "corners",  "label": "Kornery",       "mu_col": "mu_corners"},
    {"key": "offsides", "label": "Spalone",       "mu_col": "mu_offsides"},
    {"key": "shots",    "label": "Strzały",       "mu_col": "mu_shots"},
    {"key": "sot",      "label": "Strzały celne", "mu_col": "mu_sot"},
]

FILE_RE = re.compile(r"^predict_round_(\d{4}-\d{2})_K(\d{2})\.csv$")


# =============================================================================
# HELPERS
# =============================================================================

def fmt_pct(x) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x) * 100:.1f}%"


def fmt_float(x, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):.{digits}f}"


def filepart_to_season(filepart: str) -> str:
    return filepart.replace("-", "/")


def season_to_filepart(season: str) -> str:
    return season.replace("/", "-")


def discover_prediction_files() -> List[Dict]:
    rows = []
    if not PRED_DIR.exists():
        return rows

    for path in sorted(PRED_DIR.glob("predict_round_*.csv")):
        m = FILE_RE.match(path.name)
        if not m:
            continue
        filepart = m.group(1)
        kolejka = int(m.group(2))
        season = filepart_to_season(filepart)
        rows.append({
            "path": path,
            "season": season,
            "kolejka": kolejka,
            "label": f"{season} / K{kolejka:02d}",
        })

    rows.sort(key=lambda x: (x["season"], x["kolejka"]))
    return rows


@st.cache_data(show_spinner=False)
def load_predictions(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def parse_market_distribution(row: pd.Series, market_key: str) -> List[Dict]:
    pattern = re.compile(rf"^{re.escape(market_key)}_p_(over|under)_(\d+)_(\d+)$")
    tmp: Dict[float, Dict[str, float]] = {}

    for col in row.index:
        m = pattern.match(col)
        if not m:
            continue

        side = m.group(1)
        line = float(f"{m.group(2)}.{m.group(3)}")
        val = row[col]

        if pd.isna(val):
            continue

        if line not in tmp:
            tmp[line] = {
                "line": line,
                "p_over": None,
                "p_under": None,
            }

        tmp[line][f"p_{side}"] = float(val)

    out = list(tmp.values())
    out.sort(key=lambda x: x["line"])
    return out


def choose_central_line(dist: List[Dict]) -> Optional[Dict]:
    if not dist:
        return None
    return min(dist, key=lambda x: abs((x["p_over"] or 0.0) - 0.5))


def choose_preferred_pick(dist: List[Dict]) -> Optional[Dict]:
    candidates = []

    for x in dist:
        if x["p_over"] is not None:
            candidates.append({
                "side": "Over",
                "line": x["line"],
                "p": x["p_over"],
            })
        if x["p_under"] is not None:
            candidates.append({
                "side": "Under",
                "line": x["line"],
                "p": x["p_under"],
            })

    if not candidates:
        return None

    practical = [c for c in candidates if 0.56 <= c["p"] <= 0.80]
    if practical:
        practical.sort(key=lambda c: abs(c["p"] - 0.65))
        return practical[0]

    candidates.sort(key=lambda c: c["p"], reverse=True)
    return candidates[0]


def build_distribution_df(dist: List[Dict]) -> pd.DataFrame:
    if not dist:
        return pd.DataFrame(columns=["Linia", "Over", "Under"])

    rows = []
    for x in dist:
        rows.append({
            "Linia": f"{x['line']:.1f}",
            "Over": f"{(x['p_over'] or 0.0) * 100:.1f}%",
            "Under": f"{(x['p_under'] or 0.0) * 100:.1f}%",
        })
    return pd.DataFrame(rows)


def build_quick_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        match_name = f"{row['gospodarz']} vs {row['gosc']}"

        summary = {
            "Mecz": match_name,
            "Sędzia": row.get("referee_full_name", ""),
            "1": fmt_pct(row.get("p_H")),
            "X": fmt_pct(row.get("p_D")),
            "2": fmt_pct(row.get("p_A")),
            "BTTS TAK": fmt_pct(row.get("p_btts_yes")),
        }

        for market in ["fouls", "yc", "corners", "offsides"]:
            dist = parse_market_distribution(row, market)
            preferred = choose_preferred_pick(dist)
            label = next(x["label"] for x in MARKETS if x["key"] == market)
            if preferred is None:
                summary[label] = "-"
            else:
                summary[label] = f"{preferred['side']} {preferred['line']:.1f} ({preferred['p']*100:.1f}%)"

        rows.append(summary)

    return pd.DataFrame(rows)


# =============================================================================
# STREAMLIT
# =============================================================================

st.set_page_config(
    page_title="Ekstraklasa Predictor — raport kolejki",
    layout="wide",
)

st.title("Ekstraklasa Predictor — raport kolejki")
st.caption("Widok predykcji 1X2 + rynków statystycznych w układzie meczowym")

files = discover_prediction_files()

if not files:
    st.error("Brak plików predykcji w data/processed/predictions/")
    st.stop()

labels = [x["label"] for x in files]
default_idx = len(labels) - 1

selected_label = st.sidebar.selectbox("Wybierz plik predykcji", labels, index=default_idx)
selected = next(x for x in files if x["label"] == selected_label)

st.sidebar.markdown(f"**Plik:** `{selected['path']}`")

df = load_predictions(str(selected["path"]))

st.subheader(f"Sezon {selected['season']} — kolejka {selected['kolejka']}")
st.write(f"Liczba meczów: **{len(df)}**")

quick_df = build_quick_summary(df)
st.markdown("### Szybkie podsumowanie")
st.dataframe(quick_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("## Mecze")

for _, row in df.iterrows():
    home = row["gospodarz"]
    away = row["gosc"]
    ref = row.get("referee_full_name", "")
    if pd.isna(ref) or not ref:
        ref = "BRAK"

    term = f"{row.get('data_planowana', '')} {row.get('godzina', '')}".strip()
    if term == "nan nan":
        term = ""

    with st.expander(f"{home} vs {away}", expanded=False):
        col_a, col_b = st.columns([2, 1])

        with col_a:
            st.markdown(f"### {home} vs {away}")
            if term:
                st.write(f"**Termin:** {term}")
            st.write(f"**Sędzia:** {ref}")

        with col_b:
            st.metric("μ faule", fmt_float(row.get("mu_fouls")))
            st.metric("μ żółte", fmt_float(row.get("mu_yc")))

        st.markdown("#### Typy główne")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("1", fmt_pct(row.get("p_H")))
        c2.metric("X", fmt_pct(row.get("p_D")))
        c3.metric("2", fmt_pct(row.get("p_A")))
        c4.metric("BTTS TAK", fmt_pct(row.get("p_btts_yes")))

        st.write(
            f"**Gole over:** "
            f"O0.5 {fmt_pct(row.get('p_over_05'))} | "
            f"O1.5 {fmt_pct(row.get('p_over_15'))} | "
            f"O2.5 {fmt_pct(row.get('p_over_25'))} | "
            f"O3.5 {fmt_pct(row.get('p_over_35'))}"
        )

        st.markdown("---")
        st.markdown("#### Rynki statystyczne")

        for market in MARKETS:
            key = market["key"]
            label = market["label"]
            mu_col = market["mu_col"]

            dist = parse_market_distribution(row, key)
            central = choose_central_line(dist)
            preferred = choose_preferred_pick(dist)

            st.markdown(f"##### {label}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Średnia modelu", fmt_float(row.get(mu_col)))

            if central is not None:
                c2.metric(
                    "Linia centralna",
                    f"{central['line']:.1f}",
                    f"Over {fmt_pct(central['p_over'])} | Under {fmt_pct(central['p_under'])}"
                )
            else:
                c2.metric("Linia centralna", "-")

            if preferred is not None:
                c3.metric(
                    "Preferowany typ",
                    f"{preferred['side']} {preferred['line']:.1f}",
                    fmt_pct(preferred["p"])
                )
            else:
                c3.metric("Preferowany typ", "-")

            dist_df = build_distribution_df(dist)
            st.dataframe(dist_df, use_container_width=True, hide_index=True)

        st.markdown("---")

st.success("Raport załadowany.")