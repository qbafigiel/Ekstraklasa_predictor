import sqlite3
import pandas as pd
import re
import argparse
from pathlib import Path
from difflib import SequenceMatcher

# ──────────────────────────────────────────────────────────────────────────────
# KONFIG
# ──────────────────────────────────────────────────────────────────────────────

DB_PATH = "db/ekstraklasa.db"
RAW_ROOT = Path("data/raw/ekstraklasa_org")
PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("data/reports/player_identity")

SEASON_TO_RAW = {
    "2023/24": "2023-2024",
    "2024/25": "2024-2025",
    "2025/26": "2025-2026",
}

SEASON_TO_LABEL = {
    "2023/24": "2023_24",
    "2024/25": "2024_25",
    "2025/26": "2025_26",
}

SEASON_ORDER = {
    "2023/24": 1,
    "2024/25": 2,
    "2025/26": 3,
}

ALL_TEAM_MAP = {
    "Arka Gdynia": "arka-gdynia",
    "Bruk-Bet Termalica Nieciecza": "nieciecza",
    "Cracovia": "cracovia",
    "GKS Katowice": "gks-katowice",
    "Górnik Zabrze": "gornik-zabrze",
    "Jagiellonia Białystok": "jagiellonia-bialystok",
    "Korona Kielce": "korona-kielce",
    "Lech Poznań": "lech-poznan",
    "Lechia Gdańsk": "lechia-gdansk",
    "Legia Warszawa": "legia-warszawa",
    "Motor Lublin": "motor-lublin",
    "Piast Gliwice": "piast-gliwice",
    "Pogoń Szczecin": "pogon-szczecin",
    "Puszcza Niepołomice": "puszcza-niepolomice",
    "Radomiak Radom": "radomiak-radom",
    "Raków Częstochowa": "rakow-czestochowa",
    "Ruch Chorzów": "ruch-chorzow",
    "Stal Mielec": "stal-mielec",
    "Warta Poznań": "warta-poznan",
    "Widzew Łódź": "widzew-lodz",
    "Wisła Płock": "wisla-plock",
    "Zagłębie Lubin": "zagebie-lubin",
    "ŁKS Łódź": "lks-lodz",
    "Śląsk Wrocław": "slask-wroclaw",
}

OUTPUT_ROSTER = PROCESSED_DIR / "ekstra_player_roster_2023_2026.csv"
OUTPUT_TRANSFERS = PROCESSED_DIR / "ekstra_player_transfers_2023_2026.csv"

# ──────────────────────────────────────────────────────────────────────────────
# NORMALIZACJA
# ──────────────────────────────────────────────────────────────────────────────

PL_TO_ASCII = str.maketrans({
    "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n",
    "ó": "o", "ś": "s", "ź": "z", "ż": "z",
    "Ą": "A", "Ć": "C", "Ę": "E", "Ł": "L", "Ń": "N",
    "Ó": "O", "Ś": "S", "Ź": "Z", "Ż": "Z",
})

PL_TO_SLUGSTYLE = str.maketrans({
    "ą": "a", "ć": "c", "ę": "e", "ł": "",  "ń": "n",
    "ó": "o", "ś": "s", "ź": "z", "ż": "z",
    "Ą": "A", "Ć": "C", "Ę": "E", "Ł": "",  "Ń": "N",
    "Ó": "O", "Ś": "S", "Ź": "Z", "Ż": "Z",
})


def to_ascii(s: str) -> str:
    s = str(s).translate(PL_TO_ASCII).lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_slugstyle(s: str) -> str:
    s = str(s).translate(PL_TO_SLUGSTYLE).lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compact_ascii(s: str) -> str:
    return re.sub(r"[\s\-]", "", to_ascii(s))


def compact_slugstyle(s: str) -> str:
    return re.sub(r"[\s\-]", "", to_slugstyle(s))


def clean_flash_name(raw: str) -> str:
    raw = str(raw).strip()
    raw = re.sub(r"\s+\d+$", "", raw)
    return raw.strip()


def flash_base_name(name: str) -> str:
    name = clean_flash_name(name)
    cleaned = re.sub(r"(\s+[A-ZŁŚÓŹĆĄĘŃŻ]\.\s*)+$", "", name).strip()
    return cleaned


def flash_initials(name: str) -> list[str]:
    name = clean_flash_name(name)
    initials = re.findall(r"\b([A-ZŁŚÓŹĆĄĘŃŻ])\.", name)
    return [x.lower() for x in initials]


def flash_tokens(name: str) -> list[str]:
    base = flash_base_name(name)
    toks = re.split(r"[\s\-]+", to_ascii(base))
    return [t for t in toks if t]


def flash_tokens_slugstyle(name: str) -> list[str]:
    base = flash_base_name(name)
    toks = re.split(r"[\s\-]+", to_slugstyle(base))
    return [t for t in toks if t]

# ──────────────────────────────────────────────────────────────────────────────
# ŚCIEŻKI
# ──────────────────────────────────────────────────────────────────────────────

def get_label(season: str) -> str:
    return SEASON_TO_LABEL[season]


def get_mapping_path(season: str) -> Path:
    label = get_label(season)
    specific = PROCESSED_DIR / f"player_mapping_{label}.csv"
    legacy = PROCESSED_DIR / "player_mapping.csv"

    if specific.exists():
        return specific
    if season == "2025/26" and legacy.exists():
        return legacy
    return specific


def get_outputs(season: str) -> dict:
    label = get_label(season)
    return {
        "unmatched": REPORT_DIR / f"flash_unmatched_{label}.csv",
        "candidates": REPORT_DIR / f"flash_unmatched_candidates_{label}.csv",
        "report": REPORT_DIR / f"flash_identity_transfer_audit_report_{label}.txt",
    }

# ──────────────────────────────────────────────────────────────────────────────
# ROSTER EKSTRAKLASA.ORG
# ──────────────────────────────────────────────────────────────────────────────

def load_ekstra_roster() -> pd.DataFrame:
    rows = []

    for season_label, raw_folder in SEASON_TO_RAW.items():
        folder = RAW_ROOT / raw_folder
        if not folder.exists():
            continue

        for csv_path in folder.glob("*.csv"):
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            needed = {"player_slug", "klub_slug", "nazwa"}
            if not needed.issubset(df.columns):
                continue

            tmp = df[["player_slug", "klub_slug", "nazwa"]].copy()
            tmp["season"] = season_label
            tmp["source_file"] = csv_path.name
            rows.append(tmp)

    if not rows:
        raise RuntimeError("Nie znaleziono danych ekstraklasa.org w data/raw/ekstraklasa_org/*")

    roster = pd.concat(rows, ignore_index=True)
    roster = roster.dropna(subset=["player_slug", "klub_slug"])
    roster["player_slug"] = roster["player_slug"].astype(str).str.strip()
    roster["klub_slug"] = roster["klub_slug"].astype(str).str.strip()
    roster["nazwa"] = roster["nazwa"].astype(str).fillna("").str.strip()

    roster = roster.drop_duplicates(subset=["season", "player_slug", "klub_slug"]).copy()

    roster["slug_tokens"] = roster["player_slug"].apply(lambda s: [t for t in s.split("-") if t])
    roster["slug_first"] = roster["slug_tokens"].apply(lambda x: x[0] if x else "")
    roster["slug_last"] = roster["slug_tokens"].apply(lambda x: x[-1] if x else "")
    roster["slug_nonfirst"] = roster["slug_tokens"].apply(
        lambda x: "-".join(x[1:]) if len(x) > 1 else (x[0] if x else "")
    )
    roster["slug_full_compact"] = roster["player_slug"].str.replace("-", "", regex=False)
    roster["slug_nonfirst_compact"] = roster["slug_nonfirst"].str.replace("-", "", regex=False)
    roster["slug_token_count"] = roster["slug_tokens"].apply(len)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    roster.sort_values(["season", "klub_slug", "player_slug"]).to_csv(OUTPUT_ROSTER, index=False, encoding="utf-8")
    return roster


def build_transfer_table(roster: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        roster.groupby("player_slug")
        .agg(
            seasons=("season", lambda s: " | ".join(sorted(set(s)))),
            clubs=("klub_slug", lambda s: " | ".join(sorted(set(s)))),
            club_count=("klub_slug", lambda s: len(set(s))),
            season_count=("season", lambda s: len(set(s))),
            first_season=("season", "min"),
            last_season=("season", "max"),
            sample_name=("nazwa", "first"),
        )
        .reset_index()
    )

    transfers = grouped[grouped["club_count"] > 1].copy()
    transfers = transfers.sort_values(["club_count", "player_slug"], ascending=[False, True])

    details = (
        roster.groupby("player_slug")
        .apply(lambda g: " || ".join(
            f"{row.season}:{row.klub_slug}" for row in g.sort_values(["season", "klub_slug"]).itertuples()
        ))
        .reset_index(name="season_club_path")
    )

    transfers = transfers.merge(details, on="player_slug", how="left")
    transfers.to_csv(OUTPUT_TRANSFERS, index=False, encoding="utf-8")
    return transfers

# ──────────────────────────────────────────────────────────────────────────────
# FLASHSCORE - NIEDOPASOWANI
# ──────────────────────────────────────────────────────────────────────────────

def load_unmatched_flash(season: str, output_unmatched: Path) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    flash = pd.read_sql_query(
        """
        SELECT DISTINCT player_name, team_name
        FROM lineups
        WHERE sezon = ?
        """,
        conn,
        params=(season,)
    )
    conn.close()

    flash["flash_name"] = flash["player_name"].apply(clean_flash_name)
    flash["flash_team"] = flash["team_name"].astype(str).str.strip()
    flash["klub_slug_expected"] = flash["flash_team"].map(ALL_TEAM_MAP)

    missing_teams = sorted(flash.loc[flash["klub_slug_expected"].isna(), "flash_team"].unique())
    if missing_teams:
        raise RuntimeError(
            "Brak mapowania drużyn Flashscore -> klub_slug dla:\n"
            + "\n".join(f"  {x}" for x in missing_teams)
        )

    flash = flash[["flash_team", "klub_slug_expected", "flash_name"]].drop_duplicates()

    mapping_path = get_mapping_path(season)
    if mapping_path.exists():
        mapped = pd.read_csv(mapping_path)
        mapped["flash_name"] = mapped["flash_name"].astype(str).apply(clean_flash_name)
        mapped["flash_team"] = mapped["flash_team"].astype(str).str.strip()
        mapped_pairs = set(zip(mapped["flash_team"], mapped["flash_name"]))
        flash["is_mapped"] = flash.apply(lambda r: (r["flash_team"], r["flash_name"]) in mapped_pairs, axis=1)
        unmatched = flash[~flash["is_mapped"]].copy()
    else:
        unmatched = flash.copy()
        unmatched["is_mapped"] = False

    unmatched = unmatched.drop(columns=["is_mapped"], errors="ignore")
    unmatched = unmatched.sort_values(["flash_team", "flash_name"]).reset_index(drop=True)
    unmatched.to_csv(output_unmatched, index=False, encoding="utf-8")
    return unmatched

# ──────────────────────────────────────────────────────────────────────────────
# KANDYDACI GLOBALNI
# ──────────────────────────────────────────────────────────────────────────────

def candidate_relation(target_season: str, candidate_season: str, candidate_club: str, expected_club: str) -> str:
    same_season = candidate_season == target_season
    same_club = candidate_club == expected_club

    if same_season and same_club:
        return "same_season_same_club"
    if same_season and not same_club:
        return "same_season_other_club"
    if SEASON_ORDER[candidate_season] < SEASON_ORDER[target_season] and same_club:
        return "past_season_same_club"
    if SEASON_ORDER[candidate_season] < SEASON_ORDER[target_season] and not same_club:
        return "past_season_other_club"
    if SEASON_ORDER[candidate_season] > SEASON_ORDER[target_season] and same_club:
        return "future_season_same_club"
    return "future_season_other_club"


def score_candidate(flash_name: str, expected_club: str, row) -> tuple[int, list[str]]:
    reasons = []

    base = flash_base_name(flash_name)
    f_ascii_comp = compact_ascii(base)
    f_slug_comp = compact_slugstyle(base)
    f_toks = flash_tokens(flash_name)
    f_toks_slug = flash_tokens_slugstyle(flash_name)
    f_initials = flash_initials(flash_name)

    slug_tokens = row["slug_tokens"]
    slug_first = row["slug_first"]
    slug_last = row["slug_last"]
    slug_nonfirst = row["slug_nonfirst"]
    slug_full_comp = row["slug_full_compact"]

    score = 0

    if f_ascii_comp and f_ascii_comp == compact_ascii(slug_nonfirst.replace("-", " ")):
        score = max(score, 100)
        reasons.append("exact_nonfirst_ascii")

    if f_slug_comp and f_slug_comp == compact_slugstyle(slug_nonfirst.replace("-", " ")):
        score = max(score, 100)
        reasons.append("exact_nonfirst_slugstyle")

    if f_toks:
        last_flash_ascii = f_toks[-1]
        if last_flash_ascii == to_ascii(slug_last):
            score = max(score, 92)
            reasons.append("last_token_exact")

    if f_toks_slug:
        last_flash_slug = f_toks_slug[-1]
        if last_flash_slug == to_slugstyle(slug_last):
            score = max(score, 92)
            reasons.append("last_token_slugstyle")

    candidate_tokens_ascii = [to_ascii(t) for t in slug_tokens]
    candidate_tokens_slug = [to_slugstyle(t) for t in slug_tokens]

    overlap_ascii = [t for t in f_toks if t in candidate_tokens_ascii and len(t) >= 4]
    overlap_slug = [t for t in f_toks_slug if t in candidate_tokens_slug and len(t) >= 4]

    if overlap_ascii:
        score = max(score, 86)
        reasons.append(f"token_overlap_ascii:{','.join(overlap_ascii)}")

    if overlap_slug:
        score = max(score, 86)
        reasons.append(f"token_overlap_slug:{','.join(overlap_slug)}")

    sim_nonfirst = SequenceMatcher(None, f_ascii_comp, compact_ascii(slug_nonfirst.replace("-", " "))).ratio()
    sim_full = SequenceMatcher(None, f_ascii_comp, slug_full_comp).ratio()
    sim_best = max(sim_nonfirst, sim_full)

    if sim_best >= 0.92:
        score = max(score, 90)
        reasons.append(f"sim>=0.92:{sim_best:.3f}")
    elif sim_best >= 0.85:
        score = max(score, 84)
        reasons.append(f"sim>=0.85:{sim_best:.3f}")
    elif sim_best >= 0.78:
        score = max(score, 78)
        reasons.append(f"sim>=0.78:{sim_best:.3f}")

    if f_initials:
        initials_match = any(slug_first.startswith(x) for x in f_initials)
        if initials_match and score > 0:
            score += 4
            reasons.append("initial_match")

    if row["klub_slug"] == expected_club and score > 0:
        score += 3
        reasons.append("same_club_bonus")

    if score > 0 and row["season"] in SEASON_ORDER:
        score += 2 if row["season"] == max(SEASON_ORDER, key=SEASON_ORDER.get) else 0

    return score, reasons


def build_unmatched_candidates(target_season: str, roster: pd.DataFrame, unmatched: pd.DataFrame, output_candidates: Path) -> pd.DataFrame:
    candidate_rows = []

    # Buduj lookup per klub — drastycznie ogranicza przestrzeń przeszukiwania
    # Dla każdego zawodnika szukamy kandydatów tylko w:
    # 1. tym samym klubie (dowolny sezon)
    # 2. CAŁYM rosterze ale tylko z poprzednich sezonów (transfery)
    roster_records = roster.to_dict("records")

    # Index: klub_slug -> lista rekordów
    club_index = {}
    for row in roster_records:
        club = row["klub_slug"]
        if club not in club_index:
            club_index[club] = []
        club_index[club].append(row)

    for u in unmatched.itertuples(index=False):
        flash_team = u.flash_team
        expected_club = u.klub_slug_expected
        flash_name = u.flash_name

        # Kandydaci = ten sam klub (wszystkie sezony) + wszyscy z poprzednich sezonów
        season_order = SEASON_ORDER[target_season]
        past_seasons = {s for s, o in SEASON_ORDER.items() if o < season_order}

        same_club_records = club_index.get(expected_club, [])
        past_records = [r for r in roster_records if r["season"] in past_seasons]

        # Połącz i deduplikuj po player_slug
        seen = set()
        candidates_pool = []
        for r in same_club_records + past_records:
            key = (r["player_slug"], r["season"])
            if key not in seen:
                seen.add(key)
                candidates_pool.append(r)

        local_candidates = []

        for row in candidates_pool:
            score, reasons = score_candidate(flash_name, expected_club, row)
            if score <= 0:
                continue

            relation = candidate_relation(target_season, row["season"], row["klub_slug"], expected_club)

            local_candidates.append({
                "flash_team": flash_team,
                "expected_klub_slug": expected_club,
                "flash_name": flash_name,
                "candidate_player_slug": row["player_slug"],
                "candidate_klub_slug": row["klub_slug"],
                "candidate_season": row["season"],
                "relation": relation,
                "score": score,
                "reasons": " | ".join(reasons),
                "candidate_nazwa": row["nazwa"],
            })

        local_candidates = sorted(
            local_candidates,
            key=lambda x: (-x["score"], x["relation"], x["candidate_season"], x["candidate_player_slug"])
        )[:8]

        candidate_rows.extend(local_candidates)

    cand_df = pd.DataFrame(candidate_rows)

    if len(cand_df) == 0:
        cand_df = pd.DataFrame(columns=[
            "flash_team", "expected_klub_slug", "flash_name",
            "candidate_player_slug", "candidate_klub_slug", "candidate_season",
            "relation", "score", "reasons", "candidate_nazwa"
        ])
    else:
        cand_df = cand_df.sort_values(["flash_team", "flash_name", "score"], ascending=[True, True, False])

    cand_df.to_csv(output_candidates, index=False, encoding="utf-8")
    return cand_df

# ──────────────────────────────────────────────────────────────────────────────
# RAPORT
# ──────────────────────────────────────────────────────────────────────────────

def build_report(target_season: str, roster: pd.DataFrame, transfers: pd.DataFrame, unmatched: pd.DataFrame, candidates: pd.DataFrame, outputs: dict) -> str:
    lines = []

    lines.append("=" * 78)
    lines.append(f"AUDYT TOŻSAMOŚCI ZAWODNIKÓW + TRANSFERY + KANDYDACI [{target_season}]")
    lines.append("=" * 78)
    lines.append("")

    lines.append("1. EKSTRAKLASA.ORG — roster 3 sezony")
    lines.append("-" * 78)
    lines.append(f"Liczba unikalnych rekordów sezon-zawodnik-klub: {len(roster)}")
    for season in sorted(roster["season"].unique(), key=lambda x: SEASON_ORDER[x]):
        tmp = roster[roster["season"] == season]
        lines.append(
            f"  {season}: {len(tmp):4d} rekordów | "
            f"{tmp['player_slug'].nunique():3d} unikalnych player_slug | "
            f"{tmp['klub_slug'].nunique():2d} klubów"
        )
    lines.append("")

    lines.append("2. TRANSFERY / ZMIANY KLUBOWE")
    lines.append("-" * 78)
    lines.append(f"Zawodnicy z >1 klubem w 3 sezonach: {len(transfers)}")
    preview = transfers.head(30)
    if len(preview):
        lines.append("Top 30 przykładów:")
        for row in preview.itertuples(index=False):
            lines.append(f"  {row.player_slug:40s} | {row.clubs}")
    else:
        lines.append("Brak wykrytych zmian klubowych.")
    lines.append("")

    lines.append(f"3. FLASHSCORE {target_season} — obecnie niedopasowani")
    lines.append("-" * 78)
    lines.append(f"Niedopasowanych po obecnym mapowaniu: {len(unmatched)}")
    for team in sorted(unmatched["flash_team"].unique()):
        cnt = (unmatched["flash_team"] == team).sum()
        lines.append(f"  {team:30s} | {cnt}")
    lines.append("")

    lines.append("4. KANDYDACI GLOBALNI DLA NIEDOPASOWANYCH")
    lines.append("-" * 78)
    if len(candidates) == 0:
        lines.append("Brak kandydatów.")
    else:
        covered = candidates["flash_name"].nunique()
        lines.append(f"Niedopasowani z co najmniej 1 kandydatem globalnym: {covered}/{len(unmatched)}")
        lines.append("")
        lines.append("Top przykłady (najlepszy kandydat per zawodnik):")
        best = candidates.sort_values(["flash_team", "flash_name", "score"], ascending=[True, True, False]) \
                         .drop_duplicates(subset=["flash_team", "flash_name"])
        for row in best.head(60).itertuples(index=False):
            lines.append(
                f"  {row.flash_team:30s} | {row.flash_name:25s} "
                f"-> {row.candidate_player_slug:35s} | {row.candidate_klub_slug:20s} "
                f"| {row.candidate_season:7s} | score={row.score:3d} | {row.relation}"
            )

    lines.append("")
    lines.append("5. WYGNEROWANE PLIKI")
    lines.append("-" * 78)
    lines.append(f"  {OUTPUT_ROSTER}")
    lines.append(f"  {OUTPUT_TRANSFERS}")
    lines.append(f"  {outputs['unmatched']}")
    lines.append(f"  {outputs['candidates']}")
    lines.append(f"  {outputs['report']}")
    lines.append("")

    return "\n".join(lines)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Audyt tożsamości zawodników i transferów per sezon")
    parser.add_argument("--season", required=True, choices=list(SEASON_TO_RAW.keys()))
    args = parser.parse_args()

    season = args.season
    outputs = get_outputs(season)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    roster = load_ekstra_roster()
    transfers = build_transfer_table(roster)
    unmatched = load_unmatched_flash(season, outputs["unmatched"])
    candidates = build_unmatched_candidates(season, roster, unmatched, outputs["candidates"])

    report = build_report(season, roster, transfers, unmatched, candidates, outputs)
    print(report)

    with open(outputs["report"], "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()