import sqlite3
import pandas as pd
import re
from pathlib import Path
from difflib import SequenceMatcher

# ──────────────────────────────────────────────────────────────────────────────
# KONFIG
# ──────────────────────────────────────────────────────────────────────────────

DB_PATH = "db/ekstraklasa.db"
RAW_ROOT = Path("data/raw/ekstraklasa_org")
MAPPING_CSV = Path("data/processed/player_mapping.csv")

PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("data/reports/player_identity")

OUTPUT_ROSTER = PROCESSED_DIR / "ekstra_player_roster_2023_2026.csv"
OUTPUT_TRANSFERS = PROCESSED_DIR / "ekstra_player_transfers_2023_2026.csv"

OUTPUT_UNMATCHED = REPORT_DIR / "flash_unmatched_2025_26.csv"
OUTPUT_CANDIDATES = REPORT_DIR / "flash_unmatched_candidates_2025_26.csv"
OUTPUT_REPORT = REPORT_DIR / "flash_identity_transfer_audit_report.txt"

TARGET_SEASON = "2025/26"

RAW_SEASONS = {
    "2023-2024": "2023/24",
    "2024-2025": "2024/25",
    "2025-2026": "2025/26",
}

TEAM_MAP = {
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
    "Radomiak Radom": "radomiak-radom",
    "Raków Częstochowa": "rakow-czestochowa",
    "Widzew Łódź": "widzew-lodz",
    "Wisła Płock": "wisla-plock",
    "Zagłębie Lubin": "zagebie-lubin",
}

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
# EKSTRAKLSA.ORG ROSTER
# ──────────────────────────────────────────────────────────────────────────────

def load_ekstra_roster() -> pd.DataFrame:
    rows = []

    for raw_season, season_label in RAW_SEASONS.items():
        folder = RAW_ROOT / raw_season
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
    roster["slug_nonfirst"] = roster["slug_tokens"].apply(lambda x: "-".join(x[1:]) if len(x) > 1 else x[0] if x else "")
    roster["slug_full_compact"] = roster["player_slug"].str.replace("-", "", regex=False)
    roster["slug_nonfirst_compact"] = roster["slug_nonfirst"].str.replace("-", "", regex=False)
    roster["slug_token_count"] = roster["slug_tokens"].apply(len)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    roster.sort_values(["season", "klub_slug", "player_slug"]).to_csv(OUTPUT_ROSTER, index=False, encoding="utf-8")
    return roster

# ──────────────────────────────────────────────────────────────────────────────
# TABELA TRANSFERÓW / ZMIAN KLUBOWYCH
# ──────────────────────────────────────────────────────────────────────────────

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

def load_unmatched_flash() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    flash = pd.read_sql_query(
        """
        SELECT DISTINCT player_name, team_name
        FROM lineups
        WHERE sezon = ?
        """,
        conn,
        params=(TARGET_SEASON,)
    )
    conn.close()

    flash["flash_name"] = flash["player_name"].apply(clean_flash_name)
    flash["flash_team"] = flash["team_name"].astype(str).str.strip()
    flash["klub_slug_expected"] = flash["flash_team"].map(TEAM_MAP)

    flash = flash[["flash_team", "klub_slug_expected", "flash_name"]].drop_duplicates()

    if MAPPING_CSV.exists():
        mapped = pd.read_csv(MAPPING_CSV)
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
    unmatched.to_csv(OUTPUT_UNMATCHED, index=False, encoding="utf-8")
    return unmatched

# ──────────────────────────────────────────────────────────────────────────────
# KANDYDACI GLOBALNI DLA NIEDOPASOWANYCH
# ──────────────────────────────────────────────────────────────────────────────

def candidate_relation(candidate_season: str, candidate_club: str, expected_club: str) -> str:
    same_season = candidate_season == TARGET_SEASON
    same_club = candidate_club == expected_club

    if same_season and same_club:
        return "same_season_same_club"
    if same_season and not same_club:
        return "same_season_other_club"
    if candidate_season < TARGET_SEASON and same_club:
        return "past_season_same_club"
    if candidate_season < TARGET_SEASON and not same_club:
        return "past_season_other_club"
    if candidate_season > TARGET_SEASON and same_club:
        return "future_season_same_club"
    return "future_season_other_club"


def score_candidate(flash_name: str, expected_club: str, row) -> tuple[int, list[str]]:
    reasons = []

    base = flash_base_name(flash_name)
    f_ascii = to_ascii(base)
    f_slug = to_slugstyle(base)
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
    slug_nonfirst_comp = row["slug_nonfirst_compact"]

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

    if row["season"] == TARGET_SEASON and score > 0:
        score += 2
        reasons.append("same_season_bonus")

    return score, reasons


def build_unmatched_candidates(roster: pd.DataFrame, unmatched: pd.DataFrame) -> pd.DataFrame:
    candidate_rows = []

    for u in unmatched.itertuples(index=False):
        flash_team = u.flash_team
        expected_club = u.klub_slug_expected
        flash_name = u.flash_name

        local_candidates = []

        for row in roster.itertuples(index=False):
            score, reasons = score_candidate(flash_name, expected_club, row._asdict())
            if score <= 0:
                continue

            relation = candidate_relation(row.season, row.klub_slug, expected_club)

            local_candidates.append({
                "flash_team": flash_team,
                "expected_klub_slug": expected_club,
                "flash_name": flash_name,
                "candidate_player_slug": row.player_slug,
                "candidate_klub_slug": row.klub_slug,
                "candidate_season": row.season,
                "relation": relation,
                "score": score,
                "reasons": " | ".join(reasons),
                "candidate_nazwa": row.nazwa,
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

    cand_df.to_csv(OUTPUT_CANDIDATES, index=False, encoding="utf-8")
    return cand_df

# ──────────────────────────────────────────────────────────────────────────────
# RAPORT
# ──────────────────────────────────────────────────────────────────────────────

def build_report(roster: pd.DataFrame, transfers: pd.DataFrame, unmatched: pd.DataFrame, candidates: pd.DataFrame) -> str:
    lines = []

    lines.append("=" * 78)
    lines.append("AUDYT TOŻSAMOŚCI ZAWODNIKÓW + TRANSFERY + KANDYDACI DLA NIEDOPASOWANYCH")
    lines.append("=" * 78)
    lines.append("")

    lines.append("1. EKSTRAKLASA.ORG — roster 3 sezony")
    lines.append("-" * 78)
    lines.append(f"Liczba unikalnych rekordów sezon-zawodnik-klub: {len(roster)}")
    for season in sorted(roster["season"].unique()):
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

    lines.append("3. FLASHSCORE 2025/26 — obecnie niedopasowani")
    lines.append("-" * 78)
    lines.append(f"Niedopasowanych po obecnym player_mapping.csv: {len(unmatched)}")
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
    lines.append(f"  {OUTPUT_UNMATCHED}")
    lines.append(f"  {OUTPUT_CANDIDATES}")
    lines.append(f"  {OUTPUT_REPORT}")
    lines.append("")

    return "\n".join(lines)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    roster = load_ekstra_roster()
    transfers = build_transfer_table(roster)
    unmatched = load_unmatched_flash()
    candidates = build_unmatched_candidates(roster, unmatched)

    report = build_report(roster, transfers, unmatched, candidates)
    print(report)

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()