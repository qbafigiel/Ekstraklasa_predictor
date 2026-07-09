import sqlite3
import pandas as pd
import re
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# KONFIG
# ──────────────────────────────────────────────────────────────────────────────

DB_PATH = "db/ekstraklasa.db"

ROSTER_CSV = Path("data/processed/ekstra_player_roster_2023_2026.csv")
MAPPING_CSV = Path("data/processed/player_mapping.csv")

REPORT_DIR = Path("data/reports/player_identity")

OUTPUT_REVIEW = REPORT_DIR / "player_identity_review_2025_26.csv"
OUTPUT_CANDIDATES = REPORT_DIR / "player_identity_review_candidates_2025_26.csv"
OUTPUT_REPORT = REPORT_DIR / "player_identity_review_report_2025_26.txt"

TARGET_SEASON = "2025/26"

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

MANUAL_NO_DATA = {
    ("Guilherme", "Radomiak Radom"),
    ("Costa S. T.", "Jagiellonia Białystok"),
    ("Dieguez A.", "Jagiellonia Białystok"),
    ("Hirosawa T.", "Jagiellonia Białystok"),
    ("Gual M.", "Legia Warszawa"),
    ("Diaz J.", "Raków Częstochowa"),
    ("Silva J. C.", "Raków Częstochowa"),
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

def to_ascii(s: str) -> str:
    s = str(s).translate(PL_TO_ASCII).lower()
    s = re.sub(r"[^a-z0-9\s\-\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def compact(s: str) -> str:
    return re.sub(r"[\s\-\.]", "", to_ascii(s))

def tokenize_text(s: str) -> list[str]:
    return [t for t in re.split(r"[\s\-]+", to_ascii(s)) if t]

def clean_flash_name(raw: str) -> str:
    raw = str(raw).strip()
    raw = re.sub(r"\s+\d+$", "", raw)
    return raw.strip()

# ──────────────────────────────────────────────────────────────────────────────
# PARSOWANIE FLASHSCORE
# ──────────────────────────────────────────────────────────────────────────────

def parse_flash_identity(name: str) -> dict:
    raw = clean_flash_name(name)

    initials = []
    abbrev = ""

    m = re.search(r"((?:\s+[A-ZŁŚÓŹĆĄĘŃŻ]\.)+)\s*$", raw)
    if m:
        initials = [x.lower() for x in re.findall(r"([A-ZŁŚÓŹĆĄĘŃŻ])\.", m.group(1))]
        base = raw[:m.start()].strip()
    else:
        base = raw
        m2 = re.search(r"\s+([A-ZŁŚÓŹĆĄĘŃŻ][a-ząćęłńóśźż]{1,4})\.\s*$", base)
        if m2:
            abbrev = to_ascii(m2.group(1))
            base = base[:m2.start()].strip()

    tokens = tokenize_text(base)

    return {
        "raw_name": raw,
        "base_name": base,
        "tokens": tokens,
        "initials": initials,
        "abbrev": abbrev,
    }

# ──────────────────────────────────────────────────────────────────────────────
# ROSTER / REGISTRY
# ──────────────────────────────────────────────────────────────────────────────

def load_roster() -> pd.DataFrame:
    if not ROSTER_CSV.exists():
        raise FileNotFoundError(
            f"Brak pliku {ROSTER_CSV}. Najpierw uruchom:\n"
            f"python scripts/audit/audit_player_identity_transfers.py"
        )

    df = pd.read_csv(ROSTER_CSV)
    needed = {"season", "player_slug", "klub_slug"}
    if not needed.issubset(df.columns):
        raise RuntimeError(f"{ROSTER_CSV} nie ma wymaganych kolumn: {needed}")

    df["season"] = df["season"].astype(str).str.strip()
    df["player_slug"] = df["player_slug"].astype(str).str.strip()
    df["klub_slug"] = df["klub_slug"].astype(str).str.strip()
    df["nazwa"] = df.get("nazwa", "").astype(str)

    return df

def build_registry(roster: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for player_slug, g in roster.groupby("player_slug"):
        clubs_all = sorted(set(g["klub_slug"]))
        seasons_all = sorted(set(g["season"]))
        clubs_target = sorted(set(g.loc[g["season"] == TARGET_SEASON, "klub_slug"]))

        slug_tokens = [t for t in player_slug.split("-") if t]
        slug_tokens_ascii = [to_ascii(t) for t in slug_tokens if t]
        slug_compact = compact(player_slug.replace("-", " "))

        season_club_path = " | ".join(
            f"{row.season}:{row.klub_slug}"
            for row in g.sort_values(["season", "klub_slug"]).itertuples(index=False)
        )

        rows.append({
            "player_slug": player_slug,
            "slug_tokens_ascii": slug_tokens_ascii,
            "slug_compact": slug_compact,
            "clubs_all": clubs_all,
            "clubs_target": clubs_target,
            "seasons_all": seasons_all,
            "season_club_path": season_club_path,
            "sample_name": g["nazwa"].iloc[0] if len(g) else "",
        })

    reg = pd.DataFrame(rows)
    return reg

# ──────────────────────────────────────────────────────────────────────────────
# OBECNE MAPOWANIA
# ──────────────────────────────────────────────────────────────────────────────

def load_existing_mapping() -> dict:
    if not MAPPING_CSV.exists():
        return {}

    df = pd.read_csv(MAPPING_CSV)
    if len(df) == 0:
        return {}

    df["flash_team"] = df["flash_team"].astype(str).str.strip()
    df["flash_name"] = df["flash_name"].astype(str).apply(clean_flash_name)
    df["player_slug"] = df["player_slug"].astype(str).str.strip()

    return {
        (row.flash_team, row.flash_name): row.player_slug
        for row in df.itertuples(index=False)
    }

# ──────────────────────────────────────────────────────────────────────────────
# FLASHSCORE UNIQUE PLAYERS
# ──────────────────────────────────────────────────────────────────────────────

def load_flash_unique_players() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT DISTINCT player_name, team_name
        FROM lineups
        WHERE sezon = ?
        """,
        conn,
        params=(TARGET_SEASON,)
    )
    conn.close()

    df["flash_name"] = df["player_name"].apply(clean_flash_name)
    df["flash_team"] = df["team_name"].astype(str).str.strip()
    df["expected_klub_slug"] = df["flash_team"].map(TEAM_MAP)

    df = df[["flash_team", "expected_klub_slug", "flash_name"]].drop_duplicates()
    df = df.sort_values(["flash_team", "flash_name"]).reset_index(drop=True)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# SCORING ŚCISŁY — BEZ FUZZY
# ──────────────────────────────────────────────────────────────────────────────

def any_prefix_match(prefixes: list[str], tokens: list[str]) -> bool:
    for p in prefixes:
        if not p:
            continue
        for t in tokens:
            if t.startswith(p):
                return True
    return False

def score_candidate(flash: dict, expected_club: str, cand: dict) -> tuple[int, list[str], dict]:
    flash_tokens = flash["tokens"]
    initials = flash["initials"]
    abbrev = flash["abbrev"]

    cand_tokens = cand["slug_tokens_ascii"]
    clubs_all = cand["clubs_all"]
    clubs_target = cand["clubs_target"]
    seasons_all = cand["seasons_all"]

    if not flash_tokens:
        return 0, [], {}

    flash_set = set(flash_tokens)
    cand_set = set(cand_tokens)

    all_tokens_in_slug = all(t in cand_set for t in flash_tokens)

    suffix_exact = False
    if len(flash_tokens) <= len(cand_tokens):
        suffix_exact = (cand_tokens[-len(flash_tokens):] == flash_tokens)

    remaining_tokens = cand_tokens.copy()
    for ft in flash_tokens:
        if ft in remaining_tokens:
            remaining_tokens.remove(ft)

    initial_match = any_prefix_match(initials, remaining_tokens) if initials else False
    abbrev_match = any_prefix_match([abbrev], remaining_tokens) if abbrev else False

    same_club_any = expected_club in clubs_all if expected_club else False
    same_club_target = expected_club in clubs_target if expected_club else False
    target_season_present = TARGET_SEASON in seasons_all

    score = 0
    reasons = []

    if all_tokens_in_slug:
        score += 70
        reasons.append("all_flash_tokens_in_slug")

    if suffix_exact:
        score += 20
        reasons.append("suffix_exact")

    if initial_match:
        score += 15
        reasons.append("initial_match_any_given_token")

    if abbrev_match:
        score += 15
        reasons.append(f"abbrev_match:{abbrev}")

    if same_club_any:
        score += 12
        reasons.append("same_club_history")

    if same_club_target:
        score += 8
        reasons.append("same_club_target_season")
    elif target_season_present:
        score += 4
        reasons.append("present_in_target_season")

    support = suffix_exact or initial_match or abbrev_match or same_club_any

    if not all_tokens_in_slug or not support:
        return 0, [], {}

    features = {
        "all_tokens_in_slug": all_tokens_in_slug,
        "suffix_exact": suffix_exact,
        "initial_match": initial_match,
        "abbrev_match": abbrev_match,
        "same_club_any": same_club_any,
        "same_club_target": same_club_target,
        "target_season_present": target_season_present,
    }

    return score, reasons, features

# ──────────────────────────────────────────────────────────────────────────────
# GŁÓWNY REVIEW BUILD
# ──────────────────────────────────────────────────────────────────────────────

def build_review():
    roster = load_roster()
    registry = build_registry(roster)
    mapped_dict = load_existing_mapping()
    flash_df = load_flash_unique_players()

    review_rows = []
    candidate_rows = []

    reg_records = registry.to_dict("records")

    for row in flash_df.itertuples(index=False):
        flash_team = row.flash_team
        expected_club = row.expected_klub_slug
        flash_name = row.flash_name

        key = (flash_team, flash_name)

        if key in mapped_dict:
            review_rows.append({
                "flash_team": flash_team,
                "flash_name": flash_name,
                "expected_klub_slug": expected_club,
                "current_status": "matched_existing",
                "existing_player_slug": mapped_dict[key],
                "proposed_status": "matched_existing",
                "proposed_player_slug": mapped_dict[key],
                "evidence": "already_in_player_mapping_csv",
                "candidate_count": 0,
            })
            continue

        if key in MANUAL_NO_DATA:
            review_rows.append({
                "flash_team": flash_team,
                "flash_name": flash_name,
                "expected_klub_slug": expected_club,
                "current_status": "manual_no_data",
                "existing_player_slug": "",
                "proposed_status": "manual_no_data",
                "proposed_player_slug": "",
                "evidence": "known_no_data_in_ekstraklasa_org",
                "candidate_count": 0,
            })
            continue

        flash_parsed = parse_flash_identity(flash_name)
        candidates = []

        for cand in reg_records:
            score, reasons, features = score_candidate(
                flash=flash_parsed,
                expected_club=expected_club,
                cand=cand,
            )
            if score <= 0:
                continue

            relation = (
                "same_club_target_season" if features.get("same_club_target") else
                "same_club_history" if features.get("same_club_any") else
                "other_club_target_season" if features.get("target_season_present") else
                "historical_other_club"
            )

            candidates.append({
                "flash_team": flash_team,
                "flash_name": flash_name,
                "expected_klub_slug": expected_club,
                "candidate_player_slug": cand["player_slug"],
                "candidate_clubs_all": " | ".join(cand["clubs_all"]),
                "candidate_clubs_target": " | ".join(cand["clubs_target"]),
                "candidate_seasons_all": " | ".join(cand["seasons_all"]),
                "season_club_path": cand["season_club_path"],
                "score": score,
                "relation": relation,
                "reasons": " | ".join(reasons),
                "sample_name": cand["sample_name"],
                "same_club_any": int(features.get("same_club_any", False)),
                "same_club_target": int(features.get("same_club_target", False)),
                "target_season_present": int(features.get("target_season_present", False)),
                "suffix_exact": int(features.get("suffix_exact", False)),
                "initial_match": int(features.get("initial_match", False)),
                "abbrev_match": int(features.get("abbrev_match", False)),
            })

        candidates = sorted(
            candidates,
            key=lambda x: (
                -x["score"],
                -x["same_club_target"],
                -x["same_club_any"],
                -x["target_season_present"],
                x["candidate_player_slug"]
            )
        )

        top_candidates = candidates[:5]
        candidate_rows.extend(top_candidates)

        if not candidates:
            review_rows.append({
                "flash_team": flash_team,
                "flash_name": flash_name,
                "expected_klub_slug": expected_club,
                "current_status": "unresolved",
                "existing_player_slug": "",
                "proposed_status": "no_candidate",
                "proposed_player_slug": "",
                "evidence": "no_strict_candidate",
                "candidate_count": 0,
            })
            continue

        top1 = candidates[0]
        top2 = candidates[1] if len(candidates) > 1 else None
        gap = top1["score"] - top2["score"] if top2 else 999

        if top1["score"] >= 100 and gap >= 8 and top1["same_club_any"] == 1:
            proposed_status = "auto_same_club_history"
        elif top1["score"] >= 95 and gap >= 10 and top1["same_club_any"] == 0:
            proposed_status = "auto_transfer_candidate"
        elif top1["score"] >= 92 and gap >= 8:
            proposed_status = "review_high_confidence"
        else:
            proposed_status = "review_needed"

        review_rows.append({
            "flash_team": flash_team,
            "flash_name": flash_name,
            "expected_klub_slug": expected_club,
            "current_status": "unresolved",
            "existing_player_slug": "",
            "proposed_status": proposed_status,
            "proposed_player_slug": top1["candidate_player_slug"],
            "evidence": top1["reasons"],
            "candidate_count": len(candidates),
            "top1_score": top1["score"],
            "top1_relation": top1["relation"],
            "top1_clubs": top1["candidate_clubs_all"],
            "top1_seasons": top1["candidate_seasons_all"],
            "top1_path": top1["season_club_path"],
            "top2_player_slug": top2["candidate_player_slug"] if top2 else "",
            "top2_score": top2["score"] if top2 else "",
            "gap_top1_top2": gap,
        })

    review_df = pd.DataFrame(review_rows)
    cand_df = pd.DataFrame(candidate_rows)

    review_df = review_df.sort_values(["flash_team", "flash_name"]).reset_index(drop=True)
    if len(cand_df):
        cand_df = cand_df.sort_values(["flash_team", "flash_name", "score"], ascending=[True, True, False]).reset_index(drop=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(OUTPUT_REVIEW, index=False, encoding="utf-8")
    cand_df.to_csv(OUTPUT_CANDIDATES, index=False, encoding="utf-8")

    return review_df, cand_df

# ──────────────────────────────────────────────────────────────────────────────
# RAPORT
# ──────────────────────────────────────────────────────────────────────────────

def build_report(review_df: pd.DataFrame, cand_df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("PLAYER IDENTITY REVIEW 2025/26 — STRICT / TRANSFER-AWARE / NO FUZZY")
    lines.append("=" * 78)
    lines.append("")

    lines.append("1. Statusy ogółem")
    lines.append("-" * 78)
    vc = review_df["proposed_status"].value_counts()
    for status, cnt in vc.items():
        lines.append(f"  {status:28s} | {cnt}")
    lines.append("")

    unresolved = review_df[review_df["current_status"] == "unresolved"]
    lines.append("2. Tylko unresolved")
    lines.append("-" * 78)
    lines.append(f"  unresolved łącznie: {len(unresolved)}")
    for status, cnt in unresolved["proposed_status"].value_counts().items():
        lines.append(f"  {status:28s} | {cnt}")
    lines.append("")

    lines.append("3. Auto-propozycje SAME CLUB HISTORY")
    lines.append("-" * 78)
    auto_same = review_df[review_df["proposed_status"] == "auto_same_club_history"]
    for row in auto_same.head(80).itertuples(index=False):
        lines.append(
            f"  {row.flash_team:30s} | {row.flash_name:25s} -> {row.proposed_player_slug}"
        )
    lines.append("")

    lines.append("4. Auto-propozycje TRANSFER")
    lines.append("-" * 78)
    auto_transfer = review_df[review_df["proposed_status"] == "auto_transfer_candidate"]
    for row in auto_transfer.head(80).itertuples(index=False):
        lines.append(
            f"  {row.flash_team:30s} | {row.flash_name:25s} -> {row.proposed_player_slug}"
        )
    lines.append("")

    lines.append("5. Wysoki review")
    lines.append("-" * 78)
    high = review_df[review_df["proposed_status"] == "review_high_confidence"]
    for row in high.head(80).itertuples(index=False):
        lines.append(
            f"  {row.flash_team:30s} | {row.flash_name:25s} -> {row.proposed_player_slug:35s} | {row.evidence}"
        )
    lines.append("")

    lines.append("6. Brak kandydata")
    lines.append("-" * 78)
    no_cand = review_df[review_df["proposed_status"] == "no_candidate"]
    for row in no_cand.head(80).itertuples(index=False):
        lines.append(f"  {row.flash_team:30s} | {row.flash_name}")
    lines.append("")

    lines.append("7. Pliki")
    lines.append("-" * 78)
    lines.append(f"  {OUTPUT_REVIEW}")
    lines.append(f"  {OUTPUT_CANDIDATES}")
    lines.append(f"  {OUTPUT_REPORT}")
    lines.append("")

    return "\n".join(lines)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    review_df, cand_df = build_review()
    report = build_report(review_df, cand_df)
    print(report)

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)

if __name__ == "__main__":
    main()