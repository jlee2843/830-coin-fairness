import base64
import calendar
import subprocess
import tempfile
from datetime import datetime
from html import escape
from io import StringIO
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Coin Fairness Dashboard", page_icon="🪙", layout="wide")

st.markdown(
    """
    <style>
    .run-summary {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 12px 0 24px;
    }
    .run-summary-item,
    .assigned-run-panel,
    .flip-panel {
        border: 1px solid rgba(127, 127, 127, 0.22);
        border-radius: 8px;
        background: rgba(127, 127, 127, 0.06);
    }
    .run-summary-item {
        padding: 12px 14px;
    }
    .run-summary-label,
    .assigned-label,
    .flip-label {
        color: rgba(127, 127, 127, 0.95);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .run-summary-value {
        margin-top: 4px;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .assigned-run-panel,
    .flip-panel {
        padding: 16px 18px;
        margin: 8px 0 22px;
    }
    .assigned-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 16px 20px;
    }
    .assigned-value {
        margin-top: 4px;
        font-size: 1.05rem;
        font-weight: 700;
    }
    .flip-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 12px;
    }
    .flip-stat {
        padding: 10px 12px;
        border-radius: 8px;
        background: rgba(127, 127, 127, 0.08);
    }
    .flip-value {
        margin-top: 2px;
        font-size: 1.55rem;
        font-weight: 700;
        line-height: 1.15;
    }
    .flip-history {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin: 10px 0 2px;
    }
    .flip-pill {
        border-radius: 999px;
        padding: 4px 9px;
        font-size: 0.82rem;
        font-weight: 700;
        border: 1px solid rgba(127, 127, 127, 0.22);
    }
    .flip-pill-heads {
        background: rgba(46, 160, 67, 0.18);
    }
    .flip-pill-tails {
        background: rgba(59, 130, 246, 0.18);
    }
    .calendar-grid {
        display: grid;
        grid-template-columns: repeat(7, minmax(0, 1fr));
        border: 1px solid rgba(127, 127, 127, 0.22);
        border-radius: 8px;
        overflow: hidden;
        margin-top: 12px;
    }
    .calendar-weekday {
        padding: 8px 10px;
        background: rgba(127, 127, 127, 0.12);
        color: rgba(127, 127, 127, 0.95);
        font-size: 0.76rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.02em;
        border-right: 1px solid rgba(127, 127, 127, 0.18);
    }
    .calendar-day {
        position: relative;
        min-height: 96px;
        padding: 8px;
        border-top: 1px solid rgba(127, 127, 127, 0.18);
        border-right: 1px solid rgba(127, 127, 127, 0.18);
        background: rgba(127, 127, 127, 0.035);
    }
    .calendar-day-muted {
        color: rgba(127, 127, 127, 0.55);
        background: rgba(127, 127, 127, 0.015);
    }
    .calendar-day-today {
        box-shadow: inset 0 0 0 2px rgba(59, 130, 246, 0.65);
    }
    .calendar-day-number {
        font-size: 0.85rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .calendar-event {
        display: block;
        margin-top: 4px;
        padding: 4px 6px;
        border-radius: 6px;
        font-size: 0.76rem;
        font-weight: 700;
        line-height: 1.2;
        overflow-wrap: anywhere;
    }
    .calendar-event-due {
        background: rgba(239, 68, 68, 0.22);
        border: 1px solid rgba(239, 68, 68, 0.38);
    }
    .calendar-event-meeting {
        background: rgba(59, 130, 246, 0.20);
        border: 1px solid rgba(59, 130, 246, 0.35);
    }
    .calendar-event-consultation {
        background: rgba(46, 160, 67, 0.20);
        border: 1px solid rgba(46, 160, 67, 0.35);
    }
    @media (max-width: 760px) {
        .run-summary,
        .assigned-grid,
        .flip-grid {
            grid-template-columns: 1fr;
        }
        .calendar-day {
            min-height: 76px;
            padding: 6px;
        }
        .calendar-event {
            font-size: 0.68rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("STAT 830 Project: Coin Fairness")
st.write("Submit coin flip results and view the shared experiment results. One run is 10 flips.")

FLIPS_PER_TRIAL = 10
RUN_SCHEDULE_PATH = Path("data/run_schedule.csv")
# Touch this comment when Streamlit Cloud needs a fresh deploy trigger.
GITHUB_CACHE_TTL_SECONDS = 60
DATA_TOOLS_KEY = "show_submit_data_tools"
R_PLOT_RENDER_VERSION = 5

HELD_CONSTANTS = [
    "Sitting height / chair height",
    "Location",
    "Floor type: carpet",
    "Experiment room",
]

ALLOWED_TO_VARY = [
    "Room temperature",
    "Room light",
    "Number of people in the room",
    "Noise in the room",
    "Humidity",
    "Mood of the flipper",
]

RESTRICTIONS = [
    "Must flip on thumb and use the index finger as leverage.",
    "Coin must land on the floor properly. If it does not land on the floor and lands on a body instead, re-run it.",
]

cols = [
    "trial_id", "denomination", "decade",
    "posture", "flipper", "starting_side", "heads"
]

schedule_cols = [
    "run_id", "replication", "denomination", "decade",
    "posture", "flipper", "starting_side"
]

assignment_cols = [
    "denomination", "decade", "posture", "flipper", "starting_side"
]

denominations = ["Nickel", "Dime", "Quarter"]
decades = ["1977", "2019"]
postures = ["Standing", "Sitting"]
flippers = ["Jenny", "Josh", "Esther"]
starting_sides = ["Heads", "Tails"]

DECADE_VALUE_MAP = {
    "1980s": "1977",
    "1980": "1977",
    "2010s": "2019",
    "2010": "2019",
}

github_token = st.secrets.get("GITHUB_TOKEN", "")
github_repo = st.secrets.get("GITHUB_REPO", "")
github_path = st.secrets.get("GITHUB_PATH", "data/coin_experiment_data.csv")
calendar_github_path = st.secrets.get("CALENDAR_GITHUB_PATH", "data/calendar_events.csv")
notes_github_path = st.secrets.get("NOTES_GITHUB_PATH", "data/experiment_notes.csv")
github_branch = st.secrets.get("GITHUB_BRANCH", "main")
delete_password = st.secrets.get("DELETE_PASSWORD", "")


def empty_data():
    return pd.DataFrame(columns=cols)


def empty_calendar_data():
    return pd.DataFrame(columns=["date", "time", "type", "title"])


def empty_notes_data():
    return pd.DataFrame(columns=["note_id", "created_at", "author", "type", "note"])


def clean_calendar_data(df):
    df = df.copy()

    for c in ["date", "time", "type", "title"]:
        if c not in df.columns:
            df[c] = ""

    df = df[["date", "time", "type", "title"]]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["time"] = df["time"].astype(str).str.strip().replace({"nan": ""})
    df["type"] = df["type"].astype(str).str.strip().replace({"nan": ""})
    df.loc[~df["type"].isin(["Due", "Meeting", "Consultation"]), "type"] = "Meeting"
    df["title"] = df["title"].astype(str).str.strip()
    df = df.dropna(subset=["date"])
    df = df[df["title"] != ""]
    df["date"] = df["date"].astype(str)

    return df.sort_values(["date", "time", "type", "title"]).reset_index(drop=True)


def clean_notes_data(df):
    df = df.copy()

    for c in ["note_id", "created_at", "author", "type", "note"]:
        if c not in df.columns:
            df[c] = ""

    df = df[["note_id", "created_at", "author", "type", "note"]].copy()

    for c in ["note_id", "created_at", "author", "type", "note"]:
        df[c] = df[c].astype(str).str.strip().replace({"nan": ""})

    df.loc[df["type"] == "", "type"] = "Note"
    df.loc[~df["type"].isin(["Note", "Comment", "Question", "Issue"]), "type"] = "Note"
    df = df[df["note"] != ""].reset_index(drop=True)

    if len(df) > 0:
        missing_note_id = df["note_id"] == ""
        df.loc[missing_note_id, "note_id"] = [
            f"note-{i + 1}" for i in range(int(missing_note_id.sum()))
        ]

    return df.sort_values("created_at", ascending=False).reset_index(drop=True)


def load_run_schedule():
    if not RUN_SCHEDULE_PATH.exists():
        return pd.DataFrame(columns=schedule_cols)

    schedule = pd.read_csv(RUN_SCHEDULE_PATH)

    for c in schedule_cols:
        if c not in schedule.columns:
            schedule[c] = np.nan

    schedule = schedule[schedule_cols].copy()
    schedule["run_id"] = pd.to_numeric(schedule["run_id"], errors="coerce")
    schedule["replication"] = pd.to_numeric(schedule["replication"], errors="coerce")
    schedule = schedule.dropna(subset=["run_id"])

    if len(schedule) > 0:
        schedule["run_id"] = schedule["run_id"].astype(int)
        schedule["replication"] = schedule["replication"].astype("Int64")

    for c in ["denomination", "decade", "posture", "flipper", "starting_side"]:
        schedule[c] = schedule[c].astype(str).str.strip().replace({"nan": ""})

    schedule["decade"] = schedule["decade"].replace(DECADE_VALUE_MAP)

    return schedule.sort_values("run_id").reset_index(drop=True)


def next_run_id_from_valid_data(valid_df, schedule):
    if len(schedule) == 0:
        return None

    if len(valid_df) == 0:
        return int(schedule["run_id"].min())

    next_run_id = int(valid_df["trial_id"].max()) + 1

    if next_run_id > int(schedule["run_id"].max()):
        return None

    return next_run_id


def next_run_id_from_data(df, schedule):
    return next_run_id_from_valid_data(valid_submitted_data(df, schedule), schedule)


def scheduled_run(schedule, run_id):
    if run_id is None or len(schedule) == 0:
        return None

    matches = schedule[schedule["run_id"] == run_id]

    if len(matches) == 0:
        return None

    return matches.iloc[0]


def schedule_progress_from_valid_data(schedule, valid_df):
    if len(schedule) == 0:
        return schedule.copy()

    completed_ids = set(valid_df["trial_id"].astype(int)) if len(valid_df) > 0 else set()
    progress = schedule.copy()
    progress["complete"] = progress["run_id"].isin(completed_ids)
    progress["status"] = np.where(progress["complete"], "Complete", "Incomplete")

    return progress


def schedule_progress(schedule, df):
    return schedule_progress_from_valid_data(schedule, valid_submitted_data(df, schedule))


def mapped_valid_submitted_data(df, schedule):
    if len(df) == 0 or len(schedule) == 0:
        return df.iloc[0:0].copy()

    schedule_key = schedule[["run_id"] + assignment_cols].copy()
    submitted = df.reset_index(drop=True).copy()
    used_run_ids = set()
    used_row_indices = set()
    matched_rows = []

    def matching_schedule_rows(row, candidates):
        matches = candidates

        for c in assignment_cols:
            matches = matches[matches[c].astype(str) == str(row[c])]

        return matches

    for idx, row in submitted.iterrows():
        run_id = int(row["trial_id"])
        candidates = schedule_key[schedule_key["run_id"] == run_id]
        matches = matching_schedule_rows(row, candidates)

        if len(matches) > 0 and run_id not in used_run_ids:
            matched = row.copy()
            matched["trial_id"] = run_id
            matched["_source_trial_id"] = run_id
            matched_rows.append(matched)
            used_run_ids.add(run_id)
            used_row_indices.add(idx)

    available_schedule = schedule_key[~schedule_key["run_id"].isin(used_run_ids)]

    for idx, row in submitted.iterrows():
        if idx in used_row_indices:
            continue

        matches = matching_schedule_rows(row, available_schedule)

        if len(matches) == 0:
            continue

        matched_run_id = int(matches.sort_values("run_id").iloc[0]["run_id"])
        matched = row.copy()
        matched["trial_id"] = matched_run_id
        matched["_source_trial_id"] = int(row["trial_id"])
        matched_rows.append(matched)
        used_run_ids.add(matched_run_id)
        used_row_indices.add(idx)
        available_schedule = available_schedule[
            available_schedule["run_id"] != matched_run_id
        ]

    if not matched_rows:
        return df.iloc[0:0].copy()

    valid_df = pd.DataFrame(matched_rows)
    valid_df["trial_id"] = valid_df["trial_id"].astype(int)
    valid_df["_source_trial_id"] = valid_df["_source_trial_id"].astype(int)

    return valid_df.sort_values("trial_id").reset_index(drop=True)


def valid_submitted_data(df, schedule):
    valid_df = mapped_valid_submitted_data(df, schedule)

    if len(valid_df) == 0:
        return df.iloc[0:0].copy()

    return clean_data(valid_df.drop(columns=["_source_trial_id"], errors="ignore"))


def clean_data(df):
    df = df.copy()

    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[cols]

    for c in ["denomination", "decade", "posture", "flipper", "starting_side"]:
        df[c] = df[c].astype(str).str.strip().replace({"nan": ""})

    df["decade"] = df["decade"].replace(DECADE_VALUE_MAP)

    df["trial_id"] = pd.to_numeric(df["trial_id"], errors="coerce")
    df["heads"] = pd.to_numeric(df["heads"], errors="coerce")

    df = df.dropna(subset=["trial_id", "heads"])

    if len(df) > 0:
        df["trial_id"] = df["trial_id"].astype(int)
        df["heads"] = df["heads"].astype(int)

    df = df[(df["heads"] >= 0) & (df["heads"] <= FLIPS_PER_TRIAL)]

    df["total"] = FLIPS_PER_TRIAL
    df["tails"] = df["total"] - df["heads"]
    df["proportion"] = df["heads"] / df["total"]

    return df


def make_dummy_data():
    rows = []
    trial_id = 1
    rng = np.random.default_rng(830)

    base_p = {
        "Nickel": 0.50,
        "Dime": 0.46,
        "Quarter": 0.55
    }

    for denomination in denominations:
        for decade in decades:
            for posture in postures:
                for rep in range(4):
                    p = base_p[denomination]

                    if decade == "1977":
                        p -= 0.02
                    elif decade == "2019":
                        p += 0.02

                    if posture == "Standing":
                        p += 0.01
                    else:
                        p -= 0.01

                    p = min(max(p, 0.05), 0.95)

                    rows.append({
                        "trial_id": trial_id,
                        "denomination": denomination,
                        "decade": decade,
                        "posture": posture,
                        "flipper": flippers[rep % len(flippers)],
                        "starting_side": starting_sides[rep % len(starting_sides)],
                        "heads": int(rng.binomial(FLIPS_PER_TRIAL, p))
                    })

                    trial_id += 1

    return clean_data(pd.DataFrame(rows))


def render_calendar_html(year, month, events):
    month_calendar = calendar.Calendar(firstweekday=6)
    weeks = month_calendar.monthdatescalendar(year, month)
    today = datetime.now().date()
    event_map = {}

    for event in events:
        event_date = datetime.fromisoformat(event["date"]).date()
        event_time = str(event.get("time", "")).strip()
        event_type = str(event.get("type", "Meeting")).strip()
        event_title = str(event.get("title", "")).strip()
        event_time_label = ""

        if event_time:
            try:
                parsed_time = datetime.strptime(event_time, "%H:%M")
                event_time_label = parsed_time.strftime("%I:%M %p").lstrip("0")
            except ValueError:
                event_time_label = event_time

        event_label = f"{event_time_label} [{event_type}] {event_title}".strip()
        event_class = event_type.lower().replace(" ", "-")
        event_map.setdefault(event_date, []).append((event_label, event_class))

    parts = ['<div class="calendar-grid">']

    for day_name in ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]:
        parts.append(f'<div class="calendar-weekday">{day_name}</div>')

    for week in weeks:
        for day in week:
            classes = ["calendar-day"]

            if day.month != month:
                classes.append("calendar-day-muted")

            if day == today:
                classes.append("calendar-day-today")

            parts.append(f'<div class="{" ".join(classes)}">')
            parts.append(f'<div class="calendar-day-number">{day.day}</div>')

            for event_title, event_class in event_map.get(day, []):
                parts.append(
                    f'<span class="calendar-event calendar-event-{escape(event_class)}">'
                    f'{escape(event_title)}</span>'
                )

            parts.append("</div>")

    parts.append("</div>")
    return "".join(parts)


def r_analysis_code(csv_path):
    return f"""library(tidyverse)
library(car)
library(broom)
library(emmeans)

coin <- read_csv("{csv_path}", show_col_types = FALSE) |>
  mutate(
    trial_id = as.integer(trial_id),
    heads = as.integer(heads),
    total = {FLIPS_PER_TRIAL},
    tails = total - heads,
    proportion = heads / total,
    denomination = factor(denomination),
    posture = factor(posture),
    decade = factor(dplyr::recode(
      as.character(decade),
      "1980s" = "1977",
      "1980" = "1977",
      "2010s" = "2019",
      "2010" = "2019"
    )),
    flipper = factor(flipper),
    starting_side = factor(starting_side)
  )

design_summary <- coin |>
  group_by(denomination, posture, starting_side) |>
  summarise(
    runs = n(),
    total_heads = sum(heads),
    total_flips = sum(total),
    mean_proportion = mean(proportion),
    p_hat = total_heads / total_flips,
    .groups = "drop"
  )

block_summary <- coin |>
  group_by(decade, flipper) |>
  summarise(
    runs = n(),
    total_heads = sum(heads),
    total_flips = sum(total),
    mean_proportion = mean(proportion),
    p_hat = total_heads / total_flips,
    .groups = "drop"
  )

cat("Design-factor summary\\n")
print(design_summary)
cat("\\nBlock-factor summary\\n")
print(block_summary)

alpha <- 0.05
options(contrasts = c("contr.sum", "contr.poly"))

full_formula <- proportion ~
  denomination + posture + starting_side + decade + flipper +
  denomination:posture +
  denomination:starting_side +
  denomination:decade +
  denomination:flipper +
  posture:starting_side +
  posture:decade +
  posture:flipper +
  starting_side:decade +
  starting_side:flipper +
  decade:flipper +
  denomination:posture:starting_side +
  denomination:posture:decade +
  denomination:posture:flipper +
  denomination:starting_side:decade +
  denomination:starting_side:flipper +
  denomination:decade:flipper +
  posture:starting_side:decade +
  posture:starting_side:flipper +
  posture:decade:flipper +
  starting_side:decade:flipper

full_fit <- try(
  lm(
    full_formula,
    data = coin
  ),
  silent = TRUE
)

if (inherits(full_fit, "try-error")) {{
  cat("\\nModel output unavailable with the current data.\\n")
}} else {{
  cat("\\nStep 1. Full model with interactions up to order three\\n")
  cat("Design factors: denomination, posture, and starting_side\\n")
  cat("Blocking factors: decade and flipper\\n")
  cat("Replication supplies repeated observations and is not a model term.\\n")
  cat("Four-way and five-way interactions are excluded.\\n")
  cat("Significance level:", alpha, "\\n")
  cat("Model rank:", full_fit$rank, "of", ncol(model.matrix(full_fit)), "columns\\n")
  cat("This confirms that all coefficients are independent and no terms are aliased..")


  full_anova <- try(Anova(full_fit, type = 3), silent = TRUE)

  if (inherits(full_anova, "try-error")) {{
    cat("ANOVA unavailable with the current data.\\n")
  }} else {{
    cat("\\nType III ANOVA for the full model\\n")
    print(full_anova)

    anova_table <- as.data.frame(full_anova)
    anova_table$term <- rownames(anova_table)
    p_column <- grep("^Pr\\\\(", names(anova_table), value = TRUE)[1]

    significant_terms <- anova_table |>
      filter(
        !term %in% c("(Intercept)", "Residuals"),
        !is.na(.data[[p_column]]),
        .data[[p_column]] < alpha
      ) |>
      pull(term)

    main_terms <- c("denomination", "posture", "starting_side", "decade", "flipper")
    full_terms <- attr(terms(full_fit), "term.labels")
    term_order <- function(term) {{length(strsplit(term, ":", fixed = TRUE)[[1]])}}
    two_way_terms <- full_terms[vapply(full_terms, term_order, integer(1)) == 2]
    three_way_terms <- full_terms[vapply(full_terms, term_order, integer(1)) == 3]
    interaction_terms <- c(two_way_terms, three_way_terms)
    significant_interactions <- intersect(
      interaction_terms,
      significant_terms
    )
    significant_three_way <- intersect(three_way_terms, significant_terms)
    required_two_way <- unique(unlist(lapply(
      significant_three_way,
      function(term) {{
        variables <- strsplit(term, ":", fixed = TRUE)[[1]]
        combn(
          variables,
          2,
          FUN = function(parts) paste(parts, collapse = ":")
        )
      }}
    )))
    retained_interactions <- unique(c(significant_interactions, required_two_way))
    dropped_interactions <- setdiff(interaction_terms,retained_interactions)
    reduced_terms <- c(main_terms, retained_interactions)
    reduced_fit <- lm(reformulate(reduced_terms, response = "proportion"),data = coin)

    cat("\\nReduced model: remove nonsignificant interactions together\\n")
    cat("All main effects are retained to preserve hierarchy.\\n")
    cat("Interactions retained because they are significant or required by hierarchy\\n")

    if (length(retained_interactions) == 0) {{
      cat("None.\\n")
    }} else {{
      print(retained_interactions)
    }}

    cat("Interactions removed together\\n")

    if (length(dropped_interactions) == 0) {{
      cat("None.\\n")
      selected_fit <- full_fit
    }} else {{
      print(dropped_interactions)
      cat("\\nType III ANOVA after removing nonsignificant interactions\\n")
      reduced_anova <- Anova(reduced_fit, type = 3)
      print(reduced_anova)
      cat("\\nPartial F-test: reduced model versus full model\\n")
      cat("This is an extra-sum-of-squares ANOVA for two nested models.\\n")
      cat("It compares the mean variation explained by the removed interactions with the full model's residual mean variation.\\n")
      cat("H0: all removed interaction coefficients are jointly zero.\\n")
      cat("H1: at least one removed interaction coefficient is not zero.\\n")
      reduction_comparison <- anova(reduced_fit, full_fit)
      print(reduction_comparison)
      reduction_p <- reduction_comparison[["Pr(>F)"]][2]

      if (!is.na(reduction_p) && reduction_p < alpha) {{
        selected_fit <- full_fit
        cat("Removing the interactions was significant, so the full model is retained.\\n")
      }} else {{
        selected_fit <- reduced_fit
        cat("Removing the interactions was not significant, so the reduced model is selected.\\n")
        cat("The important significant terms may stay the same; reduction removes unsupported complexity rather than trying to create new significance.\\n")
      }}
    }}

    cat("\\nSelected model formula\\n")
    print(formula(selected_fit))
    cat("\\nType III ANOVA for the selected model\\n")
    selected_anova <- Anova(selected_fit, type = 3)
    print(selected_anova)

    selected_table <- as.data.frame(selected_anova)
    selected_table$term <- rownames(selected_table)
    selected_p_column <- grep("^Pr\\\\(", names(selected_table), value = TRUE)[1]
    selected_significant_terms <- selected_table |>
      filter(
        !term %in% c("(Intercept)", "Residuals"),
        !is.na(.data[[selected_p_column]]),
        .data[[selected_p_column]] < alpha
      ) |>
      pull(term)

    cat("\\nSignificant terms in the selected model at alpha = 0.05\\n")

    if (length(selected_significant_terms) == 0) {{
      cat("None.\\n")
    }} else {{
      print(selected_significant_terms)
    }}

    design_grid <- expand_grid(
      denomination = levels(coin$denomination),
      posture = levels(coin$posture),
      starting_side = levels(coin$starting_side)
    )

    predict_average_condition <- function(fit, condition) {{
      averaging_grid <- expand_grid(
        decade = levels(coin$decade),
        flipper = levels(coin$flipper)
      )

      new_data <- crossing(condition, averaging_grid) |>
        mutate(
          denomination = factor(denomination, levels = levels(coin$denomination)),
          posture = factor(posture, levels = levels(coin$posture)),
          starting_side = factor(starting_side, levels = levels(coin$starting_side)),
          decade = factor(decade, levels = levels(coin$decade)),
          flipper = factor(flipper, levels = levels(coin$flipper))
        )

      design_matrix <- model.matrix(
        delete.response(terms(fit)),
        new_data,
        contrasts.arg = fit$contrasts
      )
      average_row <- colMeans(design_matrix)
      beta <- coef(fit)
      average_row <- average_row[names(beta)]
      estimate <- drop(average_row %*% beta)
      mean_se <- sqrt(drop(average_row %*% vcov(fit) %*% average_row))
      critical_value <- qt(0.975, df.residual(fit))

      tibble(
        predicted_proportion = estimate,
        expected_heads = {FLIPS_PER_TRIAL} * estimate,
        mean_ci_low_heads = {FLIPS_PER_TRIAL} * max(0, estimate - critical_value * mean_se),
        mean_ci_high_heads = {FLIPS_PER_TRIAL} * min(1, estimate + critical_value * mean_se),
        distance_from_fair = abs(estimate - 0.5)
      )
    }}

    average_predictions <- vector("list", nrow(design_grid))

    for (row_number in seq_len(nrow(design_grid))) {{
      condition <- design_grid[row_number, ]
      average_predictions[[row_number]] <- bind_cols(
        condition,
        predict_average_condition(selected_fit, condition)
      )
    }}

    average_predictions <- bind_rows(average_predictions) |>
      arrange(desc(expected_heads))

    cat("\\nPredicted heads for every design combination\\n")
    cat("Design factors: denomination, posture, and starting_side.\\n")
    cat("Decade and flipper block levels are averaged equally.\\n")
    cat("The 95 percent confidence interval describes uncertainty in the estimated average heads.\\n")
    print(average_predictions, n = Inf, width = Inf)
    cat("\\nSelected model sigma-hat:", sigma(selected_fit), "\\n")

    highest_average_design <- average_predictions |>
      slice_max(expected_heads, n = 1, with_ties = TRUE)
    fairest_average_design <- average_predictions |>
      slice_min(distance_from_fair, n = 1, with_ties = TRUE)

    cat("\\nHighest predicted design combination averaged across decades and flippers\\n")
    print(highest_average_design, n = Inf, width = Inf)
    cat("\\nFairest design combination averaged across decades and flippers\\n")
    print(fairest_average_design, n = Inf, width = Inf)
  }}
}}
"""


def r_ggplot_code(csv_path):
    return f"""library(tidyverse)
library(car)
library(broom)
library(emmeans)

posture_colors <- c(
  Sitting = "#E76F51",
  Standing = "#168AAD"
)
decade_colors <- c(
  "1977" = "#457B9D",
  "2019" = "#E9C46A"
)
flipper_colors <- c(
  Esther = "#E76F51",
  Jenny = "#2A9D8F",
  Josh = "#457B9D"
)
interaction_colors <- c("#E76F51", "#2A9D8F", "#457B9D")
design_labels <- c(
  denomination = "Denomination",
  posture = "Posture",
  starting_side = "Starting side"
)

report_theme <- theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(
      hjust = 0.5,
      face = "bold",
      size = 16,
      color = "#20242A",
      margin = margin(b = 12)
    ),
    axis.title = element_text(face = "bold", color = "#30343B"),
    axis.text = element_text(color = "#4A4F57"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    strip.background = element_rect(fill = "#F1F3F5", color = NA),
    strip.text = element_text(face = "bold", color = "#30343B"),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    plot.margin = margin(14, 18, 12, 14)
  )

coin <- read_csv("{csv_path}", show_col_types = FALSE) |>
  mutate(
    trial_id = as.integer(trial_id),
    heads = as.integer(heads),
    total = {FLIPS_PER_TRIAL},
    tails = total - heads,
    proportion = heads / total,
    denomination = factor(denomination),
    posture = factor(posture),
    decade = factor(dplyr::recode(
      as.character(decade),
      "1980s" = "1977",
      "1980" = "1977",
      "2010s" = "2019",
      "2010" = "2019"
    )),
    flipper = factor(flipper),
    starting_side = factor(starting_side)
  )

design_summary <- coin |>
  group_by(denomination, posture, starting_side) |>
  summarise(
    runs = n(),
    mean_proportion = mean(proportion),
    se = sd(proportion) / sqrt(runs),
    .groups = "drop"
  )

p_design_mean <- ggplot(design_summary, aes(x = denomination, y = mean_proportion, fill = posture)) +
  geom_col(
    position = position_dodge(width = 0.75),
    width = 0.65,
    color = "white",
    linewidth = 0.35
  ) +
  geom_errorbar(
    aes(ymin = mean_proportion - se, ymax = mean_proportion + se),
    position = position_dodge(width = 0.75),
    width = 0.18,
    linewidth = 0.55,
    color = "#30343B"
  ) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "#555B63") +
  facet_wrap(~ starting_side) +
  scale_fill_manual(values = posture_colors) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Mean proportion of heads by design combination",
    x = "Denomination",
    y = "Mean proportion of heads",
    fill = "Posture"
  ) +
  report_theme

p_design_observed <- ggplot(coin, aes(x = denomination, y = proportion, color = posture)) +
  geom_jitter(width = 0.12, height = 0, alpha = 0.55, size = 1.8) +
  geom_boxplot(
    aes(group = interaction(denomination, posture)),
    linewidth = 0.65,
    outlier.shape = NA
  ) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "#555B63") +
  facet_grid(starting_side ~ posture) +
  scale_color_manual(values = posture_colors) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Observed proportions by design combination",
    x = "Denomination",
    y = "Proportion heads"
  ) +
  report_theme +
  theme(legend.position = "none")

p_probability_distribution <- ggplot(coin, aes(x = heads, y = after_stat(count / sum(count)))) +
  geom_histogram(
    binwidth = 1,
    boundary = -0.5,
    fill = "#2A9D8F",
    color = "white",
    linewidth = 0.45
  ) +
  scale_x_continuous(breaks = 0:{FLIPS_PER_TRIAL}, limits = c(-0.5, {FLIPS_PER_TRIAL} + 0.5)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Observed probability distribution of heads",
    x = "Heads out of 10",
    y = "Observed probability"
  ) +
  report_theme

binomial_x <- 0:{FLIPS_PER_TRIAL}
fair_binomial_p <- 0.5
observed_binomial_p <- sum(coin$heads) / sum(coin$total)

binomial_probability <- function(x, n, p) {{
  (
    factorial(n) /
    (factorial(n - x) * factorial(x))
  ) * p^x * (1 - p)^(n - x)
}}

binomial_overlay <- tibble(
  heads = binomial_x,
  observed_runs = tabulate(
    coin$heads + 1,
    nbins = {FLIPS_PER_TRIAL + 1}
  ),
  `Fair-coin model` = nrow(coin) * binomial_probability(
    binomial_x,
    {FLIPS_PER_TRIAL},
    fair_binomial_p
  ),
  `Observed-data model` = nrow(coin) * binomial_probability(
    binomial_x,
    {FLIPS_PER_TRIAL},
    observed_binomial_p
  )
)

binomial_expected <- binomial_overlay |>
  select(heads, `Fair-coin model`, `Observed-data model`) |>
  pivot_longer(
    cols = -heads,
    names_to = "distribution",
    values_to = "expected_runs"
  )

p_binomial_overlay <- ggplot() +
  geom_col(
    data = binomial_overlay,
    aes(x = heads, y = observed_runs),
    width = 0.78,
    fill = "#2A9D8F",
    color = "white",
    linewidth = 0.45,
    alpha = 0.72
  ) +
  geom_line(
    data = binomial_expected,
    aes(
      x = heads,
      y = expected_runs,
      color = distribution,
      group = distribution
    ),
    linewidth = 1.15
  ) +
  geom_point(
    data = binomial_expected,
    aes(x = heads, y = expected_runs, color = distribution),
    size = 2.4
  ) +
  scale_x_continuous(
    breaks = 0:{FLIPS_PER_TRIAL},
    limits = c(-0.5, {FLIPS_PER_TRIAL} + 0.5)
  ) +
  scale_color_manual(
    values = c(
      "Fair-coin model" = "#457B9D",
      "Observed-data model" = "#E76F51"
    )
  ) +
  labs(
    title = "Observed heads with binomial overlays",
    x = "Heads out of 10",
    y = "Number of runs",
    color = NULL
  ) +
  report_theme

p_decade <- ggplot(coin, aes(x = decade, y = proportion, fill = decade)) +
  geom_boxplot(alpha = 0.72, outlier.shape = NA, width = 0.56) +
  geom_jitter(width = 0.12, alpha = 0.48, size = 1.7) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "#555B63") +
  scale_fill_manual(values = decade_colors) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Block check by coin decade",
    x = "Decade",
    y = "Proportion heads"
  ) +
  report_theme +
  theme(legend.position = "none")

p_flipper <- ggplot(coin, aes(x = flipper, y = proportion, fill = flipper)) +
  geom_boxplot(alpha = 0.72, outlier.shape = NA, width = 0.56) +
  geom_jitter(width = 0.12, alpha = 0.48, size = 1.7) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "#555B63") +
  scale_fill_manual(values = flipper_colors) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Block check by flipper",
    x = "Flipper",
    y = "Proportion heads"
  ) +
  report_theme +
  theme(legend.position = "none")

alpha <- 0.05
options(contrasts = c("contr.sum", "contr.poly"))

full_formula <- proportion ~
  denomination + posture + starting_side + decade + flipper +
  denomination:posture +
  denomination:starting_side +
  denomination:decade +
  denomination:flipper +
  posture:starting_side +
  posture:decade +
  posture:flipper +
  starting_side:decade +
  starting_side:flipper +
  decade:flipper +
  denomination:posture:starting_side +
  denomination:posture:decade +
  denomination:posture:flipper +
  denomination:starting_side:decade +
  denomination:starting_side:flipper +
  denomination:decade:flipper +
  posture:starting_side:decade +
  posture:starting_side:flipper +
  posture:decade:flipper +
  starting_side:decade:flipper
full_fit <- lm(full_formula, data = coin)
full_anova <- Anova(full_fit, type = 3)
anova_table <- as.data.frame(full_anova)
anova_table$term <- rownames(anova_table)
p_column <- grep("^Pr\\\\(", names(anova_table), value = TRUE)[1]
significant_terms <- anova_table |>
  filter(
    !term %in% c("(Intercept)", "Residuals"),
    !is.na(.data[[p_column]]),
    .data[[p_column]] < alpha
  ) |>
  pull(term)

main_terms <- c(
  "denomination", "posture", "starting_side", "decade", "flipper"
)
full_terms <- attr(terms(full_fit), "term.labels")
term_order <- function(term) {{
  length(strsplit(term, ":", fixed = TRUE)[[1]])
}}
two_way_terms <- full_terms[
  vapply(full_terms, term_order, integer(1)) == 2
]
three_way_terms <- full_terms[
  vapply(full_terms, term_order, integer(1)) == 3
]
interaction_terms <- c(two_way_terms, three_way_terms)
significant_interactions <- intersect(
  interaction_terms,
  significant_terms
)
significant_three_way <- intersect(three_way_terms, significant_terms)
required_two_way <- unique(unlist(lapply(
  significant_three_way,
  function(term) {{
    variables <- strsplit(term, ":", fixed = TRUE)[[1]]
    combn(
      variables,
      2,
      FUN = function(parts) paste(parts, collapse = ":")
    )
  }}
)))
retained_interactions <- unique(c(
  significant_interactions,
  required_two_way
))
dropped_interactions <- setdiff(
  interaction_terms,
  retained_interactions
)
reduced_terms <- c(main_terms, retained_interactions)
reduced_fit <- lm(
  reformulate(reduced_terms, response = "proportion"),
  data = coin
)

if (length(dropped_interactions) == 0) {{
  selected_fit <- full_fit
}} else {{
  reduction_comparison <- anova(reduced_fit, full_fit)
  reduction_p <- reduction_comparison[["Pr(>F)"]][2]

  if (!is.na(reduction_p) && reduction_p < alpha) {{
    selected_fit <- full_fit
  }} else {{
    selected_fit <- reduced_fit
  }}
}}

diagnostics <- augment(selected_fit)
diagnostics$trial_id <- coin$trial_id

p_residual_run_order <- ggplot(
  diagnostics,
  aes(x = trial_id, y = .resid)
) +
  geom_line(color = "#8A9099", linewidth = 0.5) +
  geom_point(color = "#168AAD", alpha = 0.78, size = 1.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "#555B63") +
  labs(
    title = "Residuals in randomized run order",
    x = "Run",
    y = "Residual"
  ) +
  report_theme

interaction_data <- function(
  group_name,
  design_names = c("denomination", "posture", "starting_side")
) {{
  map_dfr(design_names, function(design_name) {{
    estimated_means <- as.data.frame(
      emmeans(selected_fit, specs = c(design_name, group_name))
    )

    estimated_means |>
      transmute(
        design_factor = design_name,
        design_level = as.character(.data[[design_name]]),
        group_level = as.character(.data[[group_name]]),
        emmean = emmean
      )
  }})
}}

make_interaction_plot <- function(plot_data, group_label) {{
  ggplot(
    plot_data,
    aes(
      x = design_level,
      y = emmean,
      color = group_level,
      group = group_level
    )
  ) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2.7) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "#555B63") +
    facet_wrap(
      ~ design_factor,
      scales = "free_x",
      nrow = 1,
      labeller = as_labeller(design_labels)
    ) +
    scale_color_manual(values = interaction_colors) +
    labs(
      title = paste(group_label, "interactions with the design factors"),
      x = NULL,
      y = "Adjusted mean proportion of heads",
      color = group_label
    ) +
    report_theme +
    theme(axis.text.x = element_text(angle = 25, hjust = 1))
}}

p_flipper_interactions <- make_interaction_plot(
  interaction_data("flipper"),
  "Flipper"
)

p_decade_interactions <- make_interaction_plot(
  interaction_data("decade"),
  "Coin decade"
)

p_starting_side_interactions <- make_interaction_plot(
  interaction_data(
    "starting_side",
    c("denomination", "posture")
  ),
  "Starting side"
)

p_qq <- ggplot(diagnostics, aes(sample = .std.resid)) +
  stat_qq(color = "#168AAD", alpha = 0.72, size = 2) +
  stat_qq_line(color = "#E76F51", linewidth = 0.8) +
  labs(
    title = "Normal Q-Q plot of model residuals",
    x = "Theoretical quantiles",
    y = "Standardized residuals"
  ) +
  report_theme

p_residuals <- ggplot(diagnostics, aes(x = .fitted, y = .resid)) +
  geom_point(color = "#168AAD", alpha = 0.72, size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "#555B63") +
  labs(
    title = "Residuals versus fitted values",
    x = "Fitted value",
    y = "Residual"
  ) +
  report_theme

print(p_design_mean)
print(p_design_observed)
print(p_probability_distribution)
print(p_binomial_overlay)
print(p_decade)
print(p_flipper)
print(p_flipper_interactions)
print(p_decade_interactions)
print(p_starting_side_interactions)

print(p_qq)
print(p_residuals)
print(p_residual_run_order)
"""


def full_r_script(csv_path):
    return r_analysis_code(csv_path) + "\n\n" + r_ggplot_code(csv_path)


def design_summary_table(df):
    return (
        df
        .groupby(
            ["denomination", "posture", "starting_side"],
            as_index=False
        )
        .agg(
            runs=("trial_id", "count"),
            total_heads=("heads", "sum"),
            total_flips=("total", "sum"),
            mean_proportion=("proportion", "mean")
        )
        .assign(p_hat=lambda x: x["total_heads"] / x["total_flips"])
    )


def block_summary_table(df):
    return (
        df
        .groupby(["decade", "flipper"], as_index=False)
        .agg(
            runs=("trial_id", "count"),
            total_heads=("heads", "sum"),
            total_flips=("total", "sum"),
            mean_proportion=("proportion", "mean")
        )
        .assign(p_hat=lambda x: x["total_heads"] / x["total_flips"])
    )


def r_escape_path(path):
    return str(path).replace("\\", "/").replace('"', '\\"')


def r_ggplot_render_script(input_csv, output_dir):
    plot_code = r_ggplot_code(r_escape_path(input_csv))

    return plot_code + f"""

dir.create("{r_escape_path(output_dir)}", showWarnings = FALSE, recursive = TRUE)

ggsave(file.path("{r_escape_path(output_dir)}", "01-design-means.png"), p_design_mean, width = 8, height = 5, dpi = 160)
ggsave(file.path("{r_escape_path(output_dir)}", "02-design-observed.png"), p_design_observed, width = 8, height = 5, dpi = 160)
ggsave(file.path("{r_escape_path(output_dir)}", "03-probability-distribution.png"), p_probability_distribution, width = 8, height = 5, dpi = 160)
ggsave(file.path("{r_escape_path(output_dir)}", "04-binomial-overlay.png"), p_binomial_overlay, width = 8, height = 5, dpi = 160)
ggsave(file.path("{r_escape_path(output_dir)}", "05-decade-block.png"), p_decade, width = 7, height = 5, dpi = 160)
ggsave(file.path("{r_escape_path(output_dir)}", "06-flipper-block.png"), p_flipper, width = 7, height = 5, dpi = 160)
ggsave(file.path("{r_escape_path(output_dir)}", "07-flipper-interactions.png"), p_flipper_interactions, width = 11, height = 4.8, dpi = 160)
ggsave(file.path("{r_escape_path(output_dir)}", "08-decade-interactions.png"), p_decade_interactions, width = 11, height = 4.8, dpi = 160)
ggsave(file.path("{r_escape_path(output_dir)}", "09-starting-side-interactions.png"), p_starting_side_interactions, width = 9, height = 4.8, dpi = 160)
ggsave(file.path("{r_escape_path(output_dir)}", "10-qq-residuals.png"), p_qq, width = 7, height = 5, dpi = 160)
ggsave(file.path("{r_escape_path(output_dir)}", "11-residuals-fitted.png"), p_residuals, width = 7, height = 5, dpi = 160)
ggsave(file.path("{r_escape_path(output_dir)}", "12-residuals-run-order.png"), p_residual_run_order, width = 10, height = 5, dpi = 160)
"""


@st.cache_data(show_spinner=False)
def render_r_analysis(data_csv):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_csv = tmp_path / "coin_experiment_data.csv"
        script_path = tmp_path / "run_analysis.R"

        input_csv.write_text(data_csv, encoding="utf-8")
        script_path.write_text(
            r_analysis_code(r_escape_path(input_csv)),
            encoding="utf-8"
        )

        try:
            result = subprocess.run(
                ["Rscript", str(script_path)],
                capture_output=True,
                text=True,
                timeout=90
            )
        except FileNotFoundError:
            return "", "", "Rscript was not found. Install R or make sure Rscript is on PATH."
        except subprocess.TimeoutExpired:
            return "", "", "Rscript timed out while running the R summary."

        if result.returncode != 0:
            return result.stdout, result.stderr, result.stderr or "Rscript failed."

        return result.stdout, result.stderr, ""


@st.cache_data(show_spinner=False)
def render_r_ggplots(data_csv):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_csv = tmp_path / "coin_experiment_data.csv"
        script_path = tmp_path / "render_ggplots.R"
        output_dir = tmp_path / "plots"

        input_csv.write_text(data_csv, encoding="utf-8")
        script_path.write_text(
            r_ggplot_render_script(input_csv, output_dir),
            encoding="utf-8"
        )

        try:
            result = subprocess.run(
                ["Rscript", str(script_path)],
                capture_output=True,
                text=True,
                timeout=90
            )
        except FileNotFoundError:
            return [], "Rscript was not found. Install R or make sure Rscript is on PATH."
        except subprocess.TimeoutExpired:
            return [], "Rscript timed out while rendering the ggplot images."

        if result.returncode != 0:
            return [], result.stderr or result.stdout or "Rscript failed."

        plots = []
        for plot_path in sorted(output_dir.glob("*.png")):
            plots.append((plot_path.stem, plot_path.read_bytes()))

        return plots, ""


def missing_settings():
    missing = []

    if not github_token:
        missing.append("GITHUB_TOKEN")
    if not github_repo:
        missing.append("GITHUB_REPO")
    if not github_path:
        missing.append("GITHUB_PATH")
    if not github_branch:
        missing.append("GITHUB_BRANCH")
    if not delete_password:
        missing.append("DELETE_PASSWORD")

    return missing


def gh_headers():
    return {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json"
    }


def gh_url_for(path):
    return f"https://api.github.com/repos/{github_repo}/contents/{path}"


def gh_url():
    return gh_url_for(github_path)


def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _load_from_github_uncached():
    missing = missing_settings()
    if missing:
        raise RuntimeError(f"Missing Streamlit secrets: {missing}")

    r = requests.get(
        gh_url(),
        headers=gh_headers(),
        params={"ref": github_branch},
        timeout=20
    )

    if r.status_code == 404:
        return empty_data(), None

    if r.status_code != 200:
        raise RuntimeError(f"GitHub load failed: {r.text}")

    payload = r.json()
    sha = payload.get("sha")
    content = payload.get("content", "")

    if content.strip() == "":
        return empty_data(), sha

    csv_text = base64.b64decode(content).decode("utf-8")

    if csv_text.strip() == "":
        return empty_data(), sha

    return clean_data(pd.read_csv(StringIO(csv_text))), sha


@st.cache_data(ttl=GITHUB_CACHE_TTL_SECONDS, show_spinner=False)
def load_from_github():
    return _load_from_github_uncached()


def save_to_github(df, sha):
    df = clean_data(df)

    csv_text = df[cols].to_csv(index=False)
    encoded = base64.b64encode(csv_text.encode("utf-8")).decode("utf-8")

    data = {
        "message": f"Update coin data {datetime.now().isoformat(timespec='seconds')}",
        "content": encoded,
        "branch": github_branch
    }

    if sha is not None:
        data["sha"] = sha

    r = requests.put(
        gh_url(),
        headers=gh_headers(),
        json=data,
        timeout=20
    )

    if r.status_code not in [200, 201]:
        raise RuntimeError(f"GitHub save failed: {r.text}")

    load_from_github.clear()

    return r.json()


def _load_calendar_from_github_uncached():
    missing = missing_settings()
    if missing:
        raise RuntimeError(f"Missing Streamlit secrets: {missing}")

    r = requests.get(
        gh_url_for(calendar_github_path),
        headers=gh_headers(),
        params={"ref": github_branch},
        timeout=20
    )

    if r.status_code == 404:
        return empty_calendar_data(), None

    if r.status_code != 200:
        raise RuntimeError(f"GitHub calendar load failed: {r.text}")

    payload = r.json()
    sha = payload.get("sha")
    content = payload.get("content", "")

    if content.strip() == "":
        return empty_calendar_data(), sha

    csv_text = base64.b64decode(content).decode("utf-8")

    if csv_text.strip() == "":
        return empty_calendar_data(), sha

    return clean_calendar_data(pd.read_csv(StringIO(csv_text))), sha


@st.cache_data(ttl=GITHUB_CACHE_TTL_SECONDS, show_spinner=False)
def load_calendar_from_github():
    return _load_calendar_from_github_uncached()


def _load_notes_from_github_uncached():
    missing = missing_settings()
    if missing:
        raise RuntimeError(f"Missing Streamlit secrets: {missing}")

    r = requests.get(
        gh_url_for(notes_github_path),
        headers=gh_headers(),
        params={"ref": github_branch},
        timeout=20
    )

    if r.status_code == 404:
        return empty_notes_data(), None

    if r.status_code != 200:
        raise RuntimeError(f"GitHub notes load failed: {r.text}")

    payload = r.json()
    sha = payload.get("sha")
    content = payload.get("content", "")

    if content.strip() == "":
        return empty_notes_data(), sha

    csv_text = base64.b64decode(content).decode("utf-8")

    if csv_text.strip() == "":
        return empty_notes_data(), sha

    return clean_notes_data(pd.read_csv(StringIO(csv_text))), sha


@st.cache_data(ttl=GITHUB_CACHE_TTL_SECONDS, show_spinner=False)
def load_notes_from_github():
    return _load_notes_from_github_uncached()


def save_calendar_to_github(df, sha):
    df = clean_calendar_data(df)
    csv_text = df.to_csv(index=False)
    encoded = base64.b64encode(csv_text.encode("utf-8")).decode("utf-8")

    data = {
        "message": f"Update calendar events {datetime.now().isoformat(timespec='seconds')}",
        "content": encoded,
        "branch": github_branch
    }

    if sha is not None:
        data["sha"] = sha

    r = requests.put(
        gh_url_for(calendar_github_path),
        headers=gh_headers(),
        json=data,
        timeout=20
    )

    if r.status_code not in [200, 201]:
        raise RuntimeError(f"GitHub calendar save failed: {r.text}")

    load_calendar_from_github.clear()

    return r.json()


def save_notes_to_github(df, sha):
    df = clean_notes_data(df)
    csv_text = df.to_csv(index=False)
    encoded = base64.b64encode(csv_text.encode("utf-8")).decode("utf-8")

    data = {
        "message": f"Update experiment notes {datetime.now().isoformat(timespec='seconds')}",
        "content": encoded,
        "branch": github_branch
    }

    if sha is not None:
        data["sha"] = sha

    r = requests.put(
        gh_url_for(notes_github_path),
        headers=gh_headers(),
        json=data,
        timeout=20
    )

    if r.status_code not in [200, 201]:
        raise RuntimeError(f"GitHub notes save failed: {r.text}")

    load_notes_from_github.clear()

    return r.json()


def add_calendar_event_to_github(event_date, event_time, event_type, event_title):
    old_df, sha = _load_calendar_from_github_uncached()
    new_df = pd.concat(
        [
            old_df,
            pd.DataFrame([{
                "date": event_date.isoformat(),
                "time": event_time.strftime("%H:%M"),
                "type": event_type,
                "title": event_title.strip()
            }])
        ],
        ignore_index=True
    )
    save_calendar_to_github(new_df, sha)
    refreshed, _ = load_calendar_from_github()
    return refreshed


def add_experiment_note_to_github(author, note_type, note_text):
    old_df, sha = _load_notes_from_github_uncached()
    created_at = datetime.now().isoformat(timespec="seconds")
    note_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    new_df = pd.concat(
        [
            old_df,
            pd.DataFrame([{
                "note_id": note_id,
                "created_at": created_at,
                "author": author.strip() or "Anonymous",
                "type": note_type,
                "note": note_text.strip()
            }])
        ],
        ignore_index=True
    )
    save_notes_to_github(new_df, sha)
    refreshed, _ = load_notes_from_github()
    return refreshed


def delete_experiment_note_from_github(note_id):
    old_df, sha = _load_notes_from_github_uncached()
    matches = old_df[old_df["note_id"].astype(str) == str(note_id)]

    if len(matches) == 0:
        raise RuntimeError("Note was not found.")

    new_df = old_df.drop(index=matches.index[0]).reset_index(drop=True)
    save_notes_to_github(new_df, sha)
    refreshed, _ = load_notes_from_github()
    return refreshed


def delete_calendar_event_from_github(event_date, event_time, event_type, event_title):
    old_df, sha = _load_calendar_from_github_uncached()
    event_date_text = event_date.isoformat()
    event_time_text = event_time.strip()
    event_type_text = event_type.strip()
    event_title_text = event_title.strip()

    matches = old_df[
        (old_df["date"] == event_date_text)
        & (old_df["time"] == event_time_text)
        & (old_df["type"] == event_type_text)
        & (old_df["title"] == event_title_text)
    ]

    if len(matches) == 0:
        raise RuntimeError("Calendar event was not found.")

    new_df = old_df.drop(index=matches.index[0]).reset_index(drop=True)
    save_calendar_to_github(new_df, sha)
    refreshed, _ = load_calendar_from_github()
    return refreshed


def clear_calendar_date_from_github(event_date):
    old_df, sha = _load_calendar_from_github_uncached()
    event_date_text = event_date.isoformat()
    new_df = old_df[old_df["date"] != event_date_text].copy()

    if len(new_df) == len(old_df):
        raise RuntimeError("No calendar events were found for that date.")

    save_calendar_to_github(new_df, sha)
    refreshed, _ = load_calendar_from_github()
    return refreshed


def add_row_to_github(new_row, schedule=None):
    old_df, sha = _load_from_github_uncached()

    if schedule is not None:
        old_df = valid_submitted_data(old_df, schedule)

    new_trial_id = int(new_row["trial_id"].iloc[0])

    if len(old_df) > 0:
        same_id = old_df[old_df["trial_id"].astype(int) == new_trial_id]

        if len(same_id) > 0:
            matching_same_id = same_id.copy()

            for c in assignment_cols:
                matching_same_id = matching_same_id[
                    matching_same_id[c] == new_row[c].iloc[0]
                ]

            if len(matching_same_id) > 0:
                raise RuntimeError(
                    f"Run ID {new_trial_id} already exists. Reload the dashboard to get the next run."
                )

    new_df = pd.concat(
        [old_df[cols], new_row[cols]],
        ignore_index=True
    )

    new_df = clean_data(new_df)
    save_to_github(new_df, sha)

    return new_df


def make_submission_row(run, heads):
    return pd.DataFrame([{
        "trial_id": int(run["run_id"]),
        "denomination": run["denomination"],
        "decade": run["decade"],
        "posture": run["posture"],
        "flipper": run["flipper"],
        "starting_side": run["starting_side"],
        "heads": int(heads)
    }])


def save_submission(new_row, schedule):
    refreshed_df = add_row_to_github(new_row, schedule)
    refreshed_valid_df = valid_submitted_data(refreshed_df, schedule)
    refreshed_progress = schedule_progress_from_valid_data(schedule, refreshed_valid_df)
    refreshed_next_run_id = next_run_id_from_valid_data(refreshed_valid_df, schedule)
    refreshed_next_run = scheduled_run(schedule, refreshed_next_run_id)

    return (
        refreshed_df,
        refreshed_valid_df,
        refreshed_progress,
        refreshed_next_run_id,
        refreshed_next_run,
    )


def record_flip(flip_state_key, result):
    st.session_state[DATA_TOOLS_KEY] = False
    flip_results = st.session_state.setdefault(flip_state_key, [])

    if len(flip_results) < FLIPS_PER_TRIAL:
        flip_results.append(result)


def undo_flip(flip_state_key):
    st.session_state[DATA_TOOLS_KEY] = False
    flip_results = st.session_state.setdefault(flip_state_key, [])

    if flip_results:
        flip_results.pop()


def reset_flips(flip_state_key):
    st.session_state[DATA_TOOLS_KEY] = False
    st.session_state[flip_state_key] = []


def delete_trial_from_github(trial_id):
    old_df, sha = _load_from_github_uncached()

    old_count = len(old_df)
    new_df = old_df[old_df["trial_id"] != trial_id].copy()

    if len(new_df) == old_count:
        raise RuntimeError(f"Trial ID {trial_id} was not found.")

    save_to_github(new_df, sha)

    refreshed, _ = load_from_github()
    return refreshed


def delete_valid_trial_from_github(trial_id, schedule):
    old_df, sha = _load_from_github_uncached()
    mapped_df = mapped_valid_submitted_data(old_df, schedule)

    matches = mapped_df[mapped_df["trial_id"].astype(int) == int(trial_id)]

    if len(matches) == 0:
        raise RuntimeError(f"Run ID {trial_id} was not found in the current schedule.")

    row = matches.iloc[0]
    source_trial_id = int(row["_source_trial_id"])
    delete_mask = old_df["trial_id"].astype(int) == source_trial_id

    for c in assignment_cols:
        delete_mask = delete_mask & (old_df[c].astype(str) == str(row[c]))

    delete_mask = delete_mask & (old_df["heads"].astype(int) == int(row["heads"]))

    if not delete_mask.any():
        raise RuntimeError(f"Run ID {trial_id} could not be matched to a saved CSV row.")

    new_df = old_df[~delete_mask].copy()
    save_to_github(new_df, sha)

    refreshed, _ = load_from_github()
    return refreshed


def delete_all_rows_from_github():
    old_df, sha = _load_from_github_uncached()

    new_df = empty_data()
    save_to_github(new_df, sha)

    refreshed, _ = load_from_github()
    return refreshed


st.sidebar.header("Settings")

data_mode = st.sidebar.radio(
    "Data shown in results tab",
    ["Real GitHub data", "Dummy data"],
    index=0
)

st.sidebar.write(f"Each run has **{FLIPS_PER_TRIAL} flips**.")

if st.sidebar.button("Reload data"):
    load_from_github.clear()
    load_calendar_from_github.clear()
    load_notes_from_github.clear()
    rerun_app()


missing = missing_settings()

if missing:
    st.error(f"Missing GitHub settings in `.streamlit/secrets.toml`: {missing}")
    st.stop()

try:
    df, current_sha = load_from_github()
except Exception as e:
    st.error("Could not load data from GitHub.")
    st.write(str(e))
    st.stop()

fresh_saved_csv = st.session_state.pop("fresh_saved_data_csv", None)

if fresh_saved_csv:
    df = clean_data(pd.read_csv(StringIO(fresh_saved_csv)))

run_schedule = load_run_schedule()
valid_df = valid_submitted_data(df, run_schedule)
next_run_id = next_run_id_from_valid_data(valid_df, run_schedule)
next_run = scheduled_run(run_schedule, next_run_id)
run_progress = schedule_progress_from_valid_data(run_schedule, valid_df)

if data_mode == "Dummy data":
    results_df = make_dummy_data()
else:
    results_df = valid_df


submit_tab, results_tab, instructions_tab = st.tabs([
    "Submit and edit data",
    "View results and analysis",
    "Experiment Housekeeping",
])

submission_saved = st.session_state.pop("submission_saved", False)


with submit_tab:
    st.subheader("Submit the next scheduled 10-flip run")

    if submission_saved:
        st.success("Your result was saved. The run list below has been refreshed.")

    if len(run_schedule) == 0:
        st.error(
            f"No randomized run schedule was found at `{RUN_SCHEDULE_PATH}`. "
            "Run `python3 scripts/generate_run_schedule.py` first."
        )
    elif next_run is None:
        st.success("All scheduled runs have been submitted.")
    else:
        completed_count = int(run_progress["complete"].sum())
        total_runs = len(run_progress)

        st.markdown(
            f"""
            <div class="run-summary">
                <div class="run-summary-item">
                    <div class="run-summary-label">Next run</div>
                    <div class="run-summary-value">{int(next_run["run_id"])}</div>
                </div>
                <div class="run-summary-item">
                    <div class="run-summary-label">Completed</div>
                    <div class="run-summary-value">{completed_count} / {total_runs}</div>
                </div>
                <div class="run-summary-item">
                    <div class="run-summary-label">Remaining</div>
                    <div class="run-summary-value">{total_runs - completed_count}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="assigned-run-panel">
                <div class="assigned-grid">
                    <div>
                        <div class="assigned-label">Replication</div>
                        <div class="assigned-value">{int(next_run["replication"])}</div>
                    </div>
                    <div>
                        <div class="assigned-label">Denomination</div>
                        <div class="assigned-value">{next_run["denomination"]}</div>
                    </div>
                    <div>
                        <div class="assigned-label">Decade</div>
                        <div class="assigned-value">{next_run["decade"]}</div>
                    </div>
                    <div>
                        <div class="assigned-label">Posture</div>
                        <div class="assigned-value">{next_run["posture"]}</div>
                    </div>
                    <div>
                        <div class="assigned-label">Flipper</div>
                        <div class="assigned-value">{next_run["flipper"]}</div>
                    </div>
                    <div>
                        <div class="assigned-label">Starting side</div>
                        <div class="assigned-value">{next_run["starting_side"]}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        flip_state_key = "current_flip_results"
        flip_run_key = "current_flip_run_id"

        if st.session_state.get(flip_run_key) != int(next_run["run_id"]):
            st.session_state[flip_state_key] = []
            st.session_state[flip_run_key] = int(next_run["run_id"])

        st.markdown("### Track this run")

        flip_results = st.session_state.get(flip_state_key, [])
        live_heads = flip_results.count("Heads")
        live_tails = flip_results.count("Tails")
        flips_recorded = len(flip_results)

        st.markdown(
            f"""
            <div class="flip-panel">
                <div class="flip-grid">
                    <div class="flip-stat">
                        <div class="flip-label">Flips recorded</div>
                        <div class="flip-value">{flips_recorded} / {FLIPS_PER_TRIAL}</div>
                    </div>
                    <div class="flip-stat">
                        <div class="flip-label">Heads</div>
                        <div class="flip-value">{live_heads}</div>
                    </div>
                    <div class="flip-stat">
                        <div class="flip-label">Tails</div>
                        <div class="flip-value">{live_tails}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(flips_recorded / FLIPS_PER_TRIAL)

        b1, b2, b3, b4 = st.columns([1.25, 1.25, 1, 1])
        with b1:
            st.button(
                "Heads",
                disabled=flips_recorded >= FLIPS_PER_TRIAL,
                use_container_width=True,
                type="primary",
                on_click=record_flip,
                args=(flip_state_key, "Heads")
            )
        with b2:
            st.button(
                "Tails",
                disabled=flips_recorded >= FLIPS_PER_TRIAL,
                use_container_width=True,
                on_click=record_flip,
                args=(flip_state_key, "Tails")
            )
        with b3:
            st.button(
                "Undo",
                disabled=flips_recorded == 0,
                use_container_width=True,
                on_click=undo_flip,
                args=(flip_state_key,)
            )
        with b4:
            st.button(
                "Reset",
                disabled=flips_recorded == 0,
                use_container_width=True,
                on_click=reset_flips,
                args=(flip_state_key,)
            )

        if flip_results:
            history_html = "".join(
                (
                    f'<span class="flip-pill flip-pill-{x.lower()}">'
                    f'{i + 1}: {x[0]}</span>'
                )
                for i, x in enumerate(flip_results)
            )
            st.markdown(
                f'<div class="flip-history">{history_html}</div>',
                unsafe_allow_html=True
            )

        if st.button(
            "Submit tracked run",
            disabled=flips_recorded != FLIPS_PER_TRIAL,
            use_container_width=True,
            type="primary"
        ):
            new_row = make_submission_row(next_run, live_heads)

            try:
                (
                    df,
                    valid_df,
                    run_progress,
                    next_run_id,
                    next_run,
                ) = save_submission(new_row, run_schedule)
                st.session_state[flip_state_key] = []
                st.session_state["submission_saved"] = True
                st.session_state["fresh_saved_data_csv"] = df[cols].to_csv(index=False)
                rerun_app()
            except Exception as e:
                st.error("Submission failed.")
                st.write(str(e))

        st.markdown("### Or enter the total")

        with st.form("trial_form", clear_on_submit=True):
            heads = st.selectbox(
                "Heads out of 10",
                list(range(FLIPS_PER_TRIAL + 1)),
                index=FLIPS_PER_TRIAL // 2
            )

            submitted = st.form_submit_button(
                "Submit entered total",
                use_container_width=True
            )

        if submitted:
            new_row = make_submission_row(next_run, heads)

            try:
                (
                    df,
                    valid_df,
                    run_progress,
                    next_run_id,
                    next_run,
                ) = save_submission(new_row, run_schedule)
                st.session_state[flip_state_key] = []
                st.session_state["submission_saved"] = True
                st.session_state["fresh_saved_data_csv"] = df[cols].to_csv(index=False)
                rerun_app()
            except Exception as e:
                st.error("Submission failed.")
                st.write(str(e))

    st.divider()

    with st.expander(
        "Optional: full run list, current data, and edit tools",
        expanded=False
    ):
        st.subheader("Full randomized run list")

        if len(run_schedule) == 0:
            st.info("No randomized run list is available yet.")
        else:
            progress_display = run_progress.copy()

            if len(valid_df) > 0:
                submitted_heads = (
                    valid_df[["trial_id", "heads"]]
                    .drop_duplicates(subset=["trial_id"], keep="last")
                    .rename(columns={"trial_id": "run_id", "heads": "submitted_heads"})
                )
                progress_display = progress_display.merge(
                    submitted_heads,
                    on="run_id",
                    how="left"
                )
            else:
                progress_display["submitted_heads"] = np.nan

            progress_display["proportion_heads"] = (
                progress_display["submitted_heads"] / FLIPS_PER_TRIAL
            ).round(3)

            display_cols = [
                "run_id",
                "status",
                "replication",
                "denomination",
                "decade",
                "posture",
                "flipper",
                "starting_side",
                "submitted_heads",
                "proportion_heads",
            ]

            progress_display_for_table = progress_display[display_cols].copy()
            progress_display_for_table["submitted_heads"] = progress_display_for_table[
                "submitted_heads"
            ].map(
                lambda x: "None" if pd.isna(x) else str(int(x))
            )
            progress_display_for_table["proportion_heads"] = progress_display_for_table[
                "proportion_heads"
            ].map(
                lambda x: "None" if pd.isna(x) else f"{x:.1f}"
            )

            def color_run_status(row):
                if row["status"] == "Complete":
                    return ["background-color: rgba(46, 160, 67, 0.22)"] * len(row)

                return ["background-color: rgba(239, 68, 68, 0.14)"] * len(row)

            styled_progress_display = progress_display_for_table.style.apply(
                color_run_status,
                axis=1
            )

            st.dataframe(
                styled_progress_display,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download run list with status",
                data=progress_display[display_cols].to_csv(index=False).encode("utf-8"),
                file_name="run_schedule_with_status.csv",
                mime="text/csv"
            )

        st.divider()

        st.subheader("Current shared data")

        if len(valid_df) == 0:
            st.info("No data has been submitted yet.")
        else:
            st.dataframe(valid_df, use_container_width=True)

            st.download_button(
                "Download current CSV",
                data=valid_df[cols].to_csv(index=False).encode("utf-8"),
                file_name="coin_experiment_data.csv",
                mime="text/csv"
            )

        if len(valid_df) > 0:
            st.divider()
            st.subheader("Delete a row")

            st.warning("Deleting a row updates the GitHub CSV. Use this only for mistakes.")

            delete_id = st.selectbox(
                "Choose Trial ID to delete",
                sorted(valid_df["trial_id"].unique())
            )

            confirm_delete = st.checkbox("I confirm that I want to delete this row.")

            if st.button("Delete selected row"):
                if not confirm_delete:
                    st.error("Check the confirmation box first.")
                else:
                    try:
                        df = delete_valid_trial_from_github(int(delete_id), run_schedule)
                        st.success(f"Trial ID {delete_id} was deleted.")
                        rerun_app()
                    except Exception as e:
                        st.error("Delete failed.")
                        st.write(str(e))

        if len(df) > 0:
            st.divider()
            st.subheader("Erase all data")

            st.error("Danger zone: this will erase every row in the shared GitHub CSV.")

            erase_password = st.text_input(
                "Enter admin password to erase all rows",
                type="password"
            )

            confirm_erase_all = st.checkbox(
                "I understand this will permanently erase all submitted rows."
            )

            if st.button("Erase all rows"):
                if not confirm_erase_all:
                    st.error("Check the confirmation box first.")
                elif erase_password != delete_password:
                    st.error("Incorrect password.")
                else:
                    try:
                        df = delete_all_rows_from_github()
                        st.success("All rows were erased.")
                        rerun_app()
                    except Exception as e:
                        st.error("Erase failed.")
                        st.write(str(e))

    st.divider()
    st.caption(f"Saving to: `{github_repo}/{github_path}`")


with instructions_tab:
    st.subheader("Experiment Housekeeping")

    st.markdown(f"**Vocabulary:** One run = {FLIPS_PER_TRIAL} flips.")

    st.markdown(
        "**Design factors:** denomination, posture, and starting side."
    )
    st.markdown("**Blocking factors:** decade and flipper.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Held constant")
        for item in HELD_CONSTANTS:
            st.markdown(f"- {item}")

    with c2:
        st.markdown("### Allowed to vary")
        for item in ALLOWED_TO_VARY:
            st.markdown(f"- {item}")

    st.markdown("### Restrictions")
    for item in RESTRICTIONS:
        st.markdown(f"- {item}")

    st.divider()

    st.markdown("### Comments and notes")

    try:
        notes_df, _ = load_notes_from_github()
    except Exception as e:
        notes_df = empty_notes_data()
        st.error("Could not load experiment notes from GitHub.")
        st.write(str(e))

    st.caption(f"Notes are saved to: `{github_repo}/{notes_github_path}`")

    with st.form("experiment_note_form", clear_on_submit=True):
        note_author_col, note_type_col = st.columns([2, 1])

        with note_author_col:
            note_author = st.text_input("Name")

        with note_type_col:
            note_type = st.selectbox("Type", ["Note", "Comment", "Question", "Issue"])

        note_text = st.text_area("Comment or note")
        add_note = st.form_submit_button("Add note", use_container_width=True)

    if add_note:
        if not note_text.strip():
            st.error("Write a comment or note first.")
        else:
            try:
                notes_df = add_experiment_note_to_github(
                    note_author,
                    note_type,
                    note_text
                )
                st.success("Note was saved.")
                rerun_app()
            except Exception as e:
                st.error("Note could not be saved.")
                st.write(str(e))

    if len(notes_df) == 0:
        st.info("No comments or notes yet.")
    else:
        st.dataframe(
            notes_df[["created_at", "author", "type", "note"]],
            use_container_width=True,
            hide_index=True
        )

        with st.expander("Delete a note"):
            note_label_by_id = {
                row["note_id"]: (
                    f'{row["created_at"]} [{row["type"]}] '
                    f'{row["author"]}: {row["note"][:80]}'
                )
                for _, row in notes_df.iterrows()
            }
            note_to_delete = st.selectbox(
                "Choose note to delete",
                list(note_label_by_id.keys()),
                format_func=lambda note_id: note_label_by_id[note_id]
            )
            confirm_note_delete = st.checkbox("I confirm that I want to delete this note.")

            if st.button("Delete selected note"):
                if not confirm_note_delete:
                    st.error("Check the confirmation box first.")
                else:
                    try:
                        notes_df = delete_experiment_note_from_github(note_to_delete)
                        st.success("Note was deleted.")
                        rerun_app()
                    except Exception as e:
                        st.error("Note could not be deleted.")
                        st.write(str(e))

    st.divider()
    st.markdown("### July 2026")

    selected_month = 7
    selected_year = 2026

    try:
        calendar_df, _ = load_calendar_from_github()
    except Exception as e:
        calendar_df = empty_calendar_data()
        st.error("Could not load calendar events from GitHub.")
        st.write(str(e))

    st.caption(f"Calendar events are saved to: `{github_repo}/{calendar_github_path}`")

    july_start = datetime(selected_year, 7, 1).date()
    july_end = datetime(selected_year, 7, 31).date()

    with st.form("calendar_event_form", clear_on_submit=True):
        event_date_col, event_time_col, event_type_col, event_title_col = st.columns([1, 1, 1, 2])

        with event_date_col:
            event_date = st.date_input(
                "July date",
                value=july_start,
                min_value=july_start,
                max_value=july_end
            )

        with event_time_col:
            event_time = st.time_input("Time", value=datetime(2026, 7, 1, 9, 0).time())

        with event_type_col:
            event_type = st.selectbox("Type", ["Meeting", "Due", "Consultation"])

        with event_title_col:
            event_title = st.text_input("Event")

        add_event = st.form_submit_button("Add event")

    if add_event and event_title.strip():
        try:
            calendar_df = add_calendar_event_to_github(
                event_date,
                event_time,
                event_type,
                event_title
            )
            st.success("Calendar event was saved.")
            rerun_app()
        except Exception as e:
            st.error("Calendar event could not be saved.")
            st.write(str(e))

    events = calendar_df.to_dict("records")
    selected_month_events = [
        event for event in events
        if datetime.fromisoformat(event["date"]).date().year == selected_year
        and datetime.fromisoformat(event["date"]).date().month == selected_month
    ]

    if selected_month_events:
        event_options = [
            (
                f'{event["date"]} {event.get("time", "").strip()} '
                f'[{event.get("type", "Meeting")}] - {event["title"]}'
            ).replace("  [", " [")
            for event in selected_month_events
        ]
        event_to_remove = st.selectbox("Remove one event", event_options)

        if st.button("Remove selected event"):
            remove_index = event_options.index(event_to_remove)
            event_to_delete = selected_month_events[remove_index]
            try:
                calendar_df = delete_calendar_event_from_github(
                    datetime.fromisoformat(event_to_delete["date"]).date(),
                    event_to_delete.get("time", ""),
                    event_to_delete.get("type", "Meeting"),
                    event_to_delete["title"]
                )
                st.success("Calendar event was removed.")
                rerun_app()
            except Exception as e:
                st.error("Calendar event could not be removed.")
                st.write(str(e))

        clear_date = st.date_input(
            "Date to clear",
            value=july_start,
            min_value=july_start,
            max_value=july_end
        )

        if st.button("Clear selected date"):
            try:
                calendar_df = clear_calendar_date_from_github(clear_date)
                st.success(f"All events on {clear_date.isoformat()} were cleared.")
                rerun_app()
            except Exception as e:
                st.error("Date could not be cleared.")
                st.write(str(e))

    st.markdown(
        render_calendar_html(selected_year, selected_month, calendar_df.to_dict("records")),
        unsafe_allow_html=True
    )


with results_tab:
    if data_mode == "Dummy data":
        st.info("Showing generated dummy data. This does not change the GitHub CSV.")
    else:
        st.info("Showing real submitted data from GitHub.")

    if len(results_df) == 0:
        st.info("No data has been submitted yet.")
        st.stop()

    st.subheader("Analysis plan")

    st.markdown(
        r"""
        **Experiment steps**

        1. **Fit the full model and run a Type III ANOVA.** The response is the
           proportion of heads. Denomination, posture, and starting side are
           design factors; decade and flipper are blocking factors. The full
           model contains every main effect and all interactions through order
           three.
        2. **Create and fit the reduced model.** Interactions that are not
           significant at the 5% level are removed together. Model hierarchy is
           preserved: a retained interaction keeps all of its lower-order
           components. The reduced model is fitted to the same complete set of
           experimental runs as the full model. No observations are removed;
           only unsupported terms are removed from the formula.
        3. **Compare the reduced and full models.** A partial F-test, also
           called an extra-sum-of-squares ANOVA, compares the variation
           explained per omitted coefficient with the full model's residual
           variation. This is not a raw equality-of-variances test. It asks
           whether simplifying the formula causes a statistically significant
           loss of fit.
        4. **Select the model.** If the partial F-test is significant, retain
           the full model because the removed terms matter jointly. Otherwise,
           use the reduced model because it explains the data adequately with
           fewer coefficients and more residual degrees of freedom.
        5. **Check the selected model.** Examine the residual-versus-fitted,
           normal Q-Q, and residual-versus-run-order plots before interpreting
           its predictions.
        6. **Predict every design combination.** Predict all 12 combinations of
           denomination, posture, and starting side while averaging equally
           over the decade and flipper blocks. The fairest condition is the
           predicted proportion closest to 0.50, while the condition with the
           highest predicted proportion of heads is reported separately.

        Replication contributes repeated observations to the error estimate and
        is not entered as a separate model term. The full model is:
        """
    )

    st.latex(
        r"""
        \begin{aligned}
        y_{ijklm(r)} ={}& \mu
        + \alpha_i + \beta_j + \gamma_k + \delta_l + \phi_m \\
        &+ (\alpha\beta)_{ij} + (\alpha\gamma)_{ik}
        + (\alpha\delta)_{il} + (\alpha\phi)_{im} \\
        &+ (\beta\gamma)_{jk} + (\beta\delta)_{jl}
        + (\beta\phi)_{jm} \\
        &+ (\gamma\delta)_{kl} + (\gamma\phi)_{km}
        + (\delta\phi)_{lm} \\
        &+ (\alpha\beta\gamma)_{ijk}
        + (\alpha\beta\delta)_{ijl}
        + (\alpha\beta\phi)_{ijm} \\
        &+ (\alpha\gamma\delta)_{ikl}
        + (\alpha\gamma\phi)_{ikm}
        + (\alpha\delta\phi)_{ilm} \\
        &+ (\beta\gamma\delta)_{jkl}
        + (\beta\gamma\phi)_{jkm}
        + (\beta\delta\phi)_{jlm} \\
        &+ (\gamma\delta\phi)_{klm}
        + \varepsilon_{ijklm(r)}.
        \end{aligned}
        """
    )

    st.markdown(
        r"""
        Here, $\alpha_i$, $\beta_j$, $\gamma_k$, $\delta_l$, and $\phi_m$
        represent denomination, posture, starting side, decade, and flipper,
        respectively. The first three are design-factor effects, while decade
        and flipper are blocking effects. Parenthesized products are
        interactions. Four-way and five-way interactions are not included. The matching R
        formula is:
        """
    )

    st.code(
        """
proportion ~
  denomination + posture + starting_side + decade + flipper +
  denomination:posture +
  denomination:starting_side +
  denomination:decade +
  denomination:flipper +
  posture:starting_side +
  posture:decade +
  posture:flipper +
  starting_side:decade +
  starting_side:flipper +
  decade:flipper +
  denomination:posture:starting_side +
  denomination:posture:decade +
  denomination:posture:flipper +
  denomination:starting_side:decade +
  denomination:starting_side:flipper +
  denomination:decade:flipper +
  posture:starting_side:decade +
  posture:starting_side:flipper +
  posture:decade:flipper +
  starting_side:decade:flipper
        """
        ,
        language="r"
    )

    st.markdown(
        r"""
        In the R formula, `:` means an interaction between the named factors.
        Model reduction uses $\alpha = 0.05$. All five main effects remain, and
        every lower-order component required by a retained interaction also
        remains. No Tukey tests are performed.
        """
    )

    st.markdown(
        r"""
        Terms omitted from the selected model contribute to the residual error
        estimate $\hat{\sigma}^2$. That same estimate is used in the F-tests
        and in the confidence intervals.

        A 95% confidence interval here is a range for the model-estimated
        average number of heads under a design combination. It describes
        uncertainty about that average, not the range in which every individual
        10-flip run must fall.

        The residual-versus-fitted, Q-Q, and residual-versus-run-order plots
        are used to judge the normal-model assumptions. A transformation or a
        binomial model would be considered only if those diagnostics show a
        serious pattern.

        Finally, an R loop predicts all 12 denomination-by-posture-by-starting
        side design combinations while averaging equally over the decade and
        flipper blocks. No flipper-specific highest or fairest recommendation
        is produced.
        """
    )

    st.markdown(
        """
        **Worked prediction example.** Using the current completed dataset, fix
        the design condition at Dime, Sitting, and Tails. The selected model
        gives one prediction for each of the two-decade-by-three-flipper block
        combinations. Their equal-weight average is:
        """
    )

    st.latex(
        r"""
        \begin{aligned}
        \widehat p_{\mathrm{Dime,Sitting,Tails}}
        &=
        \frac{
        0.493 + 0.576 + 0.571 + 0.483 + 0.566 + 0.561
        }{6} \\
        &= 0.5417, \\
        \widehat H
        &= 10\widehat p
        = 10(0.5417)
        = 5.42\ \text{expected heads}.
        \end{aligned}
        """
    )

    st.markdown(
        """
        The selected-model coefficients used for those six predictions were
        estimated from all completed runs. Averaging across the six block
        conditions removes individual flipper information from the reported
        result.
        """
    )

    st.subheader("Binomial distribution for 10 coin flips")

    st.code(
        """
coin <- read.csv("coin_experiment_data.csv")
n <- 10
x <- 0:n
p <- 0.5
observed_p <- sum(coin$heads) / (nrow(coin) * n)

binomial_probability <- function(x, n, p) {
  (
    factorial(n) /
    (factorial(n - x) * factorial(x))
  ) * p^x * (1 - p)^(n - x)
}

fair_probability <- binomial_probability(x, n, p)
observed_probability <- binomial_probability(x, n, observed_p)
observed_count <- tabulate(coin$heads + 1, nbins = n + 1)

binomial_distribution <- data.frame(
  heads = x,
  tails = n - x,
  observed_count = observed_count,
  observed_frequency = observed_count / nrow(coin),
  fair_probability = fair_probability,
  observed_probability = observed_probability
)

print(binomial_distribution)
        """.strip(),
        language="r"
    )

    binomial_heads = np.arange(FLIPS_PER_TRIAL + 1)
    fair_binomial_probability = np.array([
        comb(FLIPS_PER_TRIAL, int(x))
        * 0.5 ** int(x)
        * 0.5 ** (FLIPS_PER_TRIAL - int(x))
        for x in binomial_heads
    ])
    observed_head_probability = (
        results_df["heads"].sum()
        / (len(results_df) * FLIPS_PER_TRIAL)
    )
    observed_binomial_probability = np.array([
        comb(FLIPS_PER_TRIAL, int(x))
        * observed_head_probability ** int(x)
        * (1 - observed_head_probability)
        ** (FLIPS_PER_TRIAL - int(x))
        for x in binomial_heads
    ])
    observed_binomial_counts = (
        results_df["heads"]
        .astype(int)
        .value_counts()
        .reindex(binomial_heads, fill_value=0)
        .to_numpy()
    )
    binomial_results = pd.DataFrame({
        "Heads": binomial_heads,
        "Tails": FLIPS_PER_TRIAL - binomial_heads,
        "Observed runs": observed_binomial_counts,
        "Observed frequency": observed_binomial_counts / len(results_df),
        "Fair-coin probability": fair_binomial_probability,
        "Observed-p probability": observed_binomial_probability,
    })

    st.metric(
        "Observed probability of heads",
        f"{observed_head_probability:.4f}"
    )

    st.dataframe(
        binomial_results.style.format({
            "Observed frequency": "{:.4%}",
            "Fair-coin probability": "{:.4%}",
            "Observed-p probability": "{:.4%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Data preview for R")
    st.dataframe(results_df[cols], use_container_width=True)

    csv_path = "coin_experiment_data.csv"
    analysis_code = r_analysis_code(csv_path)
    plot_code = r_ggplot_code(csv_path)
    script_code = full_r_script(csv_path)

    download_script_col, download_data_col = st.columns(2)

    with download_script_col:
        st.download_button(
            "Download R analysis script",
            data=script_code.encode("utf-8"),
            file_name="coin_blocked_factorial_analysis.R",
            mime="text/plain",
            use_container_width=True
        )

    with download_data_col:
        st.download_button(
            "Download data for R",
            data=results_df[cols].to_csv(index=False).encode("utf-8"),
            file_name="coin_experiment_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.divider()

    code_tab, plot_tab = st.tabs([
        "R summaries and model",
        "R ggplot visualizations",
    ])

    with code_tab:
        analysis_data_csv = results_df[cols].to_csv(index=False)
        analysis_result_key = "r_analysis_result"
        analysis_data_key = "r_analysis_data_csv"
        run_analysis = st.button(
            "Run/update R summaries and model",
            use_container_width=True
        )

        if run_analysis:
            with st.spinner("Running R summaries and model..."):
                st.session_state[analysis_result_key] = render_r_analysis(
                    analysis_data_csv
                )
                st.session_state[analysis_data_key] = analysis_data_csv

        has_current_analysis = (
            st.session_state.get(analysis_data_key) == analysis_data_csv
            and analysis_result_key in st.session_state
        )

        if not has_current_analysis:
            st.info("Run the R summary/model when you want to refresh the analysis output.")
        else:
            analysis_output, analysis_messages, analysis_error = st.session_state[
                analysis_result_key
            ]

            if analysis_error:
                st.warning(
                    "R could not run on this deployment, so the summaries below are computed in Python."
                )
                if analysis_output:
                    st.code(analysis_output, language="text")
                st.caption(analysis_error)

                st.write("Design-factor summary")
                st.dataframe(
                    design_summary_table(results_df),
                    use_container_width=True,
                    hide_index=True
                )

                st.write("Block-factor summary")
                st.dataframe(
                    block_summary_table(results_df),
                    use_container_width=True,
                    hide_index=True
                )

                st.info("The R ANOVA/model output will appear when Rscript is available.")
            else:
                st.code(analysis_output or "R completed without printed output.", language="text")

            if analysis_messages:
                with st.expander("Show R messages"):
                    st.code(analysis_messages, language="text")

        with st.expander("Show R summary/model code"):
            st.code(analysis_code, language="r")

    with plot_tab:
        plot_data_csv = results_df[cols].to_csv(index=False)
        plot_result_key = f"r_plot_result_v{R_PLOT_RENDER_VERSION}"
        plot_data_key = f"r_plot_data_csv_v{R_PLOT_RENDER_VERSION}"
        run_plots = st.button(
            "Run/update R ggplot images",
            use_container_width=True
        )

        has_current_plots = (
            st.session_state.get(plot_data_key) == plot_data_csv
            and plot_result_key in st.session_state
        )

        if run_plots or not has_current_plots:
            with st.spinner("Rendering ggplot images with R..."):
                st.session_state[plot_result_key] = render_r_ggplots(plot_data_csv)
                st.session_state[plot_data_key] = plot_data_csv

        has_current_plots = (
            st.session_state.get(plot_data_key) == plot_data_csv
            and plot_result_key in st.session_state
        )

        if has_current_plots:
            plots, plot_error = st.session_state[plot_result_key]

            if plots:
                plot_titles = {
                    "01-design-means": "Mean proportion of heads by design combination",
                    "02-design-observed": "Observed proportions by design combination",
                    "03-probability-distribution": "Observed probability distribution of heads",
                    "04-binomial-overlay": "Observed heads with binomial overlays",
                    "05-decade-block": "Block check by coin decade",
                    "06-flipper-block": "Block check by flipper",
                    "07-flipper-interactions": "Flipper interactions with design factors",
                    "08-decade-interactions": "Coin-decade interactions with design factors",
                    "09-starting-side-interactions": "Starting-side interactions with other design factors",
                    "10-qq-residuals": "Normal Q-Q plot of selected-model residuals",
                    "11-residuals-fitted": "Residuals versus fitted values",
                    "12-residuals-run-order": "Residuals in randomized run order",
                }

                plot_lookup = dict(plots)
                interaction_plot_names = {
                    "07-flipper-interactions",
                    "08-decade-interactions",
                    "09-starting-side-interactions",
                }

                for plot_name, plot_bytes in plots:
                    if plot_name in interaction_plot_names:
                        continue

                    st.image(
                        plot_bytes,
                        caption=plot_titles.get(plot_name, plot_name),
                        use_container_width=True
                    )

                interaction_plot_columns = st.columns(3)

                for column, plot_name in zip(
                    interaction_plot_columns,
                    [
                        "07-flipper-interactions",
                        "08-decade-interactions",
                        "09-starting-side-interactions",
                    ]
                ):
                    if plot_name in plot_lookup:
                        with column:
                            st.image(
                                plot_lookup[plot_name],
                                caption=plot_titles[plot_name],
                                use_container_width=True
                            )

                if len(plots) < 12:
                    st.info(
                        "Diagnostic plots will appear once there is enough data to fit the full model."
                    )
            else:
                st.warning(
                    "R could not render ggplot images on this deployment, so these preview charts are computed in Python."
                )
                st.caption(plot_error)

                design_chart = design_summary_table(results_df).copy()
                design_chart["design_condition"] = (
                    design_chart["denomination"].astype(str)
                    + " / "
                    + design_chart["posture"].astype(str)
                    + " / "
                    + design_chart["starting_side"].astype(str)
                )

                st.write("Mean proportion of heads by design combination")
                st.bar_chart(
                    design_chart.set_index("design_condition")["mean_proportion"],
                    use_container_width=True
                )

                probability_chart = (
                    results_df["heads"]
                    .value_counts(normalize=True)
                    .reindex(range(FLIPS_PER_TRIAL + 1), fill_value=0)
                    .sort_index()
                )

                st.write("Observed probability distribution of heads")
                st.bar_chart(
                    probability_chart,
                    use_container_width=True
                )

                block_chart = block_summary_table(results_df).copy()
                block_chart["block_group"] = (
                    block_chart["decade"].astype(str)
                    + " / "
                    + block_chart["flipper"].astype(str)
                )

                st.write("Mean proportion of heads by block group")
                st.bar_chart(
                    block_chart.set_index("block_group")["mean_proportion"],
                    use_container_width=True
                )

        with st.expander("Show R ggplot code"):
            st.code(plot_code, language="r")
