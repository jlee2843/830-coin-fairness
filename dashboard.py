import base64
import calendar
import subprocess
import tempfile
from datetime import datetime
from html import escape
from io import StringIO
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
decades = ["1980s", "2010s"]
postures = ["Standing", "Sitting"]
flippers = ["Jenny", "Josh", "Esther"]
starting_sides = ["Heads", "Tails"]

github_token = st.secrets.get("GITHUB_TOKEN", "")
github_repo = st.secrets.get("GITHUB_REPO", "")
github_path = st.secrets.get("GITHUB_PATH", "data/coin_experiment_data.csv")
calendar_github_path = st.secrets.get("CALENDAR_GITHUB_PATH", "data/calendar_events.csv")
github_branch = st.secrets.get("GITHUB_BRANCH", "main")
delete_password = st.secrets.get("DELETE_PASSWORD", "")


def empty_data():
    return pd.DataFrame(columns=cols)


def empty_calendar_data():
    return pd.DataFrame(columns=["date", "time", "type", "title"])


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

                    if decade == "1980s":
                        p -= 0.02
                    elif decade == "2010s":
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


def r_data_path():
    return github_path if github_path else "data/coin_experiment_data.csv"


def r_analysis_code(csv_path):
    return f"""library(tidyverse)
library(car)
library(broom)

coin <- read_csv("{csv_path}", show_col_types = FALSE) |>
  mutate(
    trial_id = as.integer(trial_id),
    heads = as.integer(heads),
    total = {FLIPS_PER_TRIAL},
    tails = total - heads,
    proportion = heads / total,
    denomination = factor(denomination),
    posture = factor(posture),
    decade = factor(decade),
    flipper = factor(flipper),
    starting_side = factor(starting_side),
    treatment = interaction(denomination, posture, sep = " / ")
  )

treatment_summary <- coin |>
  group_by(denomination, posture) |>
  summarise(
    runs = n(),
    total_heads = sum(heads),
    total_flips = sum(total),
    mean_proportion = mean(proportion),
    p_hat = total_heads / total_flips,
    .groups = "drop"
  )

nuisance_summary <- coin |>
  group_by(decade, flipper, starting_side) |>
  summarise(
    runs = n(),
    total_heads = sum(heads),
    total_flips = sum(total),
    mean_proportion = mean(proportion),
    p_hat = total_heads / total_flips,
    .groups = "drop"
  )

cat("Treatment summary\\n")
print(treatment_summary)
cat("\\nNuisance-factor summary\\n")
print(nuisance_summary)

nuisance_fit <- try(
  lm(
    proportion ~ denomination * posture + decade + flipper + starting_side,
    data = coin
  ),
  silent = TRUE
)

if (inherits(nuisance_fit, "try-error")) {{
  cat("\\nModel output unavailable with the current data.\\n")
}} else {{
  cat("\\nType II ANOVA\\n")
  nuisance_anova <- try(Anova(nuisance_fit, type = 2), silent = TRUE)

  if (inherits(nuisance_anova, "try-error")) {{
    cat("ANOVA unavailable with the current data.\\n")
  }} else {{
    print(nuisance_anova)
  }}

  cat("\\nBinomial model summary\\n")
  binomial_fit <- try(
    glm(
      cbind(heads, tails) ~ denomination * posture + decade + flipper + starting_side,
      family = binomial,
      data = coin
    ),
    silent = TRUE
  )

  if (inherits(binomial_fit, "try-error")) {{
    cat("Binomial model unavailable with the current data.\\n")
  }} else {{
    print(summary(binomial_fit))
  }}
}}
"""


def r_ggplot_code(csv_path):
    return f"""library(tidyverse)
library(broom)

coin <- read_csv("{csv_path}", show_col_types = FALSE) |>
  mutate(
    trial_id = as.integer(trial_id),
    heads = as.integer(heads),
    total = {FLIPS_PER_TRIAL},
    tails = total - heads,
    proportion = heads / total,
    denomination = factor(denomination),
    posture = factor(posture),
    decade = factor(decade),
    flipper = factor(flipper),
    starting_side = factor(starting_side),
    treatment = interaction(denomination, posture, sep = " / ")
  )

treatment_summary <- coin |>
  group_by(denomination, posture) |>
  summarise(
    runs = n(),
    mean_proportion = mean(proportion),
    se = sd(proportion) / sqrt(runs),
    .groups = "drop"
  )

p_treatment_mean <- ggplot(treatment_summary, aes(x = denomination, y = mean_proportion, fill = posture)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65) +
  geom_errorbar(
    aes(ymin = mean_proportion - se, ymax = mean_proportion + se),
    position = position_dodge(width = 0.75),
    width = 0.2
  ) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Mean proportion of heads by treatment",
    x = "Denomination",
    y = "Mean proportion of heads",
    fill = "Posture"
  ) +
  theme_minimal()

p_treatment_observed <- ggplot(coin, aes(x = denomination, y = proportion, color = posture)) +
  geom_jitter(width = 0.12, height = 0, alpha = 0.65, size = 2) +
  geom_boxplot(aes(group = interaction(denomination, posture)), alpha = 0.25, outlier.shape = NA) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  facet_wrap(~ posture) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Observed proportions by treatment",
    x = "Denomination",
    y = "Proportion heads"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

p_probability_distribution <- ggplot(coin, aes(x = heads, y = after_stat(count / sum(count)))) +
  geom_histogram(binwidth = 1, boundary = -0.5, fill = "#4C78A8", color = "white") +
  scale_x_continuous(breaks = 0:{FLIPS_PER_TRIAL}, limits = c(-0.5, {FLIPS_PER_TRIAL} + 0.5)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Observed probability distribution of heads",
    x = "Heads out of 10",
    y = "Observed probability"
  ) +
  theme_minimal()

p_decade <- ggplot(coin, aes(x = decade, y = proportion, fill = decade)) +
  geom_boxplot(alpha = 0.55, outlier.shape = NA) +
  geom_jitter(width = 0.12, alpha = 0.65, size = 2) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Nuisance-factor check by decade",
    x = "Decade",
    y = "Proportion heads"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

p_flipper <- ggplot(coin, aes(x = flipper, y = proportion, fill = flipper)) +
  geom_boxplot(alpha = 0.55, outlier.shape = NA) +
  geom_jitter(width = 0.12, alpha = 0.65, size = 2) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Nuisance-factor check by flipper",
    x = "Flipper",
    y = "Proportion heads"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

nuisance_fit <- try(
  lm(
    proportion ~ denomination * posture + decade + flipper + starting_side,
    data = coin
  ),
  silent = TRUE
)

if (!inherits(nuisance_fit, "try-error")) {{
  diagnostics <- augment(nuisance_fit)

  p_qq <- ggplot(diagnostics, aes(sample = .std.resid)) +
    stat_qq() +
    stat_qq_line() +
    labs(
      title = "Normal Q-Q plot of model residuals",
      x = "Theoretical quantiles",
      y = "Standardized residuals"
    ) +
    theme_minimal()

  p_residuals <- ggplot(diagnostics, aes(x = .fitted, y = .resid)) +
    geom_point(alpha = 0.7) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(
      title = "Residuals versus fitted values",
      x = "Fitted value",
      y = "Residual"
    ) +
    theme_minimal()
}}

print(p_treatment_mean)
print(p_treatment_observed)
print(p_probability_distribution)
print(p_decade)
print(p_flipper)

if (exists("p_qq")) print(p_qq)
if (exists("p_residuals")) print(p_residuals)
"""


def full_r_script(csv_path):
    return r_analysis_code(csv_path) + "\n\n" + r_ggplot_code(csv_path)


def treatment_summary_table(df):
    return (
        df
        .groupby(["denomination", "posture"], as_index=False)
        .agg(
            runs=("trial_id", "count"),
            total_heads=("heads", "sum"),
            total_flips=("total", "sum"),
            mean_proportion=("proportion", "mean")
        )
        .assign(p_hat=lambda x: x["total_heads"] / x["total_flips"])
    )


def nuisance_summary_table(df):
    return (
        df
        .groupby(["decade", "flipper", "starting_side"], as_index=False)
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

ggsave(file.path("{r_escape_path(output_dir)}", "01-treatment-means.png"), p_treatment_mean, width = 8, height = 5, dpi = 160)
ggsave(file.path("{r_escape_path(output_dir)}", "02-treatment-observed.png"), p_treatment_observed, width = 8, height = 5, dpi = 160)
ggsave(file.path("{r_escape_path(output_dir)}", "03-probability-distribution.png"), p_probability_distribution, width = 8, height = 5, dpi = 160)
ggsave(file.path("{r_escape_path(output_dir)}", "04-decade-nuisance.png"), p_decade, width = 7, height = 5, dpi = 160)
ggsave(file.path("{r_escape_path(output_dir)}", "05-flipper-nuisance.png"), p_flipper, width = 7, height = 5, dpi = 160)

if (exists("p_qq")) {{
  ggsave(file.path("{r_escape_path(output_dir)}", "06-qq-residuals.png"), p_qq, width = 7, height = 5, dpi = 160)
}}

if (exists("p_residuals")) {{
  ggsave(file.path("{r_escape_path(output_dir)}", "07-residuals-fitted.png"), p_residuals, width = 7, height = 5, dpi = 160)
}}
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

    st.markdown("**Treatment factors:** denomination and posture.")
    st.markdown("**Nuisance factors:** decade, flipper, and starting side.")

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
        """
        The treatment structure is denomination by posture. Decade, flipper,
        and starting side are included as nuisance factors.

        The R model below uses:

        - Treatment factors: `denomination`, `posture`
        - Nuisance factors: `decade`, `flipper`, `starting_side`
        """
    )

    st.latex(
        r"""
        y_{ijklm}
        =
        \mu
        + \alpha_i
        + \gamma_j
        + (\alpha\gamma)_{ij}
        + d_k
        + f_l
        + s_m
        + \varepsilon_{ijklm}
        """
    )

    st.markdown(
        """
        where $d_k$, $f_l$, and $s_m$ are nuisance-factor effects. There are no
        nuisance-by-treatment interactions in this model.
        """
    )

    st.divider()

    st.subheader("Data preview for R")
    st.dataframe(results_df[cols], use_container_width=True)

    csv_path = r_data_path()
    analysis_code = r_analysis_code(csv_path)
    plot_code = r_ggplot_code(csv_path)
    script_code = full_r_script(csv_path)

    download_script_col, download_data_col = st.columns(2)

    with download_script_col:
        st.download_button(
            "Download R analysis script",
            data=script_code.encode("utf-8"),
            file_name="coin_nuisance_factor_analysis.R",
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

                st.write("Treatment summary")
                st.dataframe(
                    treatment_summary_table(results_df),
                    use_container_width=True,
                    hide_index=True
                )

                st.write("Nuisance-factor summary")
                st.dataframe(
                    nuisance_summary_table(results_df),
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
        plot_result_key = "r_plot_result"
        plot_data_key = "r_plot_data_csv"
        run_plots = st.button(
            "Run/update R ggplot images",
            use_container_width=True
        )

        if run_plots:
            with st.spinner("Rendering ggplot images with R..."):
                st.session_state[plot_result_key] = render_r_ggplots(plot_data_csv)
                st.session_state[plot_data_key] = plot_data_csv

        has_current_plots = (
            st.session_state.get(plot_data_key) == plot_data_csv
            and plot_result_key in st.session_state
        )

        if not has_current_plots:
            st.info("Run the R ggplot renderer when you want to refresh the plot images.")
        else:
            plots, plot_error = st.session_state[plot_result_key]

            if plots:
                plot_titles = {
                    "01-treatment-means": "Mean proportion of heads by treatment",
                    "02-treatment-observed": "Observed proportions by treatment",
                    "03-probability-distribution": "Observed probability distribution of heads",
                    "04-decade-nuisance": "Nuisance-factor check by decade",
                    "05-flipper-nuisance": "Nuisance-factor check by flipper",
                    "06-qq-residuals": "Normal Q-Q plot of model residuals",
                    "07-residuals-fitted": "Residuals versus fitted values",
                }

                for plot_name, plot_bytes in plots:
                    st.image(
                        plot_bytes,
                        caption=plot_titles.get(plot_name, plot_name),
                        use_container_width=True
                    )

                if len(plots) < 7:
                    st.info(
                        "Diagnostic plots will appear once there is enough data to fit the full model."
                    )
            else:
                st.warning(
                    "R could not render ggplot images on this deployment, so these preview charts are computed in Python."
                )
                st.caption(plot_error)

                treatment_chart = treatment_summary_table(results_df).copy()
                treatment_chart["treatment"] = (
                    treatment_chart["denomination"].astype(str)
                    + " / "
                    + treatment_chart["posture"].astype(str)
                )

                st.write("Mean proportion of heads by treatment")
                st.bar_chart(
                    treatment_chart.set_index("treatment")["mean_proportion"],
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

                nuisance_chart = nuisance_summary_table(results_df).copy()
                nuisance_chart["nuisance_group"] = (
                    nuisance_chart["decade"].astype(str)
                    + " / "
                    + nuisance_chart["flipper"].astype(str)
                    + " / "
                    + nuisance_chart["starting_side"].astype(str)
                )

                st.write("Mean proportion of heads by nuisance-factor group")
                st.bar_chart(
                    nuisance_chart.set_index("nuisance_group")["mean_proportion"],
                    use_container_width=True
                )

        with st.expander("Show R ggplot code"):
            st.code(plot_code, language="r")
