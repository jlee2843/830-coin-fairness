import base64
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
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
    @media (max-width: 760px) {
        .run-summary,
        .assigned-grid,
        .flip-grid {
            grid-template-columns: 1fr;
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
github_branch = st.secrets.get("GITHUB_BRANCH", "main")
delete_password = st.secrets.get("DELETE_PASSWORD", "")


def empty_data():
    return pd.DataFrame(columns=cols)


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


def next_run_id_from_data(df, schedule):
    if len(schedule) == 0:
        return None

    valid_df = valid_submitted_data(df, schedule)

    if len(valid_df) == 0:
        return int(schedule["run_id"].min())

    next_run_id = int(valid_df["trial_id"].max()) + 1

    if next_run_id > int(schedule["run_id"].max()):
        return None

    return next_run_id


def scheduled_run(schedule, run_id):
    if run_id is None or len(schedule) == 0:
        return None

    matches = schedule[schedule["run_id"] == run_id]

    if len(matches) == 0:
        return None

    return matches.iloc[0]


def schedule_progress(schedule, df):
    if len(schedule) == 0:
        return schedule.copy()

    valid_df = valid_submitted_data(df, schedule)
    completed_ids = set(valid_df["trial_id"].astype(int)) if len(valid_df) > 0 else set()
    progress = schedule.copy()
    progress["complete"] = progress["run_id"].isin(completed_ids)
    progress["status"] = np.where(progress["complete"], "Complete", "Incomplete")

    return progress


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


def condition_label(df):
    return (
        df["denomination"].astype(str)
        + " | "
        + df["decade"].astype(str)
        + " | "
        + df["posture"].astype(str)
    )


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


def gh_url():
    return f"https://api.github.com/repos/{github_repo}/contents/{github_path}"


def load_from_github():
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

    return r.json()


def add_row_to_github(new_row, schedule=None):
    old_df, sha = load_from_github()

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

    refreshed, _ = load_from_github()
    return refreshed


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
    refreshed_progress = schedule_progress(schedule, refreshed_valid_df)
    refreshed_next_run_id = next_run_id_from_data(refreshed_valid_df, schedule)
    refreshed_next_run = scheduled_run(schedule, refreshed_next_run_id)

    return (
        refreshed_df,
        refreshed_valid_df,
        refreshed_progress,
        refreshed_next_run_id,
        refreshed_next_run,
    )


def delete_trial_from_github(trial_id):
    old_df, sha = load_from_github()

    old_count = len(old_df)
    new_df = old_df[old_df["trial_id"] != trial_id].copy()

    if len(new_df) == old_count:
        raise RuntimeError(f"Trial ID {trial_id} was not found.")

    save_to_github(new_df, sha)

    refreshed, _ = load_from_github()
    return refreshed


def delete_valid_trial_from_github(trial_id, schedule):
    old_df, sha = load_from_github()
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
    old_df, sha = load_from_github()

    new_df = empty_data()
    save_to_github(new_df, sha)

    refreshed, _ = load_from_github()
    return refreshed


def run_anova(df):
    formula = (
        "proportion ~ C(denomination) * C(decade) * C(posture) "
        "+ C(flipper) + C(starting_side)"
    )

    model = smf.ols(formula, data=df).fit()
    table = anova_lm(model, typ=2)

    return model, table


def clean_term_name(term):
    term = str(term)

    replacements = {
        "C(denomination)": "denomination",
        "C(decade)": "decade",
        "C(posture)": "posture",
        "C(flipper)": "flipper",
        "C(starting_side)": "starting side",
        ":": " × "
    }

    for old, new in replacements.items():
        term = term.replace(old, new)

    return term


def make_qq_plot(model):
    residuals = pd.Series(model.resid).dropna()

    theoretical, ordered = stats.probplot(residuals, dist="norm", fit=False)

    qq_df = pd.DataFrame({
        "Theoretical quantiles": theoretical,
        "Ordered residuals": ordered
    })

    slope, intercept, r = stats.probplot(residuals, dist="norm", fit=True)[1]

    x_min = qq_df["Theoretical quantiles"].min()
    x_max = qq_df["Theoretical quantiles"].max()

    line_x = np.array([x_min, x_max])
    line_y = intercept + slope * line_x

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=qq_df["Theoretical quantiles"],
            y=qq_df["Ordered residuals"],
            mode="markers",
            name="Residuals"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name="Normal reference line"
        )
    )

    fig.update_layout(
        title="Normal Q-Q plot of residuals",
        xaxis_title="Theoretical normal quantiles",
        yaxis_title="Ordered residuals",
        template="plotly_white",
        height=550,
        margin=dict(l=20, r=20, t=70, b=40)
    )

    return fig


st.sidebar.header("Settings")

alpha = st.sidebar.number_input(
    "Significance level",
    min_value=0.001,
    max_value=0.20,
    value=0.05,
    step=0.01,
    format="%.3f"
)

data_mode = st.sidebar.radio(
    "Data shown in results tab",
    ["Real GitHub data", "Dummy data"],
    index=0
)

st.sidebar.write(f"Each run has **{FLIPS_PER_TRIAL} flips**.")

if st.sidebar.button("Reload data"):
    st.rerun()


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

run_schedule = load_run_schedule()
valid_df = valid_submitted_data(df, run_schedule)
next_run_id = next_run_id_from_data(valid_df, run_schedule)
next_run = scheduled_run(run_schedule, next_run_id)
run_progress = schedule_progress(run_schedule, valid_df)

if data_mode == "Dummy data":
    results_df = make_dummy_data()
else:
    results_df = valid_df


submit_tab, results_tab, instructions_tab = st.tabs([
    "Submit and edit data",
    "View results and analysis",
    "Experiment instructions",
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
            if st.button(
                "Heads",
                disabled=flips_recorded >= FLIPS_PER_TRIAL,
                use_container_width=True,
                type="primary"
            ):
                st.session_state[flip_state_key].append("Heads")
                st.rerun()
        with b2:
            if st.button(
                "Tails",
                disabled=flips_recorded >= FLIPS_PER_TRIAL,
                use_container_width=True
            ):
                st.session_state[flip_state_key].append("Tails")
                st.rerun()
        with b3:
            if st.button(
                "Undo",
                disabled=flips_recorded == 0,
                use_container_width=True
            ):
                st.session_state[flip_state_key].pop()
                st.rerun()
        with b4:
            if st.button(
                "Reset",
                disabled=flips_recorded == 0,
                use_container_width=True
            ):
                st.session_state[flip_state_key] = []
                st.rerun()

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
                st.rerun()
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
                st.rerun()
            except Exception as e:
                st.error("Submission failed.")
                st.write(str(e))

    st.divider()

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

        def highlight_completed_run(row):
            if row["status"] == "Complete":
                return ["background-color: rgba(46, 160, 67, 0.22)"] * len(row)
            return [""] * len(row)

        styled_progress = (
            progress_display[display_cols]
            .style
            .format(
                {
                    "submitted_heads": "{:.0f}",
                    "proportion_heads": "{:.1f}",
                },
                na_rep="None"
            )
            .apply(highlight_completed_run, axis=1)
        )

        st.dataframe(
            styled_progress,
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
                    st.rerun()
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
                    st.rerun()
                except Exception as e:
                    st.error("Erase failed.")
                    st.write(str(e))

    st.divider()
    st.caption(f"Saving to: `{github_repo}/{github_path}`")


with instructions_tab:
    st.subheader("Experiment instructions")

    st.markdown(f"**Vocabulary:** One run = {FLIPS_PER_TRIAL} flips.")

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


with results_tab:
    if data_mode == "Dummy data":
        st.info("Showing generated dummy data. This does not change the GitHub CSV.")
    else:
        st.info("Showing real submitted data from GitHub.")

    if len(results_df) == 0:
        st.info("No data has been submitted yet.")
        st.stop()

    st.subheader("Summary")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Trials", len(results_df))

    with m2:
        st.metric("Total flips", int(results_df["total"].sum()))

    with m3:
        st.metric("Total heads", int(results_df["heads"].sum()))

    with m4:
        st.metric("Overall p̂", f"{results_df['heads'].sum() / results_df['total'].sum():.3f}")

    summary_df = results_df.copy()
    summary_df["condition"] = condition_label(summary_df)

    summary = (
        summary_df
        .groupby("condition")
        .agg(
            trials=("trial_id", "count"),
            total_heads=("heads", "sum"),
            total_flips=("total", "sum"),
            mean_proportion=("proportion", "mean")
        )
        .reset_index()
    )

    summary["p_hat"] = summary["total_heads"] / summary["total_flips"]

    st.subheader("Summary by treatment condition")
    st.dataframe(summary.sort_values("condition"), use_container_width=True)

    st.divider()

    st.subheader("Test equality of mean proportions")

    st.write("Model used:")

    st.latex(
        r"""
        y_{ijklm}
        =
        \mu
        + \alpha_i
        + \beta_j
        + \gamma_k
        + (\alpha\beta)_{ij}
        + (\alpha\gamma)_{ik}
        + (\beta\gamma)_{jk}
        + (\alpha\beta\gamma)_{ijk}
        + \delta_l
        + \eta_m
        + \varepsilon_{ijklm}
        """
    )

    st.markdown(
        """
        where:

        - $y_{ijklm}$ = proportion of heads in one 10-flip trial  
        - $\\alpha_i$ = denomination effect  
        - $\\beta_j$ = decade effect  
        - $\\gamma_k$ = posture effect  
        - $\\delta_l$ = flipper/blocking effect  
        - $\\eta_m$ = starting-side effect  
        - $\\varepsilon_{ijklm}$ = random error
        """
    )

    model = None
    anova_table = None

    if len(results_df) < 10:
        st.warning("More data is needed before the ANOVA is meaningful.")
    else:
        try:
            model, anova_table = run_anova(results_df)

            clean_anova = anova_table.copy()
            clean_anova = clean_anova.reset_index().rename(columns={"index": "Term"})
            clean_anova["Term"] = clean_anova["Term"].apply(clean_term_name)

            clean_anova = clean_anova.rename(columns={
                "sum_sq": "Sum of squares",
                "df": "df",
                "F": "F statistic",
                "PR(>F)": "p-value"
            })

            for c in ["Sum of squares", "F statistic", "p-value"]:
                if c in clean_anova.columns:
                    clean_anova[c] = clean_anova[c].round(4)

            st.write("ANOVA table")
            st.dataframe(clean_anova, use_container_width=True, hide_index=True)

            sig = [
                clean_term_name(x)
                for x in anova_table[anova_table["PR(>F)"] < alpha].index.tolist()
            ]

            if sig:
                st.warning(
                    f"At alpha = {alpha}, at least one factor or interaction is significant. "
                    "This suggests that the condition means are not all equal."
                )
                st.write("Significant terms:")
                for x in sig:
                    st.write(f"- {x}")
            else:
                st.success(
                    f"At alpha = {alpha}, no factor or interaction is significant. "
                    "There is no evidence that the condition means differ."
                )

            with st.expander("Model details"):
                model_stats = pd.DataFrame({
                    "Metric": [
                        "Number of observations",
                        "Residual df",
                        "Model df",
                        "R-squared",
                        "Adjusted R-squared",
                        "F-statistic",
                        "Overall model p-value"
                    ],
                    "Value": [
                        int(model.nobs),
                        int(model.df_resid),
                        int(model.df_model),
                        round(model.rsquared, 4),
                        round(model.rsquared_adj, 4),
                        round(model.fvalue, 4) if model.fvalue is not None else np.nan,
                        round(model.f_pvalue, 4) if model.f_pvalue is not None else np.nan
                    ]
                })

                st.dataframe(model_stats, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error("ANOVA failed.")
            st.write(str(e))

    st.divider()

    st.subheader("Visualizations")

    plot_df = results_df.copy()
    plot_df["condition"] = condition_label(plot_df)

    main_plot_tab, interaction_plot_tab, block_plot_tab, distribution_plot_tab, diagnostics_tab = st.tabs([
        "Main plots",
        "Interaction plots",
        "Blocking checks",
        "Distribution",
        "Residual checks"
    ])

    with main_plot_tab:
        condition_plot = (
            plot_df
            .groupby("condition", as_index=False)
            .agg(mean_proportion=("proportion", "mean"))
            .sort_values("mean_proportion")
        )

        fig = px.bar(
            condition_plot,
            x="mean_proportion",
            y="condition",
            orientation="h",
            text="mean_proportion",
            title="Mean proportion of heads by treatment condition",
            labels={
                "mean_proportion": "Mean proportion heads",
                "condition": "Condition"
            },
            template="plotly_white"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            marker_line_width=0.5
        )

        fig.add_vline(
            x=0.5,
            line_dash="dash",
            line_width=2,
            annotation_text="Fair coin p = 0.5",
            annotation_position="top"
        )

        fig.update_xaxes(range=[0, 1])
        fig.update_layout(
            height=max(500, 30 * len(condition_plot)),
            margin=dict(l=20, r=40, t=70, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        fig = px.box(
            plot_df,
            x="denomination",
            y="proportion",
            color="denomination",
            points="all",
            title="Distribution of proportion heads by denomination",
            labels={
                "denomination": "Denomination",
                "proportion": "Proportion heads"
            },
            template="plotly_white"
        )

        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_width=2,
            annotation_text="Fair coin p = 0.5",
            annotation_position="top left"
        )

        fig.update_yaxes(range=[0, 1])
        fig.update_layout(
            height=500,
            showlegend=False,
            margin=dict(l=20, r=20, t=70, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        fig = px.box(
            plot_df,
            x="decade",
            y="proportion",
            color="decade",
            points="all",
            title="Distribution of proportion heads by decade",
            labels={
                "decade": "Decade",
                "proportion": "Proportion heads"
            },
            template="plotly_white"
        )

        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_width=2,
            annotation_text="Fair coin p = 0.5",
            annotation_position="top left"
        )

        fig.update_yaxes(range=[0, 1])
        fig.update_layout(
            height=500,
            showlegend=False,
            margin=dict(l=20, r=20, t=70, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        fig = px.box(
            plot_df,
            x="posture",
            y="proportion",
            color="posture",
            points="all",
            title="Distribution of proportion heads by posture",
            labels={
                "posture": "Posture",
                "proportion": "Proportion heads"
            },
            template="plotly_white"
        )

        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_width=2,
            annotation_text="Fair coin p = 0.5",
            annotation_position="top left"
        )

        fig.update_yaxes(range=[0, 1])
        fig.update_layout(
            height=500,
            showlegend=False,
            margin=dict(l=20, r=20, t=70, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

    with interaction_plot_tab:
        st.write("These plots show interactions among the design factors.")

        interaction_1, interaction_2, interaction_3 = st.tabs([
            "Denomination × decade",
            "Denomination × posture",
            "Decade × posture"
        ])

        with interaction_1:
            tmp = (
                plot_df
                .groupby(["denomination", "decade"], as_index=False)
                .agg(mean_proportion=("proportion", "mean"))
            )

            fig = px.line(
                tmp,
                x="denomination",
                y="mean_proportion",
                color="decade",
                markers=True,
                title="Mean proportion of heads by denomination and decade",
                labels={
                    "denomination": "Denomination",
                    "mean_proportion": "Mean proportion heads",
                    "decade": "Decade"
                },
                template="plotly_white"
            )

            fig.add_hline(
                y=0.5,
                line_dash="dash",
                line_width=2,
                annotation_text="Fair coin p = 0.5",
                annotation_position="top left"
            )

            fig.update_yaxes(range=[0, 1])
            fig.update_traces(line=dict(width=3), marker=dict(size=9))
            fig.update_layout(height=500, margin=dict(l=20, r=20, t=70, b=40))

            st.plotly_chart(fig, use_container_width=True)

        with interaction_2:
            tmp = (
                plot_df
                .groupby(["denomination", "posture"], as_index=False)
                .agg(mean_proportion=("proportion", "mean"))
            )

            fig = px.line(
                tmp,
                x="denomination",
                y="mean_proportion",
                color="posture",
                markers=True,
                title="Mean proportion of heads by denomination and posture",
                labels={
                    "denomination": "Denomination",
                    "mean_proportion": "Mean proportion heads",
                    "posture": "Posture"
                },
                template="plotly_white"
            )

            fig.add_hline(
                y=0.5,
                line_dash="dash",
                line_width=2,
                annotation_text="Fair coin p = 0.5",
                annotation_position="top left"
            )

            fig.update_yaxes(range=[0, 1])
            fig.update_traces(line=dict(width=3), marker=dict(size=9))
            fig.update_layout(height=500, margin=dict(l=20, r=20, t=70, b=40))

            st.plotly_chart(fig, use_container_width=True)

        with interaction_3:
            tmp = (
                plot_df
                .groupby(["decade", "posture"], as_index=False)
                .agg(mean_proportion=("proportion", "mean"))
            )

            fig = px.line(
                tmp,
                x="decade",
                y="mean_proportion",
                color="posture",
                markers=True,
                title="Mean proportion of heads by decade and posture",
                labels={
                    "decade": "Decade",
                    "mean_proportion": "Mean proportion heads",
                    "posture": "Posture"
                },
                template="plotly_white"
            )

            fig.add_hline(
                y=0.5,
                line_dash="dash",
                line_width=2,
                annotation_text="Fair coin p = 0.5",
                annotation_position="top left"
            )

            fig.update_yaxes(range=[0, 1])
            fig.update_traces(line=dict(width=3), marker=dict(size=9))
            fig.update_layout(height=500, margin=dict(l=20, r=20, t=70, b=40))

            st.plotly_chart(fig, use_container_width=True)

    with block_plot_tab:
        st.write("These plots check the blocking or nuisance factors.")

        block_1, block_2 = st.tabs(["Flipper", "Starting side"])

        with block_1:
            fig = px.box(
                plot_df,
                x="flipper",
                y="proportion",
                color="flipper",
                points="all",
                title="Distribution of proportion heads by flipper",
                labels={
                    "flipper": "Flipper",
                    "proportion": "Proportion heads"
                },
                template="plotly_white"
            )

            fig.add_hline(
                y=0.5,
                line_dash="dash",
                line_width=2,
                annotation_text="Fair coin p = 0.5",
                annotation_position="top left"
            )

            fig.update_yaxes(range=[0, 1])
            fig.update_layout(
                height=500,
                showlegend=False,
                margin=dict(l=20, r=20, t=70, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)

        with block_2:
            fig = px.box(
                plot_df,
                x="starting_side",
                y="proportion",
                color="starting_side",
                points="all",
                title="Distribution of proportion heads by starting side",
                labels={
                    "starting_side": "Starting side",
                    "proportion": "Proportion heads"
                },
                template="plotly_white"
            )

            fig.add_hline(
                y=0.5,
                line_dash="dash",
                line_width=2,
                annotation_text="Fair coin p = 0.5",
                annotation_position="top left"
            )

            fig.update_yaxes(range=[0, 1])
            fig.update_layout(
                height=500,
                showlegend=False,
                margin=dict(l=20, r=20, t=70, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)

    with distribution_plot_tab:
        st.write("Heads count distribution")

        fig = px.histogram(
            plot_df,
            x="heads",
            nbins=11,
            title="Distribution of heads counts per 10-flip trial",
            labels={
                "heads": "Heads out of 10",
                "count": "Number of trials"
            },
            template="plotly_white"
        )

        fig.update_xaxes(dtick=1)
        fig.update_layout(
            height=450,
            bargap=0.1,
            margin=dict(l=20, r=20, t=70, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

    with diagnostics_tab:
        st.write("Residual normality check")

        if model is None:
            st.warning("The model did not run, so residual checks are not available.")
        else:
            st.write(
                """
                The Q-Q plot checks whether the ANOVA residuals are approximately normal.
                If the points mostly follow the straight line, the normality assumption is reasonable.
                """
            )

            fig = make_qq_plot(model)
            st.plotly_chart(fig, use_container_width=True)

            residual_df = pd.DataFrame({
                "Residual": model.resid,
                "Fitted value": model.fittedvalues
            })

            st.divider()

            st.write("Residual histogram")

            fig = px.histogram(
                residual_df,
                x="Residual",
                nbins=15,
                title="Histogram of model residuals",
                labels={
                    "Residual": "Residual",
                    "count": "Frequency"
                },
                template="plotly_white"
            )

            fig.add_vline(
                x=0,
                line_dash="dash",
                line_width=2,
                annotation_text="0",
                annotation_position="top"
            )

            fig.update_layout(
                height=450,
                bargap=0.1,
                margin=dict(l=20, r=20, t=70, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            st.write("Residuals versus fitted values")

            fig = px.scatter(
                residual_df,
                x="Fitted value",
                y="Residual",
                title="Residuals versus fitted values",
                labels={
                    "Fitted value": "Fitted value",
                    "Residual": "Residual"
                },
                template="plotly_white"
            )

            fig.add_hline(
                y=0,
                line_dash="dash",
                line_width=2,
                annotation_text="0",
                annotation_position="top left"
            )

            fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=70, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)
