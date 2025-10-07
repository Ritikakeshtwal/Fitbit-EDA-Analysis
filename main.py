# -*- coding: utf-8 -*-
"""
Fitbit EDA: daily activity, sleep, weight, heart-rate
VS Code friendly script: saves CSV outputs and PNG plots.
"""

import os, warnings, sys, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Config ----------------
DATA_DIR = os.path.join("data_raw")
OUT_DIR  = os.path.join("files", "fitness_result")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

DAILY_CSV  = os.path.join(DATA_DIR, "dailyActivity_merged.csv")
SLEEP_CSV  = os.path.join(DATA_DIR, "sleepDay_merged.csv")
WEIGHT_CSV = os.path.join(DATA_DIR, "weightLogInfo_merged.csv")
HEART_CSV  = os.path.join(DATA_DIR, "heartrate_seconds_merged.csv")

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 200)
sns.set(style="whitegrid", context="notebook")

def log(msg): 
    print(f"[Fitbit-EDA] {msg}")

# ---------------- Load ----------------
def read_csv_safe(path):
    if not os.path.exists(path):
        log(f"Missing file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        log(f"Failed to read {path}: {e}")
        return None

daily  = read_csv_safe(DAILY_CSV)
sleep  = read_csv_safe(SLEEP_CSV)
weight = read_csv_safe(WEIGHT_CSV)
heart  = read_csv_safe(HEART_CSV)

for name, df in {"daily":daily,"sleep":sleep,"weight":weight,"heart":heart}.items():
    if df is None:
        log(f"WARNING: {name} not loaded")

def norm_id(df):
    if df is None or "Id" not in df.columns:
        return df
    return df.assign(Id=pd.to_numeric(df["Id"], errors="coerce").astype("Int64").astype(str))

daily  = norm_id(daily)
sleep  = norm_id(sleep)
weight = norm_id(weight)
heart  = norm_id(heart)

# ---------------- Parse dates ----------------
if daily is not None and "ActivityDate" in daily.columns:
    daily["ActivityDate"] = pd.to_datetime(daily["ActivityDate"], format="%m/%d/%Y", errors="coerce")
    daily["date"] = daily["ActivityDate"].dt.normalize()

if sleep is not None and "SleepDay" in sleep.columns:
    sleep["SleepDay"] = pd.to_datetime(sleep["SleepDay"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    sleep["date"] = sleep["SleepDay"].dt.normalize()

if weight is not None and "Date" in weight.columns:
    try:
        weight["Date"] = pd.to_datetime(weight["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="raise")
    except Exception:
        weight["Date"] = pd.to_datetime(weight["Date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")

if heart is not None:
    time_col = "Time" if "Time" in heart.columns else ("ActivitySecond" if "ActivitySecond" in heart.columns else None)
    if time_col is not None:
        try:
            heart[time_col] = pd.to_datetime(heart[time_col], format="%m/%d/%Y %I:%M:%S %p", errors="raise")
        except Exception:
            heart[time_col] = pd.to_datetime(heart[time_col], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        heart["date"] = heart[time_col].dt.normalize()
    else:
        log("WARNING: Heart file missing Time/ActivitySecond column")

# ---------------- Dedupe ----------------
for d in [daily, sleep]:
    if d is not None and "Id" in d.columns and "date" in d.columns:
        d.sort_values(["Id","date"], inplace=True)
        d.drop_duplicates(["Id","date"], keep="first", inplace=True)

# ---------------- HR daily agg ----------------
if heart is not None and "Value" in heart.columns and "Id" in heart.columns and "date" in heart.columns:
    hr_day = (heart.groupby(["Id","date"], as_index=False, observed=True)
                .agg(AvgHR=("Value","mean"),
                     MaxHR=("Value","max"),
                     MinHR=("Value","min"),
                     HRCount=("Value","size")))
    hr_day[["AvgHR","MaxHR","MinHR"]] = hr_day[["AvgHR","MaxHR","MinHR"]].round(1)
else:
    hr_day = pd.DataFrame(columns=["Id","date","AvgHR","MaxHR","MinHR","HRCount"])
log(f"hr_day rows: {len(hr_day)}")

# ---------------- Build analysis df ----------------
def safe_cols(df, cols):
    return [c for c in cols if c in (df.columns if df is not None else [])]

keep = ["Id","date","TotalSteps","Calories","SedentaryMinutes","VeryActiveMinutes","LightlyActiveMinutes"]
df = daily[safe_cols(daily, keep)].copy() if daily is not None else pd.DataFrame(columns=keep)

if sleep is not None:
    df = df.merge(sleep[safe_cols(sleep, ["Id","date","TotalMinutesAsleep"])], on=["Id","date"], how="left")
df = df.merge(hr_day, on=["Id","date"], how="left")

num_cols = ["TotalSteps","Calories","SedentaryMinutes","VeryActiveMinutes","LightlyActiveMinutes",
            "TotalMinutesAsleep","AvgHR","MaxHR","MinHR","HRCount"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if "date" in df.columns:
    df["weekday"] = pd.to_datetime(df["date"]).dt.day_name()

# QC flags
cal_median = df["Calories"].median(skipna=True) if "Calories" in df.columns else np.nan
df["zero_step_high_cal"] = (df.get("TotalSteps",0)==0) & (df.get("Calories",0)>cal_median)
df["valid_row"] = ~df["zero_step_high_cal"]
df_valid = df[df["valid_row"]].copy()
log(f"df_valid rows: {len(df_valid)}")

# ---------------- Aggregations ----------------
def groupmean(frame, by, cols):
    if frame.empty: 
        return pd.DataFrame(columns=[by]+cols)
    out = frame.groupby(by, as_index=False, observed=True).agg({c:"mean" for c in cols})
    return out

agg_by_date = groupmean(
    df_valid, "date",
    ["TotalSteps","Calories","SedentaryMinutes","VeryActiveMinutes","LightlyActiveMinutes",
     "TotalMinutesAsleep","AvgHR"]
)

agg_by_weekday = groupmean(
    df_valid, "weekday",
    ["TotalSteps","Calories","TotalMinutesAsleep","AvgHR"]
)

# Step segments
bins = [-1, 5000, 10000, np.inf]
labels = ["<5k","5k-10k",">10k"]
if "TotalSteps" in df_valid.columns:
    df_valid["steps_bucket"] = pd.cut(df_valid["TotalSteps"], bins=bins, labels=labels, include_lowest=True, ordered=True)
else:
    df_valid["steps_bucket"] = pd.Categorical([])

seg = (df_valid.groupby("steps_bucket", as_index=False, observed=True)
       .agg(Calories=("Calories","mean"),
            TotalMinutesAsleep=("TotalMinutesAsleep","mean"),
            SedentaryMinutes=("SedentaryMinutes","mean"),
            AvgHR=("AvgHR","mean"))).round(1)

# ---------------- Save CSVs ----------------
def save_csv(obj, name):
    path = os.path.join(OUT_DIR, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj.to_csv(path, index=False, encoding="utf-8")
    log(f"Saved: {path}")

save_csv(df_valid, "clean_daily_sleep.csv")
save_csv(agg_by_date, "agg_by_date.csv")
save_csv(agg_by_weekday, "agg_by_weekday.csv")
save_csv(seg, "segments_steps.csv")
save_csv(hr_day, "heartrate_daily.csv")

# ---------------- Plots (headless) ----------------
def savefig(fig, name):
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved plot: {path}")

# 1) Steps over time
if not agg_by_date.empty and "TotalSteps" in agg_by_date.columns:
    fig, ax = plt.subplots(figsize=(10,4))
    t = agg_by_date.sort_values("date")
    ax.plot(t["date"], t["TotalSteps"], color="#1f77b4", linewidth=2)
    ax.set_title("Average Daily Steps"); ax.set_xlabel("Date"); ax.set_ylabel("Steps")
    savefig(fig, "steps_over_time.png")

# 2) Calories vs Steps
if not df_valid.empty:
    fig, ax = plt.subplots(figsize=(6,5))
    samp = df_valid.sample(min(3000, len(df_valid)), random_state=42)
    sns.regplot(data=samp, x="TotalSteps", y="Calories",
                scatter_kws={"alpha":0.25, "s":12}, line_kws={"color":"red"}, ax=ax)
    ax.set_title("Calories vs Steps")
    savefig(fig, "calories_vs_steps.png")

# 3) Weekday bars
if not agg_by_weekday.empty:
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    w = (agg_by_weekday.set_index("weekday")
         .reindex(order)
         .reset_index())
    fig, ax = plt.subplots(1,2, figsize=(12,4), sharex=True)
    sns.barplot(data=w, x="weekday", y="TotalSteps", ax=ax[0], color="#1f77b4")
    ax[0].set_title("Avg Steps by Weekday"); ax[0].tick_params(axis="x", rotation=30)
    sns.barplot(data=w, x="weekday", y="TotalMinutesAsleep", ax=ax[1], color="#2ca02c")
    ax[1].set_title("Avg Sleep by Weekday"); ax[1].tick_params(axis="x", rotation=30)
    savefig(fig, "weekday_bars.png")

# 4) Segments comparison
if not seg.empty:
    melt_cols = [c for c in ["Calories","TotalMinutesAsleep","SedentaryMinutes"] if c in seg.columns]
    m = seg.melt(id_vars="steps_bucket", value_vars=melt_cols, var_name="Metric", value_name="Value")
    fig, ax = plt.subplots(figsize=(7,4))
    sns.barplot(data=m, x="Metric", y="Value", hue="steps_bucket", ax=ax)
    ax.set_title("Metrics by Steps Bucket"); ax.set_xlabel(""); ax.set_ylabel("Average")
    savefig(fig, "segments_compare.png")

# 5) Average HR over time
if "AvgHR" in agg_by_date.columns and agg_by_date["AvgHR"].notna().any():
    fig, ax = plt.subplots(figsize=(10,4))
    t = agg_by_date.sort_values("date")
    ax.plot(t["date"], t["AvgHR"], color="#ff7f0e", linewidth=2)
    ax.set_title("Average Daily Heart Rate"); ax.set_xlabel("Date"); ax.set_ylabel("Avg HR (bpm)")
    savefig(fig, "avg_hr_over_time.png")

# 6) Sleep distribution
if "TotalMinutesAsleep" in df_valid.columns and df_valid["TotalMinutesAsleep"].notna().any():
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df_valid["TotalMinutesAsleep"].dropna(), bins=30, kde=True, color="#2ca02c", ax=ax)
    ax.axvline(420, color="red", linestyle="--", label="7h target"); ax.legend()
    ax.set_title("Sleep Minutes Distribution"); ax.set_xlabel("Minutes"); ax.set_ylabel("Days")
    savefig(fig, "sleep_distribution.png")

# 7) Sedentary vs Steps
if not df_valid.empty:
    fig, ax = plt.subplots(figsize=(6,5))
    samp = df_valid.sample(min(3000, len(df_valid)), random_state=42)
    sns.regplot(data=samp, x="TotalSteps", y="SedentaryMinutes",
                scatter_kws={"alpha":0.25, "s":12}, line_kws={"color":"red"}, ax=ax)
    ax.set_title("Sedentary Minutes vs Steps")
    savefig(fig, "sedentary_vs_steps.png")

log(f"Done. Outputs in: {OUT_DIR}")