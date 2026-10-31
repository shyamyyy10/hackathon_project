from pathlib import Path
from datetime import datetime
import pandas as pd

CSV_PATH = Path("data/actions.csv")
COLUMNS = ["timestamp", "category", "condition", "action"]

def load_actions():
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(CSV_PATH)
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[COLUMNS]
    except Exception:
        return pd.DataFrame(columns=COLUMNS)

def log_action(category, condition, action):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = load_actions()
    row = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "category": category,
        "condition": condition,
        "action": action
    }])
    pd.concat([df, row], ignore_index=True).to_csv(CSV_PATH, index=False)

def summarize_actions(df):
    if df.empty:
        return {"total": 0, "reuse": 0, "recycle": 0}
    actions = df["action"].fillna("").str.lower()
    return {
        "total": int(len(df)),
        "reuse": int(actions.str.contains("reuse").sum()),
        "recycle": int(actions.str.contains("recycl").sum())
    }
