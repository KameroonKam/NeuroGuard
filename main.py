from dotenv import load_dotenv
load_dotenv()

import os
import sqlite3
from typing import Dict, Any, Optional

import requests
from flask import Flask, render_template, request, session, jsonify, redirect

from ml_model import get_mental_state, retrain_model, MODEL_PATH
from SQLfile import create_user_tables

app = Flask(__name__)

# Use a real secret key in .env for sessions (fallback keeps dev working)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")


# Gemini config (optional)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_NAME = "models/gemini-2.5-flash"
GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    if GEMINI_API_KEY
    else None
)


def _db_connect():
    # check_same_thread=False can help if you ever move to multi-threaded server
    return sqlite3.connect("User_Data.db", check_same_thread=False)


def _get_full_history() -> str:
    with _db_connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT previous_suggestion FROM data WHERE previous_suggestion IS NOT NULL")
        rows = cur.fetchall()
        items = [row[0] for row in rows if row and row[0]]
        return "\n- ".join(items)


def _update_latest_row(mental_state: int, summary_text: str) -> None:
    with _db_connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM data ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return
        latest_id = row[0]
        cur.execute(
            "UPDATE data SET emotional_state = ?, previous_suggestion = ? WHERE id = ?",
            (mental_state, summary_text.strip(), latest_id),
        )
        conn.commit()


def _fallback_suggestions(mental_state: int, user_data: Dict[str, Any]) -> str:
    # Simple offline tips when Gemini key is missing
    if mental_state < 35:
        return (
            "1) Take a 10–15 minute screen break and do slow breathing (4 seconds in, 6 out) for 3 minutes.\n"
            "2) Prioritise sleep tonight: aim for a consistent bedtime and reduce screens 30 minutes before bed.\n"
            "3) Add a short walk outside (even 10 minutes) to reset attention and stress levels."
        )
    if mental_state < 70:
        return (
            "1) Do a quick body reset: stretch your neck/shoulders for 2 minutes and drink water.\n"
            "2) Reduce screen time for the next hour (notifications off) and finish one small task.\n"
            "3) Get daylight if possible and add 15–20 minutes of light activity."
        )
    return (
        "1) Maintain what works: keep sleep stable and take short breaks between focused sessions.\n"
        "2) Protect focus: batch notifications and use a 25/5 timer (25 min work, 5 min break).\n"
        "3) Do something restorative today (walk, journaling, or a calm hobby) to prevent burnout."
    )


def _call_gemini(prompt: str) -> Optional[str]:
    if not GEMINI_API_URL:
        return None

    try:
        resp = requests.post(
            GEMINI_API_URL,
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=20,
        )
    except Exception:
        return None

    if resp.status_code != 200:
        return None

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return None


def generate_suggestions(mental_state: int, user_data: Dict[str, Any], history_text: str = "") -> str:
    # If no Gemini key, return offline tips
    if not GEMINI_API_URL:
        return _fallback_suggestions(mental_state, user_data)

    prompt = (
        "You are NeuroGuard, an advanced mental wellness AI from the year 2080.\n"
        "Offer exactly three personalised tips to help the user improve or stabilise their mental well-being.\n"
        "Each tip must be practical and concise. Respond with a numbered list only. Use British spelling.\n\n"
        f"Estimated mental state score: {mental_state}/100\n"
        f"Weekday: {user_data.get('weekday')}\n"
        f"Sleep: {user_data.get('sleep_duration_hours')} hours\n"
        f"Screen time: {user_data.get('screen_time_minutes')} minutes\n"
        f"Physical activity: {user_data.get('physical_activity_minutes')} minutes\n"
        f"Safety perception: {user_data.get('safety')}/100\n"
        f"Sunlight exposure: {user_data.get('sunlight_hours')} hours\n"
        f"Hour of day: {user_data.get('hour')}\n"
        f"Daily goal progress: {user_data.get('daily_goal_progression')}/100\n"
    )

    if history_text:
        prompt += f"\nUser history (previous advice):\n- {history_text}\n"

    text = _call_gemini(prompt)
    return text if text else _fallback_suggestions(mental_state, user_data)


def summarise_for_history(suggestion_text: str) -> str:
    if not GEMINI_API_URL:
        first_line = suggestion_text.splitlines()[0].strip() if suggestion_text else ""
        return first_line[:200] if first_line else "Offline summary saved."

    summary_prompt = (
        "Summarise the following mental wellness advice into one short paragraph to store as historical context. "
        "Return only the summary, no title.\n\n"
        f"{suggestion_text}"
    )
    text = _call_gemini(summary_prompt)
    return text if text else "Summary generation failed."


@app.route("/")
def home():
    return render_template("1_mainPage.html")


@app.route("/input")
def input_page():
    return render_template("2_InputPage.html")


@app.route("/reset")
def reset():
    # Clears the last result so a new run starts cleanly
    session.clear()
    return redirect("/")


@app.route("/results")
def results_page():
    predicted = session.get("predicted_state")
    suggestion = session.get("suggestion_text")

    # If user opens /results directly without submitting, show friendly text
    if predicted is None or suggestion is None:
        return render_template(
            "3_ResultPage.html",
            value1=0,
            value2="Submit your data to get results.",
        )

    return render_template("3_ResultPage.html", value1=predicted, value2=suggestion)


@app.route("/submit", methods=["POST"])
def handle_submission():
    data = request.get_json(force=True) or {}

    # Build ML input (use the keys the frontend sends)
    latest_data = {
        "sleep_duration_hours": float(data.get("sleep_duration", 0) or 0),
        "screen_time_minutes": int(data.get("screen", 0) or 0),
        "physical_activity_minutes": int(data.get("activity", 0) or 0),
        "hour": int(data.get("hours", 0) or 0),  # frontend key: "hours"
        "weekday": int(data.get("weekday", 0) or 0),
        "sunlight_hours": int(data.get("sunlight", 0) or 0),
        "safety": int(data.get("safety", 0) or 0),
        "daily_goal_progression": int(data.get("goals", 0) or 0),
    }

    # Optional: mark session as "processing" so /results could show a loading state if you add it later
    session["predicted_state"] = None
    session["suggestion_text"] = None

    # Save raw input to DB (hour key is already fixed + include daily_goal_progress)
    with _db_connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO data (
                user, emotional_state, sleep_duration_hours, screen_time_minutes,
                physical_activity_minutes, hour, weekday, sunlight_hours, safety, daily_goal_progress
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default_user",
                None,
                latest_data["sleep_duration_hours"],
                latest_data["screen_time_minutes"],
                latest_data["physical_activity_minutes"],
                latest_data["hour"],
                latest_data["weekday"],
                latest_data["sunlight_hours"],
                latest_data["safety"],
                latest_data["daily_goal_progression"],
            ),
        )
        conn.commit()

    # Predict
    predicted_state = get_mental_state(latest_data)

    # Suggestions (Gemini or offline)
    history_text = _get_full_history()
    suggestion_text = generate_suggestions(predicted_state, latest_data, history_text=history_text)

    # Store in session so /results can show it reliably
    session["predicted_state"] = predicted_state
    session["suggestion_text"] = suggestion_text

    # Save summary to DB
    summary_text = summarise_for_history(suggestion_text)
    _update_latest_row(predicted_state, summary_text)

    # IMPORTANT: return redirect url so frontend waits for JSON and then navigates
    # This avoids the "submit again / refresh" issue caused by session cookie timing.
    return jsonify({"ok": True, "redirect": "/results"})


def ensure_ready():
    create_user_tables()
    if not os.path.exists(MODEL_PATH):
        retrain_model(print_metrics=True)


if __name__ == "__main__":
    ensure_ready()
    app.run(debug=True, port=4000)
