"""
Streamlit UI for the Moral AI choice predictor.

Card-game style flow:
    Menu  -> enter personality + demographics, press Start
    Play  -> one scenario card at a time:
                 - see image + description
                 - swipe LEFT or RIGHT to make a choice
                 - model's prediction is revealed
                 - press Next card
    End   -> session summary with agreement rate

Run:
    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model.joblib"
META_PATH = ROOT / "model_meta.json"
IMAGES_DIR = ROOT / "images"

SCENARIOS = {
    "1. Fitness": {
        "scenariotype": "Fitness",
        "default_choice": "Athletic people",
        "nondefault_choice": "Overweight people",
        "numberofcharacters": 2,
        "diffnumberofcharacters": 0,
        "barrier": 0,
        "crossingsignal": 2,
        "intervention": 0,
        "pedped": 1,
        "image": "fitness.png",
        "description": "Athletic (crossing on RED) vs Overweight (crossing on GREEN)",
    },
    "2. Age": {
        "scenariotype": "Age",
        "default_choice": "Young people",
        "nondefault_choice": "Elderly people",
        "numberofcharacters": 1,
        "diffnumberofcharacters": 0,
        "barrier": 0,
        "crossingsignal": 0,
        "intervention": 0,
        "pedped": 1,
        "image": "age.png",
        "description": "Young vs Elderly",
    },
    "3. Species": {
        "scenariotype": "Species",
        "default_choice": "Pets",
        "nondefault_choice": "Humans",
        "numberofcharacters": 4,
        "diffnumberofcharacters": 0,
        "barrier": 0,
        "crossingsignal": 2,
        "intervention": 0,
        "pedped": 1,
        "image": "species.png",
        "description": "Pets (crossing on GREEN) vs Humans (crossing on RED)",
    },
    "4. Passenger Sacrifice (Barrier)": {
        "scenariotype": "Utilitarian",
        "default_choice": "Passengers (sacrificed via wall)",
        "nondefault_choice": "Pedestrians",
        "numberofcharacters": 4,
        "diffnumberofcharacters": 0,
        "barrier": 1,
        "crossingsignal": 1,
        "intervention": 0,
        "pedped": 0,
        "image": "barrier.png",
        "description": "4 Passengers (straight into wall) vs 4 Pedestrians (swerve)",
    },
    "5. Random": {
        "scenariotype": "Random",
        "default_choice": "Group B",
        "nondefault_choice": "Group A",
        "numberofcharacters": 3,
        "diffnumberofcharacters": 0,
        "barrier": 0,
        "crossingsignal": 2,
        "intervention": 0,
        "pedped": 1,
        "image": "random.png",
        "description": "Group B (3 people, crossing on RED) vs Group A (3 people, crossing on GREEN)",
    },
    "6. Passengers vs Pets": {
        "scenariotype": "Species",
        "default_choice": "Pets",
        "nondefault_choice": "Passengers (sacrificed via wall)",
        "numberofcharacters": 5,
        "diffnumberofcharacters": 0,
        "barrier": 1,
        "crossingsignal": 1,
        "intervention": 0,
        "pedped": 0,
        "image": "pets.png",
        "description": "5 Pets (straight, crossing on GREEN) vs 5 Car passengers (swerve into wall)",
    },
}

SCENARIO_NAMES = list(SCENARIOS.keys())


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model…")
def load_model():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        st.error(
            "`model.joblib` or `model_meta.json` not found.\n\n"
            "Train the model first by running:\n\n```\npython train_model.py\n```"
        )
        st.stop()
    pipe = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return pipe, meta


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def build_feature_row(profile: dict, scenario: dict, meta: dict) -> pd.DataFrame:
    row = {col: meta["defaults"].get(col) for col in meta["feature_columns"]}

    # Big Five raw IPIP-120 totals (24-120) used as-is.
    for col in ["ipip_openness", "ipip_conscientiousness",
                "ipip_extraversion", "ipip_agreeableness", "ipip_neuroticism"]:
        if col in row:
            val = profile.get(col)
            if val is not None:
                row[col] = float(val)

    if "age" in row and profile.get("age") is not None:
        row["age"] = float(profile["age"])
    if "gender" in row and profile.get("gender"):
        row["gender"] = profile["gender"].lower()

    for key in ["scenariotype", "barrier", "crossingsignal", "intervention",
                "pedped", "numberofcharacters", "diffnumberofcharacters"]:
        if key in row and key in scenario:
            row[key] = scenario[key]
    if "defaultchoiceisomission" in row:
        row["defaultchoiceisomission"] = 1
    return pd.DataFrame([row], columns=meta["feature_columns"])


def predict(pipe, profile: dict, scenario: dict, meta: dict):
    X = build_feature_row(profile, scenario, meta)
    pred = int(pipe.predict(X)[0])
    proba = pipe.predict_proba(X)[0]
    return pred, float(np.max(proba))


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "stage": "menu",          # menu | play | end
        "card_idx": 0,
        "card_phase": "choose",   # choose | reveal
        "history": [],
        "profile": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_session():
    st.session_state.stage = "menu"
    st.session_state.card_idx = 0
    st.session_state.card_phase = "choose"
    st.session_state.history = []
    st.session_state.profile = None


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_menu():
    # Hero header
    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 2.5rem 1rem 2rem 1rem;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #6dd5ed 100%);
            border-radius: 18px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        ">
            <div style="font-size: 4rem; line-height: 1;">🚗⚖️🧠</div>
            <h1 style="color: white; font-size: 2.6rem; margin: 0.5rem 0 0.3rem 0; font-weight: 800;">
                Moral AI Predictor
            </h1>
            <p style="color: #e6f0ff; font-size: 1.05rem; margin: 0; font-style: italic;">
                Can an AI guess your moral choices from your personality?
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Centered description
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        st.markdown(
            """
            <div style="text-align: center; font-size: 1rem; line-height: 1.6; color: #444; margin-bottom: 1.5rem;">
                A card-game version of the
                <a href="https://moralmachine.net" target="_blank">Moral Machine</a>
                experiment.<br>
                Enter your Big Five scores from
                <a href="https://bigfive-test.com" target="_blank">bigfive-test.com</a>,
                then play through trolley-style scenarios one card at a time.<br>
                After each choice, the AI reveals what someone with <em>your</em> personality
                was predicted to do.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.subheader("👤 Your Big Five Personality")
    st.caption("Enter the raw trait totals (24–120) from your bigfive-test.com results page.")

    c1, c2 = st.columns(2)
    with c1:
        openness = st.slider("🌱 Openness", 24, 120, 72, 1)
        conscientiousness = st.slider("🎯 Conscientiousness", 24, 120, 72, 1)
        extraversion = st.slider("🎉 Extraversion", 24, 120, 72, 1)
    with c2:
        agreeableness = st.slider("🤝 Agreeableness", 24, 120, 72, 1)
        neuroticism = st.slider("🌊 Neuroticism", 24, 120, 72, 1)

    st.markdown("---")
    st.subheader("📋 Demographics")
    c3, c4 = st.columns(2)
    age = c3.number_input("Age", 10, 100, 25, 1)
    gender = c4.selectbox("Gender", ["female", "male"])

    st.markdown("---")
    _, mid_btn, _ = st.columns([1, 2, 1])
    with mid_btn:
        if st.button("▶️  Start the game", type="primary", use_container_width=True):
            st.session_state.profile = {
                "ipip_openness": openness,
                "ipip_conscientiousness": conscientiousness,
                "ipip_extraversion": extraversion,
                "ipip_agreeableness": agreeableness,
                "ipip_neuroticism": neuroticism,
                "age": age,
                "gender": gender,
            }
            st.session_state.stage = "play"
            st.session_state.card_idx = 0
            st.session_state.card_phase = "choose"
            st.session_state.history = []
            st.rerun()


def handle_choice(pipe, meta, scenario, name, user_default: bool):
    pred, conf = predict(pipe, st.session_state.profile, scenario, meta)
    model_default = bool(pred == 1)
    st.session_state.history.append({
        "scenario": name,
        "user": scenario["default_choice"] if user_default else scenario["nondefault_choice"],
        "model": scenario["default_choice"] if model_default else scenario["nondefault_choice"],
        "user_default": user_default,
        "model_default": model_default,
        "confidence": conf,
        "agree": user_default == model_default,
    })
    st.session_state.card_phase = "reveal"


def page_play(pipe, meta):
    idx = st.session_state.card_idx
    name = SCENARIO_NAMES[idx]
    scenario = SCENARIOS[name]

    st.progress((idx + 1) / len(SCENARIO_NAMES),
                text=f"Card {idx + 1} of {len(SCENARIO_NAMES)} — {name}")

    # --- Card image + description ---
    card = st.container(border=True)
    with card:
        img_path = IMAGES_DIR / scenario["image"]
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.info(f"📷 Drop an image at `images/{scenario['image']}` to illustrate this scenario.")

        st.markdown(f"### {name}")
        st.write(scenario["description"])

        n_def = scenario["numberofcharacters"]
        n_non = n_def - scenario["diffnumberofcharacters"]
        st.markdown(
            f"- 👈 **LEFT** to save **{n_def} × {scenario['default_choice']}** (stay straight)\n"
            f"- 👉 **RIGHT** to save **{n_non} × {scenario['nondefault_choice']}** (swerve)"
        )
        if scenario.get("crossingsignal") == 2:
            st.warning("⚠️ A red traffic light is involved in this scenario.")
        if scenario.get("barrier") == 1:
            st.warning("🛡️ Swerving means hitting a wall, sacrificing the passengers.")

    st.markdown("")

    if st.session_state.card_phase == "choose":
        c1, c2 = st.columns(2)
        if c1.button(f"👈 Save {scenario['default_choice']}",
                     use_container_width=True, key=f"left_{idx}"):
            handle_choice(pipe, meta, scenario, name, user_default=True)
            st.rerun()
        if c2.button(f"Save {scenario['nondefault_choice']} 👉",
                     use_container_width=True, key=f"right_{idx}"):
            handle_choice(pipe, meta, scenario, name, user_default=False)
            st.rerun()

    else:  # reveal
        last = st.session_state.history[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("👤 You chose", f"Save {last['user']}")
        c2.metric("🤖 Model chose", f"Save {last['model']}",
                  delta=f"{last['confidence']*100:.1f}% confidence")
        c3.metric("🤝 Agreement", "✅ Match" if last["agree"] else "❌ Disagree")

        is_last = idx == len(SCENARIO_NAMES) - 1
        label = "🏁 See results" if is_last else "➡️ Next card"
        if st.button(label, type="primary", use_container_width=True, key=f"next_{idx}"):
            if is_last:
                st.session_state.stage = "end"
            else:
                st.session_state.card_idx += 1
                st.session_state.card_phase = "choose"
            st.rerun()


def page_end():
    st.title("🏁 Session results")
    hist_df = pd.DataFrame(st.session_state.history)
    if hist_df.empty:
        st.warning("No scenarios played.")
        if st.button("Back to menu"):
            reset_session()
            st.rerun()
        return

    agree_rate = hist_df["agree"].mean() * 100
    avg_conf = hist_df["confidence"].mean() * 100
    s1, s2, s3 = st.columns(3)
    s1.metric("Cards played", len(hist_df))
    s2.metric("Agreement rate", f"{agree_rate:.1f}%")
    s3.metric("Avg confidence", f"{avg_conf:.1f}%")

    st.dataframe(
        hist_df[["scenario", "user", "model", "agree", "confidence"]],
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")
    c1, c2 = st.columns(2)
    if c1.button("🔁 Play again with same profile", use_container_width=True):
        st.session_state.stage = "play"
        st.session_state.card_idx = 0
        st.session_state.card_phase = "choose"
        st.session_state.history = []
        st.rerun()
    if c2.button("🏠 Back to menu", use_container_width=True):
        reset_session()
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Moral AI Predictor", page_icon="🚗", layout="centered")
init_state()
pipe, meta = load_model()

stage = st.session_state.stage
if stage == "menu":
    page_menu()
elif stage == "play":
    with st.sidebar:
        st.header("Session")
        st.write(f"Card {st.session_state.card_idx + 1} / {len(SCENARIO_NAMES)}")
        played = len(st.session_state.history)
        if played:
            agree = sum(h["agree"] for h in st.session_state.history) / played * 100
            st.metric("Agreement so far", f"{agree:.0f}%")
        if st.button("🚪 Quit to menu"):
            reset_session()
            st.rerun()
    page_play(pipe, meta)
elif stage == "end":
    page_end()
