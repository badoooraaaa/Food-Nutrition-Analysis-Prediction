"""
app.py — Baby Food Quality Analyzer
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from recommendation import compute_recommendation_score
from label_map import LABEL_MAP, LABEL_COLORS
from meals_data import BABY_MEALS
from growth_tracker import render_growth_tracker

st.set_page_config(page_title="🍼 Baby Food Quality Analyzer", page_icon="🍼",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');
html,body,[class*="css"]{font-family:'Nunito',sans-serif;}
.main{background:#FFF8F0;}
.block-container{padding-top:1.5rem;}

/* ─── Cards ─── */
.card{background:#fff;border-radius:20px;padding:24px;
      box-shadow:0 6px 24px rgba(0,0,0,0.07);margin-bottom:18px;
      transition:transform .2s,box-shadow .2s;}
.card:hover{transform:translateY(-3px);box-shadow:0 10px 32px rgba(0,0,0,0.10);}

/* ─── Big Score Ring ─── */
.score-wrap{text-align:center;padding:10px 0;}
.score-label{font-size:.82rem;color:#999;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;}

/* ─── Nutrient Pills ─── */
.pill-row{display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;}
.pill{display:inline-flex;align-items:center;gap:6px;
      padding:7px 14px;border-radius:30px;font-size:.82rem;font-weight:700;}
.pill-good{background:#E8FBF4;color:#06D6A0;border:1.5px solid #06D6A020;}
.pill-warn{background:#FFF8E6;color:#F0A500;border:1.5px solid #F0A50020;}
.pill-bad {background:#FFE9E9;color:#E74C3C;border:1.5px solid #E74C3C20;}

/* ─── Recommendation Banner ─── */
.rec-banner{padding:22px 28px;border-radius:20px;border-left:7px solid;
            margin-top:10px;font-size:1.05rem;font-weight:700;line-height:1.8;}
.rec-sub{font-weight:500;font-size:.92rem;opacity:.85;margin-top:4px;}

/* ─── Tip Cards ─── */
.tip-card{display:flex;align-items:flex-start;gap:12px;
          background:#F9F9FF;border-radius:14px;padding:12px 16px;
          margin-bottom:8px;font-size:.88rem;font-weight:600;color:#3D3D3D;}

/* ─── Progress Bar ─── */
.pb-wrap{margin-bottom:14px;}
.pb-label{display:flex;justify-content:space-between;
          font-size:.83rem;font-weight:700;color:#555;margin-bottom:5px;}
.pb-track{background:#F0F0F0;border-radius:10px;height:13px;overflow:hidden;}
.pb-fill{height:13px;border-radius:10px;transition:width .8s cubic-bezier(.4,0,.2,1);}

/* ─── Sidebar ─── */
.sidebar-logo{font-size:1.6rem;font-weight:900;color:#FF8C69;margin-bottom:2px;}
.sidebar-sub{font-size:.8rem;color:#999;margin-bottom:16px;}

/* ─── Button ─── */
.stButton>button{background:linear-gradient(135deg,#FF8C69,#FF5E5E);color:#fff;
  border:none;border-radius:14px;font-family:'Nunito',sans-serif;
  font-weight:800;font-size:1.05rem;padding:13px 0;width:100%;
  letter-spacing:.5px;transition:opacity .2s,transform .2s;box-shadow:0 4px 14px #FF8C6940;}
.stButton>button:hover{opacity:.88;transform:translateY(-1px);}

h1,h2,h3{font-family:'Nunito',sans-serif;font-weight:900;}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return (joblib.load("food_quality_model.pkl"),
                joblib.load("scaler.pkl"),
                joblib.load("feature_cols.pkl"),
                joblib.load("idx_to_label.pkl"))
    except FileNotFoundError:
        return None, None, None, None

@st.cache_data
def load_data():
    df = pd.read_csv("food_Ingredients.csv")
    df = df.dropna(subset=["quality"]).fillna(df.median(numeric_only=True))
    return df

@st.cache_resource(show_spinner="Initializing XAI Explainers...")
def load_explainers(_model, _scaler, _df, _feature_cols):
    if _model is None or _df.empty:
        return None, None
    X = _df[_feature_cols].copy()
    X_sc = _scaler.transform(X)
    
    # 1. SHAP Explainer (using K-Means for speed)
    background = shap.kmeans(X_sc, 10)
    shap_explainer = shap.KernelExplainer(_model.predict_proba, background)
    
    # 2. LIME Explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_sc,
        feature_names=_feature_cols,
        class_names=["Poor Quality", "Good Quality", "Average Quality", "Excellent Quality"],
        mode='classification',
        random_state=42
    )
    return shap_explainer, lime_explainer

model, scaler, feature_cols, idx_to_label = load_model()
df = load_data()
shap_explainer, lime_explainer = load_explainers(model, scaler, df, feature_cols)

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🍼 BabyFoodAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Nutrition Safety for Infants</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", ["🔍 Analyze Food", "📈 Growth Tracker", "🍽️ Meal Guide", "ℹ️ About"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**🧠 Model:** Voting Classifier")
    st.markdown("RF · XGBoost · GBM")
    if model:
        st.success(f"✅ Model ready — {len(df)} foods")
    else:
        st.error("⚠️ Run train_model.py first")

# ── Helpers ──────────────────────────────────────────────────────────────
def gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 13, "family": "Nunito", "color": "#555"}},
        number={"suffix": "/100", "font": {"size": 22, "family": "Nunito", "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#ddd",
                     "tickfont": {"size": 9}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#f8f8f8",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 45],  "color": "#FFE9E9"},
                {"range": [45, 70], "color": "#FFF8E6"},
                {"range": [70, 100],"color": "#E8FBF4"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.75, "value": value},
        }
    ))
    fig.update_layout(height=180, margin=dict(l=20, r=20, t=40, b=10),
                      paper_bgcolor="#fff", font=dict(family="Nunito"))
    return fig

def radar(nutrition):
    cats  = ["Protein","Fiber","Sugars","Sodium","Fats","Calories"]
    vals  = [nutrition.get("protein_g",0), nutrition.get("fiber_g",0),
             nutrition.get("sugar_g",0),   nutrition.get("sod_mg",0)/10,
             nutrition.get("fats_g",0),    nutrition.get("calories_kcal",0)/10]
    maxv  = [10, 5, 15, 30, 20, 20]
    norm  = [min(v/m, 1.0) for v,m in zip(vals, maxv)] + [min(vals[0]/maxv[0],1.0)]
    theta = cats + [cats[0]]
    fig = go.Figure(go.Scatterpolar(
        r=norm, theta=theta, fill="toself",
        fillcolor="rgba(255,140,105,0.18)",
        line=dict(color="#FF8C69", width=2.5),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1],
                   tickfont=dict(size=8), gridcolor="#F0F0F0"),
                   angularaxis=dict(tickfont=dict(size=11, family="Nunito"),
                                    gridcolor="#F0F0F0"),
                   bgcolor="#FFFAF6"),
        showlegend=False, paper_bgcolor="#fff",
        margin=dict(l=30,r=30,t=20,b=20), height=290,
        font=dict(family="Nunito"),
    )
    return fig

def progress_bar(label, val, color, icon=""):
    st.markdown(f"""
    <div class="pb-wrap">
      <div class="pb-label"><span>{icon} {label}</span><span style="color:{color};">{val:.0f} / 100</span></div>
      <div class="pb-track"><div class="pb-fill" style="width:{val}%;background:{color};"></div></div>
    </div>""", unsafe_allow_html=True)

def pill(text, level):
    cls = {"good":"pill-good","warn":"pill-warn","bad":"pill-bad"}[level]
    return f'<span class="pill {cls}">{text}</span>'

def nutrient_pills(n):
    items = []
    items.append((f"🔥 {n.get('calories_kcal',0):.0f} kcal",
                  "good" if 50<=n.get('calories_kcal',0)<=150 else "warn"))
    items.append((f"💪 Protein {n.get('protein_g',0):.1f}g",
                  "good" if 1.5<=n.get('protein_g',0)<=4 else "warn"))
    items.append((f"🍬 Sugar {n.get('sugar_g',0):.1f}g",
                  "good" if n.get('sugar_g',0)<=5 else "bad" if n.get('sugar_g',0)>10 else "warn"))
    items.append((f"🧂 Sodium {n.get('sod_mg',0):.0f}mg",
                  "good" if n.get('sod_mg',0)<=100 else "bad" if n.get('sod_mg',0)>200 else "warn"))
    items.append((f"🧈 Fat {n.get('fats_g',0):.1f}g",
                  "good" if n.get('fats_g',0)<=10 else "warn"))
    items.append((f"🌾 Fiber {n.get('fiber_g',0):.1f}g",
                  "good" if 0.5<=n.get('fiber_g',0)<=3 else "warn"))
    items.append((f"🩺 Chol. {n.get('cholesterol_mg',0):.0f}mg",
                  "good" if n.get('cholesterol_mg',0)<=50 else "bad"))
    return '<div class="pill-row">' + "".join(pill(t,l) for t,l in items) + "</div>"

# ════════════════════════════════════════════════════════════════════════
# PAGE 1 — ANALYZE
# ════════════════════════════════════════════════════════════════════════
if page == "🔍 Analyze Food":
    st.markdown("## 🍼 Baby Food Quality Analyzer")
    st.markdown("##### AI-powered nutrition safety check for your little one")
    st.markdown("---")

    if model is None:
        st.error("⚠️ Please run `python train_model.py` first to generate the model files.")
        st.stop()

    method = st.radio("Input Method", ["📝 Enter Manually", "🔎 Pick from Dataset"], horizontal=True)
    nutrition = {c: 0.0 for c in feature_cols}

    if method == "🔎 Pick from Dataset":
        opts = [f"Food #{i+1}" for i in range(len(df))]
        sel  = st.selectbox("Select a food item", opts)
        row  = df.iloc[opts.index(sel)]
        for col in feature_cols:
            if col in row:
                nutrition[col] = float(row[col])
        st.info("ℹ️ Values auto-filled — adjust if needed.")

    with st.expander("🧪 Nutritional Values (per 100g)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            nutrition["calories_kcal"]  = st.number_input("🔥 Calories (kcal)", 0.0, 900.0, float(nutrition.get("calories_kcal",80.0)), step=1.0)
            nutrition["protein_g"]      = st.number_input("💪 Protein (g)",      0.0, 50.0,  float(nutrition.get("protein_g",2.0)),       step=0.1)
            nutrition["fats_g"]         = st.number_input("🧈 Total Fat (g)",    0.0, 80.0,  float(nutrition.get("fats_g",2.0)),          step=0.1)
            nutrition["carb_g"]         = st.number_input("🍞 Carbs (g)",        0.0,100.0,  float(nutrition.get("carb_g",10.0)),         step=0.5)
        with c2:
            nutrition["fiber_g"]        = st.number_input("🌾 Fiber (g)",        0.0, 20.0,  float(nutrition.get("fiber_g",1.0)),         step=0.1)
            nutrition["sugar_g"]        = st.number_input("🍬 Sugars (g)",       0.0, 50.0,  float(nutrition.get("sugar_g",3.0)),         step=0.1)
            nutrition["sod_mg"]         = st.number_input("🧂 Sodium (mg)",      0.0,2000.0, float(nutrition.get("sod_mg",50.0)),         step=1.0)
            nutrition["cholesterol_mg"] = st.number_input("🩺 Cholesterol (mg)", 0.0, 300.0, float(nutrition.get("cholesterol_mg",5.0)),  step=1.0)
        with c3:
            nutrition["calcium_mg"]     = st.number_input("🦴 Calcium (mg)",     0.0,1000.0, float(nutrition.get("calcium_mg",20.0)),     step=1.0)
            nutrition["vitC_mg"]        = st.number_input("🍊 Vitamin C (mg)",   0.0, 200.0, float(nutrition.get("vitC_mg",5.0)),         step=0.5)
            nutrition["vitA_g"]         = st.number_input("🥕 Vitamin A (g)",    0.0,   5.0, float(nutrition.get("vitA_g",0.1)),          step=0.01)
            nutrition["potassium_mg"]   = st.number_input("⚡ Potassium (mg)",   0.0,2000.0, float(nutrition.get("potassium_mg",100.0)),  step=1.0)

    col_btn = st.columns([1,2,1])[1]
    with col_btn:
        analyze = st.button("🔍 Analyze Now", use_container_width=True)

    if analyze:
        # ── Validate inputs before analysis ───────────────────────────
        key_nutrients = ["calories_kcal", "protein_g", "fats_g", "carb_g",
                         "fiber_g", "sugar_g", "sod_mg"]
        total_input = sum(abs(nutrition.get(k, 0.0)) for k in key_nutrients)

        if nutrition.get("calories_kcal", 0.0) == 0 or total_input < 0.5:
            st.error("""
            ⚠️ **Cannot Analyze — All values are zero!**

            Please enter the actual nutritional values of the food before clicking Analyze.
            A score calculated on zero inputs is meaningless and misleading.

            👉 Enter at least: **Calories**, **Protein**, and one other nutrient.
            """)
            st.stop()

        vec     = [nutrition.get(c, 0.0) for c in feature_cols]
        X_sc    = scaler.transform(np.array([vec]))
        enc     = model.predict(X_sc)[0]
        proba   = model.predict_proba(X_sc)[0]
        orig_lbl= idx_to_label[enc]
        lbl_txt = LABEL_MAP[orig_lbl]
        lbl_clr = LABEL_COLORS[orig_lbl]
        result  = compute_recommendation_score(proba.tolist(), nutrition)
        conf    = result["model_quality_score"]

        st.markdown("---")
        st.markdown("### 📋 Analysis Results")

        # ── TOP ROW: Quality badge + Final score gauge + Safety gauge ──
        t1, t2, t3 = st.columns([1.2, 1, 1])

        with t1:
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:28px 20px;">
              <div style="font-size:.78rem;color:#999;font-weight:700;
                          letter-spacing:1px;text-transform:uppercase;margin-bottom:12px;">
                AI Prediction
              </div>
              <div style="font-size:1.7rem;font-weight:900;
                          color:{lbl_clr};margin-bottom:10px;">{lbl_txt}</div>
              <div style="background:{lbl_clr}18;border:2px solid {lbl_clr}40;
                          border-radius:30px;display:inline-block;
                          padding:5px 18px;font-size:.85rem;font-weight:700;color:{lbl_clr};">
                Confidence: {conf:.0f}%
              </div>
              <div style="margin-top:16px;font-size:.8rem;color:#aaa;">
                Ensemble: RF · XGBoost · GBM
              </div>
            </div>""", unsafe_allow_html=True)

        with t2:
            fc = result["final_score"]
            fc_color = "#06D6A0" if fc>=70 else "#F0A500" if fc>=45 else "#E74C3C"
            st.plotly_chart(gauge(fc, "Final Score", fc_color), use_container_width=True, config={"displayModeBar":False})

        with t3:
            sc = result["child_safety_score"]
            sc_color = "#06D6A0" if sc>=70 else "#F0A500" if sc>=45 else "#E74C3C"
            st.plotly_chart(gauge(sc, "Child Safety", sc_color), use_container_width=True, config={"displayModeBar":False})

        # ── NUTRIENT STATUS PILLS ──────────────────────────────────────
        st.markdown(f"""
        <div class="card">
          <div style="font-size:.82rem;font-weight:700;color:#999;
                      letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;">
            Nutrient Status
          </div>
          {nutrient_pills(nutrition)}
        </div>""", unsafe_allow_html=True)

        # ── MIDDLE ROW: Score breakdown + Radar ───────────────────────
        left, right = st.columns([1, 1])

        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**📊 Score Breakdown**")
            progress_bar("Model Confidence",    result["model_quality_score"],    "#FF8C69", "🤖")
            progress_bar("Nutrition Balance",   result["nutrition_balance_score"], "#7EC8A4", "🥗")
            progress_bar("Child Safety",        result["child_safety_score"],      "#74B9FF", "🛡️")
            st.markdown("---")
            st.markdown("**💡 Insights**")
            for tip in result["tips"]:
                st.markdown(f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**🕸️ Nutritional Radar**")
            st.plotly_chart(radar(nutrition), use_container_width=True,
                            config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # ── RECOMMENDATION BANNER ─────────────────────────────────────
        st.markdown(f"""
        <div class="rec-banner"
             style="background:{result['rec_color']};border-color:{result['rec_border']};">
          {result['recommendation']}
          <div class="rec-sub">{result['rec_message']}</div>
        </div>""", unsafe_allow_html=True)

        # ── PROBA BAR CHART ───────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        classes = [LABEL_MAP[idx_to_label[i]] for i in range(len(proba))]
        colors  = [LABEL_COLORS[idx_to_label[i]] for i in range(len(proba))]
        fig_bar = go.Figure(go.Bar(
            x=classes, y=[p*100 for p in proba],
            marker_color=colors,
            text=[f"{p*100:.1f}%" for p in proba],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="Model Probability per Quality Class",
            yaxis=dict(range=[0,115], title="Probability (%)", gridcolor="#F0F0F0"),
            xaxis_title="Quality Class",
            paper_bgcolor="#fff", plot_bgcolor="#fff",
            font=dict(family="Nunito", size=12),
            margin=dict(l=20,r=20,t=50,b=20),
            height=280,
        )
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar":False})
            st.markdown("</div>", unsafe_allow_html=True)

        # ── SHAP EXPLANATION ───────────────────────────────────────────
        if shap_explainer is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"**🧠 AI Decision Breakdown (SHAP)**")
            st.markdown("<span style='font-size:0.85rem;color:#777;'>Shows which nutrients pushed the AI towards or away from its prediction.</span>", unsafe_allow_html=True)
            
            with st.spinner("Calculating SHAP values..."):
                try:
                    shap_vals = shap_explainer.shap_values(X_sc)
                    # KernelExplainer on predict_proba returns a list of shap values per class.
                    # We want the explanation for the predicted class `enc`
                    if isinstance(shap_vals, list):
                        sv = shap_vals[enc]
                    else:
                        sv = shap_vals
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    # Fallback to summary_plot as a simple bar chart
                    shap.summary_plot(sv, X_sc, feature_names=feature_cols, plot_type="bar", show=False)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not generate SHAP plot: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
# PAGE 2 — GROWTH TRACKER
# ════════════════════════════════════════════════════════════════════════
elif page == "📈 Growth Tracker":
    render_growth_tracker(df)

# ════════════════════════════════════════════════════════════════════════
# PAGE 3 — MEAL GUIDE
# ════════════════════════════════════════════════════════════════════════
elif page == "🍽️ Meal Guide":
    st.markdown("## 🍽️ Baby Meal Guide")
    st.markdown("##### Curated baby-safe meals with AI nutrition scoring & ingredient breakdown")
    st.markdown("---")

    if model is None:
        st.error("⚠️ Please run `python train_model.py` first.")
        st.stop()

    # ── Nutrient element glossary ────────────────────────────────────────
    NUTRIENTS_INFO = {
        "protein_g":      {"label":"Protein",      "icon":"💪", "why":"Essential for muscle, organ & brain cell building. Babies need 1.5–4g/100g."},
        "fats_g":         {"label":"Total Fat",    "icon":"🧈", "why":"Provides energy & supports brain development. Healthy fats are crucial under 2 years."},
        "carb_g":         {"label":"Carbohydrates","icon":"🍞", "why":"Primary energy source. Complex carbs (oats, rice) are preferred over simple sugars."},
        "fiber_g":        {"label":"Fiber",        "icon":"🌾", "why":"Supports gut health & regular digestion. Keep under 3g to avoid bloating in infants."},
        "sugar_g":        {"label":"Sugars",       "icon":"🍬", "why":"Natural sugars (fruit) are fine in moderation. Added sugar is not safe for babies."},
        "sod_mg":         {"label":"Sodium",       "icon":"🧂", "why":"Babies' kidneys can't process excess salt. Keep under 100mg per serving."},
        "calcium_mg":     {"label":"Calcium",      "icon":"🦴", "why":"Critical for bone & teeth development. Dairy, broccoli & fortified foods are great sources."},
        "vitC_mg":        {"label":"Vitamin C",    "icon":"🍊", "why":"Boosts immune system & helps iron absorption. Found in fruits & vegetables."},
        "vitA_g":         {"label":"Vitamin A",    "icon":"🥕", "why":"Supports vision, immune health & skin. Found in orange/yellow vegetables."},
        "cholesterol_mg": {"label":"Cholesterol",  "icon":"🩺", "why":"Moderate amounts support brain development. Keep under 50mg per serving."},
        "potassium_mg":   {"label":"Potassium",    "icon":"⚡", "why":"Regulates heart rhythm & fluid balance. Bananas, potatoes & lentils are great sources."},
    }

    # ── Category filter ─────────────────────────────────────────────────
    all_cats = ["All"] + sorted({m["category"] for m in BABY_MEALS})
    cat_filter = st.selectbox("🏷️ Filter by Category", all_cats)
    filtered = BABY_MEALS if cat_filter == "All" else [m for m in BABY_MEALS if m["category"] == cat_filter]

    st.markdown(f"**{len(filtered)} meal(s) found**")
    st.markdown("---")

    for meal in filtered:
        n = meal["nutrition"]

        # ── Compute AI score ─────────────────────────────────────────────
        vec   = [n.get(c, 0.0) for c in feature_cols]
        X_sc  = scaler.transform(np.array([vec]))
        enc   = model.predict(X_sc)[0]
        proba = model.predict_proba(X_sc)[0]
        orig_lbl = idx_to_label[enc]
        lbl_txt  = LABEL_MAP[orig_lbl]
        lbl_clr  = LABEL_COLORS[orig_lbl]
        result   = compute_recommendation_score(proba.tolist(), n)
        fc       = result["final_score"]
        fc_color = "#06D6A0" if fc >= 70 else "#F0A500" if fc >= 45 else "#E74C3C"

        # ── Meal header ──────────────────────────────────────────────────
        with st.expander(
            f"{meal['emoji']}  {meal['name']}   ·   {meal['age']}   ·   Score: {fc:.0f}/100",
            expanded=False
        ):
            # Header row
            hc1, hc2 = st.columns([1.8, 1])
            with hc1:
                st.markdown(f"""
                <div style="background:{meal['bg']};border-radius:16px;padding:20px 24px;
                            border-left:5px solid {meal['color']};">
                  <div style="font-size:1.5rem;font-weight:900;color:#3D3D3D;margin-bottom:6px;">
                    {meal['emoji']} {meal['name']}
                  </div>
                  <div style="font-size:.82rem;font-weight:700;color:{meal['color']};
                              margin-bottom:10px;letter-spacing:.5px;">
                    🎂 {meal['age']} &nbsp;|&nbsp; 🏷️ {meal['category']}
                  </div>
                  <div style="font-size:.92rem;color:#555;line-height:1.7;">
                    {meal['description']}
                  </div>
                </div>""", unsafe_allow_html=True)

            with hc2:
                st.markdown(f"""
                <div class="card" style="text-align:center;padding:20px 16px;">
                  <div style="font-size:.75rem;color:#999;font-weight:700;
                              letter-spacing:1px;text-transform:uppercase;">AI Score</div>
                  <div style="font-size:2.8rem;font-weight:900;color:{fc_color};
                              line-height:1.1;">{fc:.0f}</div>
                  <div style="font-size:.78rem;color:#bbb;margin-bottom:10px;">/100</div>
                  <div style="background:{result['rec_border']}18;border:2px solid {result['rec_border']}40;
                              border-radius:20px;padding:4px 12px;
                              font-size:.8rem;font-weight:700;color:{result['rec_border']};">
                    {result['recommendation'].split(' ')[0]} {result['recommendation'].split(' ')[1]}
                  </div>
                  <div style="margin-top:10px;font-size:.75rem;color:#bbb;">
                    ML Class: {lbl_txt}
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── 3 columns: Ingredients | Scores | Nutrient pills ─────────
            c1, c2, c3 = st.columns([1.1, 1, 1])

            with c1:
                st.markdown("**🥘 Ingredients & Their Role**")
                for ing in meal["ingredients"]:
                    st.markdown(f"""
                    <div style="background:#FAFAFA;border-radius:12px;padding:12px 14px;
                                margin-bottom:8px;border-left:3px solid {meal['color']};">
                      <div style="font-size:.95rem;font-weight:800;">{ing['icon']} {ing['name']}</div>
                      <div style="font-size:.78rem;color:#999;margin:2px 0;">{ing['amount']}</div>
                      <div style="font-size:.83rem;color:#666;">{ing['role']}</div>
                    </div>""", unsafe_allow_html=True)

            with c2:
                st.markdown("**📊 Score Breakdown**")
                progress_bar("Model Confidence",  result["model_quality_score"],    "#FF8C69", "🤖")
                progress_bar("Nutrition Balance",  result["nutrition_balance_score"],"#7EC8A4", "🥗")
                progress_bar("Child Safety",       result["child_safety_score"],     "#74B9FF", "🛡️")
                st.markdown("---")
                st.markdown("**✅ Benefits**")
                for b in meal["benefits"]:
                    st.markdown(f"<div style='font-size:.84rem;margin-bottom:5px;'>{b}</div>", unsafe_allow_html=True)
                if meal["watch"]:
                    st.markdown("**⚠️ Watch Out**")
                    for w in meal["watch"]:
                        st.markdown(f"<div style='font-size:.84rem;color:#E67E22;margin-bottom:5px;'>{w}</div>", unsafe_allow_html=True)

            with c3:
                st.markdown("**🔬 Nutritional Elements**")
                for key, info in NUTRIENTS_INFO.items():
                    val = n.get(key, 0.0)
                    if val == 0:
                        continue
                    unit = "mg" if "mg" in key else "g" if "_g" in key else ""
                    st.markdown(f"""
                    <div style="margin-bottom:8px;padding:10px 12px;
                                background:#F8F8FF;border-radius:10px;">
                      <div style="font-size:.85rem;font-weight:800;margin-bottom:3px;">
                        {info['icon']} {info['label']}
                        <span style="float:right;color:#FF8C69;font-weight:900;">
                          {val:.1f}{unit}
                        </span>
                      </div>
                      <div style="font-size:.75rem;color:#888;line-height:1.5;">
                        {info['why']}
                      </div>
                    </div>""", unsafe_allow_html=True)

            # ── Radar ─────────────────────────────────────────────────────
            st.markdown("**🕸️ Nutritional Radar**")
            st.plotly_chart(radar(n), use_container_width=True,
                            config={"displayModeBar": False})

            # ── Recommendation banner ──────────────────────────────────────
            st.markdown(f"""
            <div class="rec-banner"
                 style="background:{result['rec_color']};border-color:{result['rec_border']};">
              {result['recommendation']}
              <div class="rec-sub">{result['rec_message']}</div>
            </div>""", unsafe_allow_html=True)
            
            # ── LIME Explanation ──────────────────────────────────────────
            if lime_explainer is not None:
                st.markdown("<br>**🤖 Why AI likes this meal (LIME Analysis):**", unsafe_allow_html=True)
                with st.spinner("Generating explanation..."):
                    try:
                        # explain_instance expects 1D array
                        exp = lime_explainer.explain_instance(X_sc[0], model.predict_proba, num_features=3, top_labels=1)
                        # Get explanations for the predicted class
                        lime_list = exp.as_list(label=exp.available_labels()[0])
                        for feature_desc, weight in lime_list:
                            icon = "✅" if weight > 0 else "⚠️"
                            color = "#27ae60" if weight > 0 else "#e74c3c"
                            st.markdown(f"<div style='font-size:.9rem; margin-bottom:4px;'><span style='color:{color}'>{icon}</span> <b>{feature_desc}</b> (Impact: {weight:.2f})</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.write("Could not generate LIME explanation.")

        st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("## ℹ️ About BabyFoodAI")
    st.markdown("---")
    st.markdown("""<div class="card">
    <h3>🍼 What does this app do?</h3>
    <p>An AI-powered tool that predicts nutritional quality of baby food items and provides
    a composite safety score for infants using an ensemble ML model.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🧠 Ensemble Model")
    st.markdown("""| Model | Config |
|-------|--------|
| Random Forest | 200 trees, max_depth=10 |
| XGBoost | 200 rounds, lr=0.05, depth=6 |
| Gradient Boosting | 200 rounds, lr=0.05, depth=5 |
| **Voting** | **Soft (probability avg)** |""")

    st.markdown("### 📐 Final Score Formula")
    st.markdown("""| Component | Weight |
|-----------|--------|
| 🤖 Model Confidence | 50% |
| 🥗 Nutrition Balance | 30% |
| 🛡️ Child Safety | 20% |""")

    st.markdown("### 🏷️ Quality Labels")
    for k, v in LABEL_MAP.items():
        c = LABEL_COLORS[k]
        st.markdown(f'<span class="pill pill-good" style="background:{c}18;color:{c};border-color:{c}30;">{v}</span>', unsafe_allow_html=True)
