import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPulse · Student Intelligence System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Inter:wght@300;400;500;600&display=swap');

/* ── root ── */
:root {
    --bg-deep:   #03040a;
    --bg-panel:  #080c1a;
    --bg-card:   #0d1226;
    --accent-1:  #00d4ff;
    --accent-2:  #7c3aed;
    --accent-3:  #10b981;
    --accent-warn:#f59e0b;
    --accent-red: #ef4444;
    --text-main: #e2e8f0;
    --text-dim:  #64748b;
    --border:    rgba(0,212,255,0.15);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-deep) !important;
    color: var(--text-main) !important;
}

/* ── hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0rem 2rem 2rem 2rem !important; max-width: 1600px !important; }

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06091a 0%, #080c1a 100%) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-main) !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stRadio label { font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent-1) !important; }

/* ── selectboxes & inputs ── */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-main) !important;
}

/* ── sliders ── */
.stSlider .rc-slider-track { background: var(--accent-1) !important; }
.stSlider .rc-slider-handle { border-color: var(--accent-1) !important; background: var(--accent-1) !important; }

/* ── metric cards ── */
div[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
div[data-testid="metric-container"] label { color: var(--text-dim) !important; font-size: 0.72rem !important; letter-spacing: 0.1em; text-transform: uppercase; }
div[data-testid="metric-container"] div[data-testid="metric-value"] { font-size: 1.6rem !important; font-weight: 600; color: var(--accent-1) !important; }

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px 8px 0 0;
    color: var(--text-dim) !important;
    font-size: 0.82rem;
    letter-spacing: 0.06em;
    padding: 0.6rem 1.4rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(0,212,255,0.1) 0%, transparent 100%) !important;
    border-bottom: 2px solid var(--accent-1) !important;
    color: var(--accent-1) !important;
}

/* ── divider ── */
hr { border-color: var(--border) !important; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--accent-2); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ─── Load Assets ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("Student_performance(KNN).pkl")

@st.cache_data
def load_data():
    return pd.read_csv("Student_academic.csv")

model = load_model()
df_raw = load_data()

CAT_COLS = ['gender','NationalITy','PlaceofBirth','StageID','GradeID','SectionID',
            'Topic','Semester','Relation','ParentAnsweringSurvey',
            'ParentschoolSatisfaction','StudentAbsenceDays']
NUM_COLS = ['raisedhands','VisITedResources','AnnouncementsView','Discussion']

# Class mapping discovered: model outputs {0->L, 1->H, 2->M}
CLASS_MAP  = {0: 'L', 1: 'H', 2: 'M'}
CLASS_LABEL= {'H': '🏆 High Performer', 'M': '📈 Mid-Level', 'L': '⚠️ Needs Support'}
CLASS_COLOR= {'H': '#10b981', 'M': '#f59e0b', 'L': '#ef4444'}
CLASS_DESC = {
    'H': 'This student shows strong academic signals. They are likely to achieve an <b>A or B grade</b>. Engagement levels are healthy.',
    'M': 'This student is performing <b>adequately</b> but has potential for improvement. Targeted interventions can push them higher.',
    'L': 'This student is at <b>risk of underperformance</b>. Immediate attention, counselling, and parental engagement are recommended.',
}

def encode_and_predict(row_dict):
    """Encode a single input and return (pred_class, proba_array)."""
    row = pd.DataFrame([row_dict])
    full = pd.concat([df_raw.drop(columns=['Class']), row], ignore_index=True)
    enc  = pd.get_dummies(full, columns=CAT_COLS)
    last = enc.iloc[[-1]]
    proba = model.predict_proba(last)[0]   # shape (3,)
    pred_int = model.predict(last)[0]
    pred_class = CLASS_MAP[pred_int]
    # proba order: 0->L, 1->H, 2->M  => map to H,M,L
    proba_dict = {CLASS_MAP[i]: proba[i] for i in range(3)}
    return pred_class, proba_dict

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
  background: linear-gradient(135deg, #06091a 0%, #0d1226 50%, #09122a 100%);
  border: 1px solid rgba(0,212,255,0.2);
  border-radius: 16px;
  padding: 2rem 2.5rem;
  margin-bottom: 1.8rem;
  position: relative;
  overflow: hidden;
">
  <div style="position:absolute;top:-40px;right:-40px;width:200px;height:200px;
              background:radial-gradient(circle,rgba(124,58,237,0.25) 0%,transparent 70%);border-radius:50%;"></div>
  <div style="position:absolute;bottom:-30px;left:20%;width:150px;height:150px;
              background:radial-gradient(circle,rgba(0,212,255,0.15) 0%,transparent 70%);border-radius:50%;"></div>
  <p style="font-family:'Orbitron',sans-serif;font-size:0.7rem;letter-spacing:0.3em;
             color:#7c3aed;margin:0 0 0.4rem 0;text-transform:uppercase;">
    AI-POWERED ACADEMIC INTELLIGENCE</p>
  <h1 style="font-family:'Orbitron',sans-serif;font-size:2.4rem;font-weight:800;
              background:linear-gradient(90deg,#00d4ff,#7c3aed);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              margin:0 0 0.5rem 0;letter-spacing:0.05em;">
    EduPulse
  </h1>
  <p style="color:#94a3b8;font-size:0.95rem;margin:0;max-width:600px;font-weight:300;">
    Student Performance Prediction System &nbsp;·&nbsp; KNN Classifier
    &nbsp;·&nbsp; <span style="color:#00d4ff;">480 training records</span>
  </p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar Inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 1.5rem;">
      <p style="font-family:'Orbitron',sans-serif;font-size:1rem;color:#00d4ff;
                letter-spacing:0.15em;margin:0;">STUDENT PROFILE</p>
      <p style="font-size:0.72rem;color:#475569;margin:0.3rem 0 0;">Configure parameters below</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 👤 Demographics")
    gender     = st.selectbox("Gender",          ['M','F'])
    nationality= st.selectbox("Nationality",     sorted(df_raw['NationalITy'].unique()))
    place      = st.selectbox("Place of Birth",  sorted(df_raw['PlaceofBirth'].unique()))

    st.markdown("---")
    st.markdown("##### 🏫 Academic Profile")
    stage   = st.selectbox("School Stage",  ['HighSchool','MiddleSchool','lowerlevel'])
    grade   = st.selectbox("Grade ID",      sorted(df_raw['GradeID'].unique()))
    section = st.selectbox("Section",       ['A','B','C'])
    topic   = st.selectbox("Subject Topic", sorted(df_raw['Topic'].unique()))
    semester= st.radio("Semester", ['F','S'], horizontal=True)

    st.markdown("---")
    st.markdown("##### 📊 Engagement Metrics")
    raised   = st.slider("Raised Hands",         0, 100, 45)
    visited  = st.slider("Visited Resources",    0, 100, 55)
    announce = st.slider("Announcements Viewed", 0, 100, 38)
    discuss  = st.slider("Discussion Participation", 0, 100, 43)

    st.markdown("---")
    st.markdown("##### 👨‍👩‍👧 Parent & Attendance")
    relation   = st.radio("Responsible Relation",    ['Father','Mum'], horizontal=True)
    parent_ans = st.radio("Parent Answered Survey",  ['Yes','No'],     horizontal=True)
    parent_sat = st.radio("Parent School Satisfaction", ['Good','Bad'],horizontal=True)
    absence    = st.radio("Student Absence Days",    ['Under-7','Above-7'], horizontal=True)

    predict_btn = st.button("⚡  PREDICT PERFORMANCE", use_container_width=True)

# ─── Build Input Dict ────────────────────────────────────────────────────────
input_dict = {
    'gender': gender, 'NationalITy': nationality, 'PlaceofBirth': place,
    'StageID': stage, 'GradeID': grade, 'SectionID': section,
    'Topic': topic, 'Semester': semester, 'Relation': relation,
    'raisedhands': raised, 'VisITedResources': visited,
    'AnnouncementsView': announce, 'Discussion': discuss,
    'ParentAnsweringSurvey': parent_ans,
    'ParentschoolSatisfaction': parent_sat,
    'StudentAbsenceDays': absence,
}

# Auto-predict on load / sidebar change
pred_class, proba_dict = encode_and_predict(input_dict)

# ─── Main Tabs ───────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯  Prediction & Analysis", "📊  Dataset Explorer", "🔬  What-If Simulator"])

# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Prediction
# ════════════════════════════════════════════════════════════════════════════
with tab1:

    col_pred, col_gauge = st.columns([1, 1], gap="large")

    # ── Prediction Card ──────────────────────────────────────────────────────
    with col_pred:
        clr  = CLASS_COLOR[pred_class]
        lbl  = CLASS_LABEL[pred_class]
        desc = CLASS_DESC[pred_class]
        conf = proba_dict[pred_class]

        st.markdown(f"""
        <div style="
          background: linear-gradient(135deg,{clr}18,{clr}08);
          border: 1px solid {clr}55;
          border-radius: 16px;
          padding: 2rem;
          text-align: center;
          margin-bottom:1.2rem;
        ">
          <p style="font-family:'Orbitron',sans-serif;font-size:0.65rem;
                     letter-spacing:0.35em;color:{clr};margin:0 0 0.6rem;">
            PREDICTED CLASS</p>
          <div style="font-family:'Orbitron',sans-serif;font-size:5rem;
                      font-weight:800;color:{clr};line-height:1;margin-bottom:0.3rem;">
            {pred_class}
          </div>
          <p style="font-size:1.1rem;color:#e2e8f0;font-weight:500;margin:0 0 0.8rem;">
            {lbl}</p>
          <div style="background:{clr}22;border-radius:8px;padding:0.6rem 1rem;
                      display:inline-block;margin-bottom:1rem;">
            <span style="font-family:'Orbitron',sans-serif;font-size:1.5rem;
                          font-weight:700;color:{clr};">{conf*100:.1f}%</span>
            <span style="color:#94a3b8;font-size:0.8rem;margin-left:0.4rem;">confidence</span>
          </div>
          <p style="color:#94a3b8;font-size:0.88rem;line-height:1.6;margin:0;">
            {desc}</p>
        </div>
        """, unsafe_allow_html=True)

        # Mini probability bars
        st.markdown("**Class Probability Distribution**")
        for cls in ['H','M','L']:
            p   = proba_dict[cls]
            bar_clr = CLASS_COLOR[cls]
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.5rem;">
              <span style="font-family:'Orbitron',sans-serif;font-size:0.9rem;
                            color:{bar_clr};width:24px;text-align:center;font-weight:700;">{cls}</span>
              <div style="flex:1;background:#0d1226;border-radius:99px;height:10px;overflow:hidden;
                          border:1px solid {bar_clr}33;">
                <div style="width:{p*100:.1f}%;height:100%;
                             background:linear-gradient(90deg,{bar_clr},{bar_clr}99);
                             border-radius:99px;transition:width 0.4s;"></div>
              </div>
              <span style="font-size:0.8rem;color:#94a3b8;width:42px;text-align:right;">
                {p*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Gauge ────────────────────────────────────────────────────────────────
    with col_gauge:
        # Overall score heuristic: weighted average of engagement metrics
        engage_score = (raised*0.3 + visited*0.3 + announce*0.2 + discuss*0.2)
        bonus = 5 if parent_ans=='Yes' else 0
        bonus += 5 if parent_sat=='Good' else 0
        bonus -= 10 if absence=='Above-7' else 0
        final_score = min(100, max(0, engage_score + bonus))

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=final_score,
            delta={'reference': 50, 'valueformat':'.1f',
                   'increasing':{'color':'#10b981'}, 'decreasing':{'color':'#ef4444'}},
            number={'suffix':'', 'font':{'size':48,'color':'#e2e8f0','family':'Orbitron'}},
            gauge={
                'axis':{'range':[0,100],'tickcolor':'#334155','tickfont':{'color':'#64748b','size':11}},
                'bar':{'color': CLASS_COLOR[pred_class],'thickness':0.28},
                'bgcolor':'#080c1a',
                'borderwidth':0,
                'steps':[
                {'range':[0,40],  'color':'rgba(239, 68, 68, 0.13)'},
                {'range':[40,70], 'color':'rgba(245, 158, 11, 0.13)'},
                {'range':[70,100],'color':'rgba(16, 185, 129, 0.13)'},
            ],
                'threshold':{
                    'line':{'color':'#00d4ff','width':3},
                    'thickness':0.85,
                    'value': final_score,
                }
            },
            title={'text':"Engagement Score",'font':{'size':14,'color':'#64748b'}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
            margin=dict(t=40,b=20,l=30,r=30),
            font_color='#e2e8f0',
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar':False})

        # Feature contribution radar
        features = ['Raised Hands','Resources','Announcements','Discussion']
        vals     = [raised, visited, announce, discuss]
        fig_radar = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=features + [features[0]],
            fill='toself',
            fillcolor='rgba(0,212,255,0.12)',
            line=dict(color='#00d4ff', width=2),
            marker=dict(color='#00d4ff', size=6),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0,100],
                                gridcolor='rgba(255,255,255,0.07)',
                                tickfont=dict(color='#475569', size=10),
                                color='#475569'),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.07)',
                                 linecolor='rgba(255,255,255,0.07)',
                                 tickfont=dict(color='#94a3b8', size=11)),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            height=260,
            margin=dict(t=30,b=20,l=40,r=40),
            title=dict(text='Engagement Radar',font=dict(color='#64748b',size=13)),
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar':False})

    # ── KPIs row ─────────────────────────────────────────────────────────────
    st.markdown("---")
    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: st.metric("Raised Hands",        f"{raised}/100",  f"{raised-47:+d} vs avg")
    with k2: st.metric("Resources Visited",   f"{visited}/100", f"{visited-55:+d} vs avg")
    with k3: st.metric("Announcements",       f"{announce}/100",f"{announce-38:+d} vs avg")
    with k4: st.metric("Discussion",          f"{discuss}/100", f"{discuss-43:+d} vs avg")
    with k5:
        risk = "🟢 Low" if pred_class=='H' else ("🟡 Medium" if pred_class=='M' else "🔴 High")
        st.metric("Risk Level", risk)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Dataset Explorer
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Dataset Overview")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total Students", len(df_raw))
    m2.metric("High Performers (H)", int((df_raw['Class']=='H').sum()))
    m3.metric("Mid-Level (M)",       int((df_raw['Class']=='M').sum()))
    m4.metric("Need Support (L)",    int((df_raw['Class']=='L').sum()))

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        # Class distribution donut
        vc = df_raw['Class'].value_counts()
        fig_donut = go.Figure(go.Pie(
            labels=[CLASS_LABEL[c] for c in vc.index],
            values=vc.values,
            hole=0.6,
            marker=dict(colors=[CLASS_COLOR[c] for c in vc.index],
                        line=dict(color='#03040a', width=3)),
            textfont=dict(color='#e2e8f0', size=12),
            hovertemplate='%{label}<br>Count: %{value}<br>Share: %{percent}<extra></extra>',
        ))
        fig_donut.add_annotation(text="Class<br>Split", x=0.5, y=0.5,
                                  showarrow=False, font=dict(color='#94a3b8', size=13))
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True, height=320,
            legend=dict(font=dict(color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
            margin=dict(t=20,b=10,l=10,r=10),
            title=dict(text='Performance Class Distribution',font=dict(color='#64748b',size=14)),
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar':False})

        # Gender breakdown
        fig_gen = px.histogram(df_raw, x='gender', color='Class',
                               color_discrete_map=CLASS_COLOR, barmode='group',
                               title='Performance by Gender',
                               labels={'gender':'Gender','count':'Students'})
        fig_gen.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'), height=300,
            title=dict(font=dict(color='#64748b',size=14)),
            legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#94a3b8')),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            margin=dict(t=40,b=20,l=10,r=10),
        )
        st.plotly_chart(fig_gen, use_container_width=True, config={'displayModeBar':False})

    with col_b:
        # Engagement box plots
        fig_box = go.Figure()
        colors_box = ['#00d4ff','#7c3aed','#10b981','#f59e0b']
        for i, col in enumerate(NUM_COLS):
            for cls in ['H','M','L']:
                sub = df_raw[df_raw['Class']==cls][col]
                fig_box.add_trace(go.Box(
                    y=sub, name=f"{col[:6]}-{cls}",
                    marker_color=CLASS_COLOR[cls],
                    line_color=CLASS_COLOR[cls],
                    opacity=0.8,
                    visible=(i==0),
                ))
        fig_box.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'), height=320,
            title=dict(text='Engagement Distribution by Class',font=dict(color='#64748b',size=14)),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#94a3b8')),
            margin=dict(t=40,b=20,l=10,r=10),
        )
        st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar':False})

        # Topic performance heatmap
        topic_class = df_raw.groupby(['Topic','Class']).size().unstack(fill_value=0)
        fig_heat = px.imshow(topic_class,
                             color_continuous_scale='Blues',
                             aspect='auto',
                             title='Students per Topic × Class')
        fig_heat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'), height=300,
            title=dict(font=dict(color='#64748b',size=14)),
            coloraxis_colorbar=dict(tickfont=dict(color='#94a3b8')),
            margin=dict(t=40,b=20,l=10,r=10),
            xaxis=dict(tickfont=dict(color='#94a3b8')),
            yaxis=dict(tickfont=dict(color='#94a3b8',size=10)),
        )
        st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar':False})

    # Scatter: Raised Hands vs Resources
    st.markdown("---")
    fig_scatter = px.scatter(
        df_raw, x='raisedhands', y='VisITedResources',
        color='Class', color_discrete_map=CLASS_COLOR,
        size='Discussion', hover_data=['Topic','gender','StageID'],
        title='Raised Hands vs Resources Visited (bubble size = Discussion)',
        opacity=0.75,
    )
    fig_scatter.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'), height=380,
        title=dict(font=dict(color='#64748b',size=14)),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Raised Hands'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Resources Visited'),
        legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#94a3b8')),
        margin=dict(t=40,b=20,l=10,r=10),
    )
    st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar':False})

# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — What-If Simulator
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔬 What-If Sensitivity Analysis")
    st.markdown(
        "<p style='color:#64748b;font-size:0.88rem;'>Sweep a single parameter while holding all "
        "others fixed at sidebar values to see how the prediction probability changes.</p>",
        unsafe_allow_html=True,
    )

    sweep_col = st.selectbox("Select parameter to sweep:",
                              ['raisedhands','VisITedResources','AnnouncementsView','Discussion'])

    sweep_vals = list(range(0, 101, 5))
    results = {'H':[], 'M':[], 'L':[]}

    for v in sweep_vals:
        test = dict(input_dict)
        test[sweep_col] = v
        _, proba = encode_and_predict(test)
        for cls in ['H','M','L']:
            results[cls].append(proba[cls] * 100)

    fig_sweep = go.Figure()
    FILL_COLORS = {
    'H': 'rgba(16, 185, 129, 0.10)',
    'M': 'rgba(245, 158, 11, 0.10)',
    'L': 'rgba(239, 68, 68, 0.10)'
}

for cls in ['H','M','L']:
    fig_sweep.add_trace(go.Scatter(
        x=sweep_vals,
        y=results[cls],
        name=CLASS_LABEL[cls],
        line=dict(color=CLASS_COLOR[cls], width=2.5),
        fill='tozeroy',
        fillcolor=FILL_COLORS[cls],
        hovertemplate=f"{cls}: %{{y:.1f}}%<extra></extra>",
    ))

    # Mark current value
    cur_val = input_dict[sweep_col]
    fig_sweep.add_vline(x=cur_val, line_dash='dash', line_color='#00d4ff', line_width=1.5,
                         annotation_text=f"Current: {cur_val}",
                         annotation_font_color='#00d4ff', annotation_font_size=11)

    fig_sweep.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'), height=380,
        title=dict(text=f'Sensitivity: {sweep_col}', font=dict(color='#64748b',size=14)),
        xaxis=dict(title=sweep_col, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title='Probability (%)', range=[0,100], gridcolor='rgba(255,255,255,0.05)'),
        legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#94a3b8')),
        hovermode='x unified',
        margin=dict(t=40,b=30,l=10,r=10),
    )
    st.plotly_chart(fig_sweep, use_container_width=True, config={'displayModeBar':False})

    # Absence / parental toggle analysis
    st.markdown("---")
    st.markdown("#### 📋 Factor Impact Table")
    scenarios = [
        ("Absence: Under-7 + Parent: Yes + Satisfaction: Good", {'StudentAbsenceDays':'Under-7','ParentAnsweringSurvey':'Yes','ParentschoolSatisfaction':'Good'}),
        ("Absence: Above-7 + Parent: Yes + Satisfaction: Good", {'StudentAbsenceDays':'Above-7','ParentAnsweringSurvey':'Yes','ParentschoolSatisfaction':'Good'}),
        ("Absence: Under-7 + Parent: No  + Satisfaction: Bad",  {'StudentAbsenceDays':'Under-7','ParentAnsweringSurvey':'No', 'ParentschoolSatisfaction':'Bad'}),
        ("Absence: Above-7 + Parent: No  + Satisfaction: Bad",  {'StudentAbsenceDays':'Above-7','ParentAnsweringSurvey':'No', 'ParentschoolSatisfaction':'Bad'}),
    ]
    rows = []
    for label, overrides in scenarios:
        test = dict(input_dict)
        test.update(overrides)
        cls, prob = encode_and_predict(test)
        rows.append({
            'Scenario': label,
            'Predicted': cls,
            'H%': f"{prob['H']*100:.0f}%",
            'M%': f"{prob['M']*100:.0f}%",
            'L%': f"{prob['L']*100:.0f}%",
        })
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
    )

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 0.5rem;color:#1e293b;font-size:0.75rem;">
  EduPulse · KNN Classifier · Academic Intelligence System
</div>
""", unsafe_allow_html=True)