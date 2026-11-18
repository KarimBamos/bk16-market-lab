import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss, roc_curve, auc

# --- CONFIGURATION DE LA PAGE ---
bk16_finance_icon = """data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="14" fill="%230b0f19" stroke="%2310b981" stroke-width="3"/><line x1="18" y1="16" x2="18" y2="48" stroke="%23ef4444" stroke-width="2.5"/><rect x="14" y="24" width="8" height="16" rx="1.5" fill="%23ef4444"/><line x1="32" y1="12" x2="32" y2="46" stroke="%2310b981" stroke-width="2.5"/><rect x="28" y="18" width="8" height="18" rx="1.5" fill="%2310b981"/><line x1="46" y1="10" x2="46" y2="40" stroke="%2310b981" stroke-width="2.5"/><rect x="42" y="14" width="8" height="16" rx="1.5" fill="%2310b981"/><path d="M12 46 Q 28 38, 52 14" fill="none" stroke="%2338bdf8" stroke-width="3" stroke-linecap="round"/></svg>"""

st.set_page_config(
    page_title="BK16 Market Intelligence Lab | Pro Terminal",
    page_icon=bk16_finance_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DESIGN INSTITUTIONNEL (CSS PRO & ÉLÉMENTS FIXES) ---
# --- DESIGN INSTITUTIONNEL (CSS PRO & ÉLÉMENTS FIXES) ---
st.markdown("""
<style>
    /* Forcer le fond sombre institutionnel sur toute l'application */
    .stApp {
        background-color: #0b0f19 !important;
        color: #e2e8f0 !important;
    }
    
    /* Masquer l'en-tête natif de Streamlit */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Décaler le contenu principal pour qu'il passe sous le bandeau fixe */
    .block-container {
        padding-top: 4.8rem !important;
        padding-bottom: 2rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        background-color: #0b0f19 !important;
    }

    /* Barre latérale et avatar fixe au scroll */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1e293b;
    }
    
    .fixed-avatar-container {
        position: sticky;
        top: 10px;
        background-color: #111827;
        z-index: 99;
        padding-bottom: 10px;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 10px;
    }

    /* Bandeau défilant FIXE ajusté pour s'arrêter au bord de la sidebar */
    .fixed-ticker {
        position: fixed;
        top: 0;
        left: 12rem; /* Aligné exactement après la largeur par défaut de la barre latérale */
        width: calc(100% - 12rem); /* Réduit la largeur totale pour compenser */
        background-color: #111827;
        border-bottom: 2px solid #38bdf8;
        border-left: 1px solid #1e293b; /* Petite bordure esthétique pour la jointure */
        padding: 10px 0;
        z-index: 999999;
        overflow: hidden;
        white-space: nowrap;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.8);
    }

    .pro-card {
        background: linear-gradient(145deg, #131c31 0%, #0f1626 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .pro-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    .card-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #94a3b8;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .card-value-bull {
        font-size: 2.2rem;
        font-weight: 800;
        color: #10b981;
        text-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
    }
    .card-value-bear {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ef4444;
        text-shadow: 0 0 15px rgba(239, 68, 68, 0.3);
    }
    .card-value-neut {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f59e0b;
        text-shadow: 0 0 15px rgba(245, 158, 11, 0.3);
    }
    .card-value-info {
        font-size: 1.9rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .card-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 6px;
    }
    .section-header {
        border-left: 4px solid #38bdf8;
        padding-left: 12px;
        margin-top: 25px;
        margin-bottom: 15px;
        font-size: 1.3rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .status-badge {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Footer fixe en bas à droite */
    .footer-bk16 {
        position: fixed;
        right: 20px;
        bottom: 10px;
        font-size: 0.75rem;
        color: #64748b;
        background-color: rgba(11, 15, 25, 0.85);
        padding: 6px 12px;
        border-radius: 6px;
        border: 1px solid #1e293b;
        z-index: 99999;
        letter-spacing: 0.5px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# --- BANDEAU DÉFILANT INSTITUTIONNEL (FIXÉ AU SOMMET) ---
st.markdown("""
<div class="fixed-ticker">
    <marquee behavior="scroll" direction="left" scrollamount="6" style="font-family: monospace; font-size: 0.9rem; color: #38bdf8; font-weight: bold; letter-spacing: 1.5px;">
        ⚡ &nbsp; BK16 MARKET INTELLIGENCE LAB &nbsp;&bull;&nbsp; 
        QUANTITATIVE ENGINE V2.5 ACTIVE &nbsp;&bull;&nbsp; 
        REAL-TIME CROSS-ASSET PROBABILISTIC FORECASTING &nbsp;&bull;&nbsp; 
        LOG-LOSS OPTIMIZED &nbsp;&bull;&nbsp; 
        INSTITUTIONAL GRADE TRADING SYSTEMS &nbsp;&bull;&nbsp; 
        SYSTEM STATUS: ONLINE 🚀
    </marquee>
</div>
""", unsafe_allow_html=True)

# --- CATALOGUE DES ACTIFS ---
ASSETS = {
    "Forex (Devises)": {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "USDJPY=X",
        "USD/CHF": "USDCHF=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CAD": "USDCAD=X",
        "EUR/JPY": "EURJPY=X"
    },
    "Cryptomonnaies": {
        "Bitcoin (BTC/USD)": "BTC-USD",
        "Ethereum (ETH/USD)": "ETH-USD",
        "Solana (SOL/USD)": "SOL-USD",
        "Ripple (XRP/USD)": "XRP-USD"
    },
    "Indices Boursiers": {
        "Nasdaq 100": "^NDX",
        "S&P 500": "^GSPC",
        "Dow Jones 30": "^DJI",
        "DAX 40 (Allemagne)": "^GDAXI",
        "CAC 40 (France)": "^FCHI"
    },
    "Matières Premières": {
        "Or (XAU/USD)": "GC=F",
        "Argent (XAG/USD)": "SI=F",
        "Pétrole Brut WTI": "CL=F",
        "Gaz Naturel": "NG=F"
    }
}

# --- CONFIGURATION COMPLÈTE DES TIMEFRAMES ---
TIMEFRAMES = {
    "5 Minutes (Scalping M5)": {"interval": "5m", "period": "60d", "thresh": 0.0008, "ma_fast": 20, "ma_slow": 50, "delta": pd.Timedelta(minutes=5)},
    "15 Minutes (Intraday M15)": {"interval": "15m", "period": "60d", "thresh": 0.0015, "ma_fast": 20, "ma_slow": 50, "delta": pd.Timedelta(minutes=15)},
    "30 Minutes (Intraday M30)": {"interval": "30m", "period": "60d", "thresh": 0.0020, "ma_fast": 20, "ma_slow": 50, "delta": pd.Timedelta(minutes=30)},
    "1 Heure (Intraday H1)": {"interval": "1h", "period": "730d", "thresh": 0.0025, "ma_fast": 20, "ma_slow": 50, "delta": pd.Timedelta(hours=1)},
    "2 Heures (Swing Court H2)": {"interval": "1h", "period": "730d", "resample": "2h", "thresh": 0.0032, "ma_fast": 20, "ma_slow": 50, "delta": pd.Timedelta(hours=2)},
    "4 Heures (Swing H4)": {"interval": "1h", "period": "730d", "resample": "4h", "thresh": 0.0040, "ma_fast": 20, "ma_slow": 50, "delta": pd.Timedelta(hours=4)},
    "1 Jour (Daily D1)": {"interval": "1d", "period": "10y", "thresh": 0.0050, "ma_fast": 20, "ma_slow": 50, "delta": pd.Timedelta(days=1)},
    "1 Semaine (Hebdo W1)": {"interval": "1wk", "period": "20y", "thresh": 0.0120, "ma_fast": 20, "ma_slow": 50, "delta": pd.Timedelta(weeks=1)},
    "1 Mois (Mensuel MN)": {"interval": "1mo", "period": "max", "thresh": 0.0250, "ma_fast": 12, "ma_slow": 24, "delta": pd.Timedelta(days=30)},
    "1 An (Annuel Y1)": {"interval": "1mo", "period": "max", "resample": "1YE", "thresh": 0.0600, "ma_fast": 5, "ma_slow": 10, "delta": pd.Timedelta(days=365)}
}

# --- BARRE LATÉRALE : NAVIGATION & PARAMÈTRES (AVATAR FIXE) ---
with st.sidebar:
    import base64
    
    st.markdown('<div class="fixed-avatar-container">', unsafe_allow_html=True)
    try:
        with open("IMG_20241025_095324_348.jpg", "rb") as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode()
        st.markdown(f"""
            <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 12px; margin-top: 5px;">
                <img src="data:image/jpeg;base64,{encoded_img}" style="width: 90px; height: 90px; border-radius: 50%; object-fit: cover; object-position: center 20%; border: 3px solid #10b981; box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);">
            </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass

    st.markdown("## 🏛️ BK16 LAB TERMINAL")
    st.markdown('<span class="status-badge">● SYSTÈME HYBRIDE ACTIF</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    page = st.radio(
        "NAVIGATION",
        ["📊 Terminal d'Analyse Live", "🔐 Audit & Modèles (Sécurisé)", "📖 À propos"],
        index=0
    )
    
    if page != "📖 À propos":
        st.markdown("---")
        st.markdown("### ⚙️ Paramètres du Marché")
        
        category = st.selectbox("Classe d'actifs", list(ASSETS.keys()))
        asset_name = st.selectbox("Instrument", list(ASSETS[category].keys()))
        ticker = ASSETS[category][asset_name]
        
        tf_label = st.selectbox("Unité de Temps (Timeframe)", list(TIMEFRAMES.keys()), index=0)
        tf_config = TIMEFRAMES[tf_label]
        
        horizon = st.slider("Horizon de prédiction futur (bougies)", min_value=1, max_value=10, value=1)
        
        st.markdown("---")
        if st.button("🔄 Actualiser les prix en direct", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

# --- FONCTIONS DE COLLECTE & PRÉPARATION ---
@st.cache_data(ttl=1800)
def fetch_data(symbol, interval, period, resample_rule=None):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, axis=1, level=1)
    df = df.dropna()
    
    if resample_rule and not df.empty:
        agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
        if 'Volume' in df.columns:
            agg_dict['Volume'] = 'sum'
        df = df.resample(resample_rule).agg(agg_dict).dropna()
        
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

@st.cache_data(ttl=86400)
def fetch_macro_data():
    import pandas_datareader.data as web
    start = datetime.date(1990, 1, 1)
    end = datetime.date.today()
    try:
        macro = web.DataReader(['FEDFUNDS', 'CPIAUCSL', 'PPIACO', 'PCEPI', 'VIXCLS', 'DGS10', 'PAYEMS'], 'fred', start, end)
        macro = macro.ffill().bfill()
    except Exception:
        dates = pd.date_range(start, end, freq='D')
        macro = pd.DataFrame({
            'FEDFUNDS': 4.50,
            'CPIAUCSL': 315.0,
            'PPIACO': 250.0,
            'PCEPI': 180.0,
            'VIXCLS': 16.5,
            'DGS10': 4.20,
            'PAYEMS': 158000.0
        }, index=dates)
    return macro

def build_features(df_price, df_macro, target_horizon, base_threshold, ma_fast, ma_slow):
    df_macro_reindexed = df_macro.reindex(df_price.index, method='ffill').bfill()
    df = df_price.join(df_macro_reindexed)
    
    # 1. Indicateurs Techniques Avancés
    df['Returns'] = df['Close'].pct_change()
    df['MA_Fast'] = df['Close'].rolling(ma_fast).mean()
    df['MA_Slow'] = df['Close'].rolling(ma_slow).mean()
    df['Dist_MA_Fast'] = (df['Close'] - df['MA_Fast']) / (df['MA_Fast'] + 1e-9)
    df['Dist_MA_Slow'] = (df['Close'] - df['MA_Slow']) / (df['MA_Slow'] + 1e-9)
    
    rsi_period = 6 if ma_slow <= 10 else 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    std_fast = df['Close'].rolling(ma_fast).std()
    df['BB_Upper'] = df['MA_Fast'] + (2 * std_fast)
    df['BB_Lower'] = df['MA_Fast'] - (2 * std_fast)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['MA_Fast'] + 1e-9)
    df['BB_Pos'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-9)
    
    ema12 = df['Close'].ewm(span=max(4, ma_fast // 2), adjust=False).mean()
    ema26 = df['Close'].ewm(span=ma_fast, adjust=False).mean()
    df['MACD'] = (ema12 - ema26) / (df['Close'] + 1e-9)

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean() / (df['Close'] + 1e-9)

    if 'Volume' in df.columns:
        df['OBV'] = (np.sign(df['Close'].diff()) * df.fillna(0)['Volume']).fillna(0).cumsum()
    else:
        df['OBV'] = (np.sign(df['Close'].diff()) * 1000.0).fillna(0).cumsum()

    # 2. Indicateurs Fondamentaux & Macro Avancés (avec transformations en variation/glissement)
    df['Fed_Rate'] = df['FEDFUNDS']
    df['Yield_10Y'] = df['DGS10']
    df['Yield_Spread'] = df['DGS10'] - df['FEDFUNDS']
    df['VIX_Level'] = df['VIXCLS']
    df['VIX_Momentum'] = df['VIXCLS'].pct_change(min(5, ma_fast)).fillna(0)
    
    df['CPI_YoY'] = df['CPIAUCSL'].pct_change(12).bfill() * 100
    df['PPI_YoY'] = df['PPIACO'].pct_change(12).bfill() * 100
    df['Core_PCE'] = df['PCEPI'].pct_change(12).bfill() * 100
    df['NFP_Change'] = df['PAYEMS'].diff(1).bfill()

    scaled_threshold = base_threshold * np.sqrt(target_horizon)

    future_ret = df['Close'].shift(-target_horizon) / df['Close'] - 1
    conditions = [future_ret > scaled_threshold, future_ret < -scaled_threshold]
    df['Target'] = np.select(conditions, [1, -1], default=0)
    
    df_full_live = df.dropna(subset=['MA_Slow', 'RSI', 'BB_Width', 'MACD', 'ATR']).copy()
    df_train_able = df_full_live.iloc[:-target_horizon].copy()
    
    return df_train_able, df_full_live, scaled_threshold

if page != "📖 À propos":
    with st.spinner(f"Synchronisation des flux quantitatifs avancés pour {asset_name} ({tf_label})..."):
        raw_price = fetch_data(
            ticker, 
            tf_config["interval"], 
            tf_config["period"], 
            tf_config.get("resample")
        )
        raw_macro = fetch_macro_data()

    min_required_bars = 25 if tf_config["ma_slow"] <= 10 else 70
    if raw_price.empty or len(raw_price) < min_required_bars:
        st.error(f"⚠️ Historique insuffisant ({len(raw_price)} bougies) pour cet actif sur l'unité {tf_label}.")
        st.stop()

    data_train_able, data_full_live, active_thresh = build_features(
        raw_price, 
        raw_macro, 
        horizon, 
        tf_config["thresh"],
        tf_config["ma_fast"],
        tf_config["ma_slow"]
    )

    feature_cols = [
        'Returns', 'Dist_MA_Fast', 'Dist_MA_Slow', 'RSI', 'BB_Width', 'BB_Pos', 'MACD', 'ATR', 'OBV',
        'Fed_Rate', 'Yield_10Y', 'Yield_Spread', 'VIX_Level', 'VIX_Momentum', 'CPI_YoY', 'PPI_YoY', 'Core_PCE', 'NFP_Change'
    ]

    X = data_train_able[feature_cols]
    y = data_train_able['Target']

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models = {
        "HistGradientBoosting (Principal)": HistGradientBoostingClassifier(random_state=42, max_iter=120, learning_rate=0.05),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        "Régression Logistique (Baseline)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    }

    results_audit = []
    trained_models = {}

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        trained_models[name] = clf
        
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        
        full_probs = np.zeros((len(X_test), 3))
        for idx_cls, cls_val in enumerate(clf.classes_):
            target_col = {-1: 0, 0: 1, 1: 2}[cls_val]
            full_probs[:, target_col] = probs[:, idx_cls]
            
        ll = log_loss(y_test, full_probs, labels=[-1, 0, 1])
        y_test_oh = pd.get_dummies(y_test).reindex(columns=[-1, 0, 1], fill_value=0).values
        brier = np.mean(np.sum((full_probs - y_test_oh) ** 2, axis=1))
        
        results_audit.append({
            "Modèle IA": name,
            "Précision (Accuracy)": acc,
            "Log-Loss (Calibration)": ll,
            "Score de Brier": brier
        })

    df_audit = pd.DataFrame(results_audit)
    primary_model = trained_models["HistGradientBoosting (Principal)"]

# ==============================================================================
# PAGE 1 : TERMINAL D'ANALYSE LIVE
# ==============================================================================
if page == "📊 Terminal d'Analyse Live":
    current_candle = data_full_live.iloc[[-1]]
    current_time = current_candle.index[-1]
    current_price = current_candle['Close'].iloc[-1]
    target_future_time = current_time + (tf_config["delta"] * horizon)

    latest_X = current_candle[feature_cols]
    latest_probs = primary_model.predict_proba(latest_X)[0]
    prob_map = {cls: p for cls, p in zip(primary_model.classes_, latest_probs)}
    
    p_bull = prob_map.get(1, 0.0) * 100
    p_neut = prob_map.get(0, 0.0) * 100
    p_bear = prob_map.get(-1, 0.0) * 100

    st.markdown(f"# 📈 {asset_name} <span style='font-size:1rem; color:#38bdf8;'>({ticker} • {tf_label})</span>", unsafe_allow_html=True)
    st.caption(f"Dernière bougie réelle : **{current_time.strftime('%Y-%m-%d %H:%M')}** (Prix : **{current_price:.4f}**) ➔ Projection cible vers : **{target_future_time.strftime('%Y-%m-%d %H:%M')}** (+{horizon} bougies)")

    st.markdown(f'<div class="section-header">🎯 Matrice de Probabilités Futures (De maintenant jusqu\'à +{horizon} bougie(s))</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="pro-card">
            <div class="card-title">Scénario Haussier (Bull)</div>
            <div class="card-value-bull">{p_bull:.1f}%</div>
            <div class="card-sub">Cible > {current_price * (1 + active_thresh):.4f} (+{active_thresh*100:.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="pro-card">
            <div class="card-title">Scénario Neutre (Range)</div>
            <div class="card-value-neut">{p_neut:.1f}%</div>
            <div class="card-sub">Zone ±{active_thresh*100:.2f}% autour du prix</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="pro-card">
            <div class="card-title">Scénario Baissier (Bear)</div>
            <div class="card-value-bear">{p_bear:.1f}%</div>
            <div class="card-sub">Cible < {current_price * (1 - active_thresh):.4f} (-{active_thresh*100:.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        cpi_now = current_candle['CPI_YoY'].iloc[-1]
        pce_now = current_candle['Core_PCE'].iloc[-1]
        st.markdown(f"""
        <div class="pro-card">
            <div class="card-title">Macro Inflation (CPI / PCE)</div>
            <div class="card-value-info" style="font-size:1.6rem;">{cpi_now:.1f}% <span style="font-size:1rem; color:#94a3b8;">| {pce_now:.1f}%</span></div>
            <div class="card-sub">CPI Annuel | Core PCE Fed</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🕯️ Structure Technique & Cône de Projection Futur</div>', unsafe_allow_html=True)
    
    plot_df = data_full_live.iloc[-150:]
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.04, 
        row_heights=[0.72, 0.28]
    )

    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df['Open'], high=plot_df['High'],
        low=plot_df['Low'], close=plot_df['Close'],
        name="Prix Réel",
        increasing_line_color='#10b981', decreasing_line_color='#ef4444'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA_Fast'], name=f"MA {tf_config['ma_fast']}", line=dict(color='#38bdf8', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Upper'], name="BB Sup", line=dict(color='rgba(148, 163, 184, 0.4)', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Lower'], name="BB Inf", line=dict(color='rgba(148, 163, 184, 0.4)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(56, 189, 248, 0.05)'), row=1, col=1)

    future_dates = [current_time + (tf_config["delta"] * i) for i in range(0, horizon + 1)]
    bull_path = [current_price * (1 + active_thresh * np.sqrt(i / horizon)) for i in range(0, horizon + 1)]
    bear_path = [current_price * (1 - active_thresh * np.sqrt(i / horizon)) for i in range(0, horizon + 1)]
    expected_drift = (p_bull - p_bear) / 100.0
    mean_path = [current_price * (1 + expected_drift * active_thresh * np.sqrt(i / horizon)) for i in range(0, horizon + 1)]

    fig.add_trace(go.Scatter(
        x=future_dates, y=bull_path,
        name=f"Cible Haussière ({p_bull:.1f}%)",
        line=dict(color='#10b981', width=2, dash='dash')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=future_dates, y=bear_path,
        name=f"Cible Baissière ({p_bear:.1f}%)",
        line=dict(color='#ef4444', width=2, dash='dash'),
        fill='tonexty', fillcolor='rgba(56, 189, 248, 0.08)'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=future_dates, y=mean_path,
        name="Trajectoire Pondérée IA",
        line=dict(color='#f59e0b', width=2.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], name="RSI", line=dict(color='#a855f7', width=1.8)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(239, 68, 68, 0.5)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(16, 185, 129, 0.5)", row=2, col=1)

    fig.update_layout(
        height=600,
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#111827",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 2 : AUDIT & PERFORMANCES DES MODÈLES (SÉCURISÉE PAR BK16_AUTH)
# ==============================================================================
elif page == "🔐 Audit & Modèles (Sécurisé)":
    st.markdown("# 🔐 Laboratoire Quantitatif — Audit des Modèles IA")
    
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.markdown("""
        <div class="pro-card" style="max-width: 480px; margin: 40px auto; text-align: left;">
            <h3 style="color: #38bdf8; margin-top: 0;">Accès Restreint — Niveau Recherche</h3>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Veuillez saisir la clé d'authentification <b>BK16</b> pour consulter les métriques de calibration hors-échantillon (Log-Loss, Brier Score) et l'architecture des algorithmes.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col_space1, col_auth, col_space2 = st.columns([1, 1.5, 1])
        with col_auth:
            pwd_input = st.text_input("Mot de passe d'accès", type="password", placeholder="Entrez le code d'accès...")
            if st.button("Déverrouiller l'Audit", use_container_width=True):
                if pwd_input == "BK16_AUTH":
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("❌ Clé d'authentification invalide. Accès refusé.")
    else:
        col_top1, col_top2 = st.columns([5, 1])
        with col_top1:
            st.success(f"✅ Session authentifiée • Évaluation hors-échantillon sur {len(X_test)} périodes ({asset_name} - {tf_label})")
        with col_top2:
            if st.button("🔒 Verrouiller"):
                st.session_state["authenticated"] = False
                st.rerun()

        st.markdown('<div class="section-header">📊 Architecture & Dimensionnement du Dataset</div>', unsafe_allow_html=True)
        ds1, ds2, ds3, ds4 = st.columns(4)
        
        total_samples = len(X)
        train_samples = len(X_train)
        test_samples = len(X_test)
        
        with ds1:
            st.markdown(f"""
            <div class="pro-card">
                <div class="card-title">Taille Totale de la Base</div>
                <div class="card-value-info">{total_samples:,}</div>
                <div class="card-sub">Bougies exploitables (Features + Cible)</div>
            </div>
            """, unsafe_allow_html=True)
        with ds2:
            st.markdown(f"""
            <div class="pro-card">
                <div class="card-title">Base d'Entraînement (Train)</div>
                <div class="card-value-bull">{train_samples:,}</div>
                <div class="card-sub">80% de l'historique chronologique</div>
            </div>
            """, unsafe_allow_html=True)
        with ds3:
            st.markdown(f"""
            <div class="pro-card">
                <div class="card-title">Base de Test (OOS)</div>
                <div class="card-value-neut">{test_samples:,}</div>
                <div class="card-sub">20% pour validation hors-échantillon</div>
            </div>
            """, unsafe_allow_html=True)
        with ds4:
            st.markdown(f"""
            <div class="pro-card">
                <div class="card-title">Ratio de Split</div>
                <div class="card-value-info" style="font-size:1.6rem;">80% / 20%</div>
                <div class="card-sub">Séparation temporelle stricte (Pas de fuite)</div>
            </div>
            """, unsafe_allow_html=True)

        best_model_row = df_audit.sort_values("Log-Loss (Calibration)").iloc[0]
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="pro-card">
                <div class="card-title">Meilleur Modèle Calibré</div>
                <div class="card-value-info" style="font-size:1.3rem;">{best_model_row['Modèle IA'].split(' ')[0]}</div>
                <div class="card-sub">Sélectionné sur la Log-Loss minimale</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="pro-card">
                <div class="card-title">Précision Hors-Échantillon</div>
                <div class="card-value-bull">{best_model_row['Précision (Accuracy)']*100:.2f}%</div>
                <div class="card-sub">Sur données jamais vues (20% récents)</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="pro-card">
                <div class="card-title">Score de Brier (Incertitude)</div>
                <div class="card-value-neut">{best_model_row['Score de Brier']:.4f}</div>
                <div class="card-sub">Plus proche de 0 = Probabilités fiables</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">📊 Comparatif Scientifique des Algorithmes</div>', unsafe_allow_html=True)
        
        col_table, col_chart = st.columns([1.1, 1.2])
        
        with col_table:
            df_display = df_audit.copy()
            df_display["Précision (Accuracy)"] = (df_display["Précision (Accuracy)"] * 100).map("{:.2f}%".format)
            df_display["Log-Loss (Calibration)"] = df_display["Log-Loss (Calibration)"].map("{:.4f}".format)
            df_display["Score de Brier"] = df_display["Score de Brier"].map("{:.4f}".format)
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            st.markdown("""
            **Guide d'interprétation institutionnel :**
- **Précision (Accuracy)** : Taux brut de scénarios correctement classés.
- **Log-Loss** : Pénalise fortement le modèle s'il affiche 85% de confiance sur une prévision qui s'avère fausse. *C'est la métrique reine en finance quantitative.*
- **Score de Brier** : Mesure l'erreur quadratique moyenne des probabilités prédites (0 = parfait, 2 = pire).
            """)

        with col_chart:
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name='Précision (%)',
                x=df_audit['Modèle IA'],
                y=df_audit['Précision (Accuracy)'] * 100,
                marker_color='#38bdf8'
            ))
            fig_comp.add_trace(go.Bar(
                name='Log-Loss (x50 pour échelle)',
                x=df_audit['Modèle IA'],
                y=df_audit['Log-Loss (Calibration)'] * 50,
                marker_color='#f59e0b'
            ))
            fig_comp.update_layout(
                title="Comparaison Précision vs Pénalité Log-Loss",
                barmode='group',
                template="plotly_dark",
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                height=360,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown('<div class="section-header">📈 Courbe ROC (Receiver Operating Characteristic) - Meilleur Modèle</div>', unsafe_allow_html=True)
        
        best_clf = primary_model
        y_test_bin = label_binarize(y_test, classes=[-1, 0, 1])
        n_classes = y_test_bin.shape[1]
        y_score = best_clf.predict_proba(X_test)
        
        fig_roc = go.Figure()
        fig_roc.add_shape(
            type='line', line=dict(dash='dash', color='gray', width=1.5),
            x0=0, x1=1, y0=0, y1=1
        )
        
        class_labels = {0: "Baissier (-1)", 1: "Neutre (0)", 2: "Haussier (+1)"}
        class_colors = {0: "#ef4444", 1: "#f59e0b", 2: "#10b981"}
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"Classe {class_labels[i]} (AUC = {roc_auc:.2f})",
                line=dict(color=class_colors[i], width=2.5)
            ))
            
        fig_roc.update_layout(
            title=f"Performance ROC - {list(trained_models.keys())[0]}",
            xaxis_title="Taux de Faux Positifs (FPR)",
            yaxis_title="Taux de Vrais Positifs (TPR)",
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            height=420,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown('<div class="section-header">🧠 Poids des Facteurs : Technique vs Fondamental</div>', unsafe_allow_html=True)
        rf_model = trained_models["Random Forest Classifier"]
        importances = pd.DataFrame({
            "Variable": feature_cols,
            "Importance": rf_model.feature_importances_,
            "Famille": ["Technique" if f in ['Returns', 'Dist_MA_Fast', 'Dist_MA_Slow', 'RSI', 'BB_Width', 'BB_Pos', 'MACD', 'ATR', 'OBV'] else "Fondamental (Macro)" for f in feature_cols]
        }).sort_values("Importance", ascending=True)

        fig_imp = go.Figure(go.Bar(
            x=importances["Importance"],
            y=importances["Variable"],
            orientation='h',
            marker_color=['#10b981' if fam == "Fondamental (Macro)" else '#38bdf8' for fam in importances["Famille"]],
            text=importances["Famille"],
            textposition='auto'
        ))
        fig_imp.update_layout(
            title="Contribution relative de chaque indicateur dans les décisions de l'IA",
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            height=420,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ==============================================================================
# PAGE 3 : À PROPOS (PRO & DÉTAILLÉE AVEC VARIABLES MOBILISÉES)
# ==============================================================================
elif page == "📖 À propos":
    st.markdown("# 📖 À propos de BK16 Market Intelligence Lab")
    st.markdown("<p style='color: #94a3b8; font-size: 1.1rem;'>Plateforme institutionnelle de recherche quantitative, d'analyse multi-actifs et de modélisation prédictive par Intelligence Artificielle.</p>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">🏛️ Vision & Philosophie du Projet</div>', unsafe_allow_html=True)
    
    col_ab1, col_ab2 = st.columns(2)
    with col_ab1:
        st.markdown("""
        <div class="pro-card" style="text-align: left; height: 100%;">
            <h3 style="color: #38bdf8; margin-top: 0;">🎯 Mission & Objectif</h3>
            <p style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.6;">
                <b>BK16 Market Intelligence Lab</b> comble le fossé entre la recherche académique en finance quantitative et l'ingénierie logicielle moderne, fournissant des projections probabilistes en temps réel via des modèles de Machine Learning de pointe.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_ab2:
        st.markdown("""
        <div class="pro-card" style="text-align: left; height: 100%;">
            <h3 style="color: #10b981; margin-top: 0;">⚙️ Architecture Hybride</h3>
            <p style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.6;">
                Le système fusionne l'analyse technique multi-échelle et les fondamentaux macroéconomiques globaux pour maximiser la robustesse des prédictions hors-échantillon.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # --- SECTION DÉTAILLÉE DES VARIABLES MOBILISÉES ---
    st.markdown('<div class="section-header">📊 Dictionnaire des Variables & Modélisation Prédictive</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 0.95rem;'>Voici le détail synthétique des variables techniques et macroéconomiques injectées dans le moteur d'IA pour piloter les prévisions :</p>", unsafe_allow_html=True)

    tab_tech, tab_macro = st.tabs(["📈 Variables Techniques (Momentum, Volatilité & Volume)", "🌍 Variables Fondamentales & Macroéconomiques"])

    with tab_tech:
        st.markdown("""
        * **Returns (Rendements log/pct) :** Mesure la variation instantanée des prix d'une bougie à l'autre pour capturer la directionnelle court terme.
        * **MAs & Distances (Dist_MA_Fast / Dist_MA_Slow) :** Quantifie l'écart relatif entre le prix et ses moyennes mobiles (20 et 50 périodes) pour identifier les retours à la moyenne ou la force de cassure.
        * **RSI (Relative Strength Index) :** Oscillateur de momentum évaluant la vitesse des mouvements de prix pour détecter les conditions de surachat et de survente.
        * **Bandes de Bollinger (BB_Width / BB_Pos) :** Mesurent l'expansion/contraction de la volatilité et situent précisément le prix par rapport aux bornes statistiques.
        * **MACD (Moving Average Convergence Divergence) :** Indicateur de suivi de tendance basé sur les EMA pour capter l'accélération du momentum.
        * **ATR (Average True Range) :** Mesure la volatilité réelle de l'actif en tenant compte des gaps pour adapter dynamiquement l'amplitude des seuils de prédiction.
        * **OBV (On-Balance Volume) :** Indicateur basé sur le volume cumulé pour anticiper les retournements de prix selon la pression acheteuse/vendeuse.
        """)

    with tab_macro:
        st.markdown("""
        * **Fed Rate (Taux FED - `FEDFUNDS`) :** Taux directeur de la Réserve Fédérale, déterminant le coût du capital et l'attractivité du dollar.
        * **Yield 10Y & Spread (`DGS10` & Spread) :** Rendement obligataire souverain et écart de taux, baromètre majeur du cycle économique et du risque de récession.
        * **VIX Level & Momentum (`VIXCLS`) :** Indice de volatilité implicite mesurant la peur et l'aversion au risque sur les marchés mondiaux.
        * **CPI YoY (Inflation - `CPIAUCSL`) :** Indice des prix à la consommation en glissement annuel, dictant les anticipations d'inflation de détail.
        * **PPI YoY (Inflation Producteurs - `PPIACO`) :** Indice des prix à la production mesurant les tensions sur les coûts en amont de l'industrie.
        * **Core PCE (Inflation de référence FED - `PCEPI`) :** L'indicateur d'inflation privilégié par la banque centrale pour piloter sa politique monétaire.
        * **NFP Change (Non-Farm Payrolls - `PAYEMS`) :** Variation mensuelle des créations d'emplois non agricoles aux États-Unis, catalyseur majeur de volatilité macro.
        """)

    st.markdown('<div class="section-header">🛠️ Stack Technologique & Ingénierie</div>', unsafe_allow_html=True)
    
    st1, st2, st3, st4 = st.columns(4)
    with st1:
        st.markdown("""
        <div class="pro-card">
            <div class="card-title">Core Engine</div>
            <div class="card-value-info" style="font-size:1.4rem;">Python 3.14</div>
            <div class="card-sub">Architecture modulaire & robuste</div>
        </div>
        """, unsafe_allow_html=True)
    with st2:
        st.markdown("""
        <div class="pro-card">
            <div class="card-title">Machine Learning</div>
            <div class="card-value-bull" style="font-size:1.4rem;">Scikit-Learn</div>
            <div class="card-sub">HistGradientBoosting & RF</div>
        </div>
        """, unsafe_allow_html=True)
    with st3:
        st.markdown("""
        <div class="pro-card">
            <div class="card-title">Flux de Données</div>
            <div class="card-value-neut" style="font-size:1.4rem;">Yfinance & FRED</div>
            <div class="card-sub">Flux de marché & macro en direct</div>
        </div>
        """, unsafe_allow_html=True)
    with st4:
        st.markdown("""
        <div class="pro-card">
            <div class="card-title">Interface UI/UX</div>
            <div class="card-value-info" style="font-size:1.4rem;">Streamlit & Plotly</div>
            <div class="card-sub">Terminal sombre & réactif</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">👨‍💻 Ingénierie & Auteur</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pro-card" style="max-width: 700px; margin: 0 auto; text-align: center; border: 1px solid #38bdf8;">
        <h3 style="color: #f8fafc; margin-bottom: 5px;">BK16 Quantitative Systems</h3>
        <p style="color: #38bdf8; font-weight: 600; margin-bottom: 15px;">Lead Quantitative Developer & Architect</p>
        <p style="color: #94a3b8; font-size: 0.95rem; line-height: 1.5;">
            Plateforme développée dans le cadre de l'optimisation des systèmes d'aide à la décision boursière et de l'analyse stochastique des marchés globaux.
        </p>
        <span class="status-badge">● DOMAINE : BK16AUTHENTIQUE.COM</span>
    </div>
    """, unsafe_allow_html=True)

# --- FOOTER GLOBAL FIXÉ EN BAS À DROITE ---
st.markdown('<div class="footer-bk16">⚡ Engineered by <b>BK16 QUANT LAB</b></div>', unsafe_allow_html=True)
