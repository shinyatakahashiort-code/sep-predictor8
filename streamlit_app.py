"""
SE_p予測 - Streamlit Webアプリケーション
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from predictor import SEPredictor, ModelEnsemble
import json

# ページ設定
st.set_page_config(
    page_title="SE_p予測アプリ",
    page_icon="👁️",
    layout="wide"
)

st.markdown("# 👁️ SE_p予測システム")
st.markdown("眼科検査データから**SE_p（球面等価屈折度）**を予測します。")

# モデルの読み込み
@st.cache_resource
def load_models():
    try:
        mlp = SEPredictor(model_name='MLP')
        extra_trees = SEPredictor(model_name='ExtraTrees')
        catboost = SEPredictor(model_name='CatBoost')
        ensemble = ModelEnsemble()
        return {'MLP': mlp, 'ExtraTrees': extra_trees, 'CatBoost': catboost, 'Ensemble': ensemble}
    except Exception as e:
        st.error(f"モデルの読み込みに失敗: {e}")
        return None

models = load_models()
if models is None:
    st.stop()

# サイドバー
st.sidebar.header("⚙️ 設定")
model_choice = st.sidebar.selectbox(
    "予測モデルを選択",
    ['Ensemble（推奨）', 'MLP', 'ExtraTrees', 'CatBoost']
)

# 入力フォーム
st.markdown("## 📝 入力データ")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("年齢", min_value=0, max_value=100, value=50)
    k_avg = st.number_input("K（AVG）", min_value=40.0, max_value=50.0, value=44.0, format="%.2f")

with col2:
    gender = st.selectbox("性別", [0, 1], format_func=lambda x: "男性" if x == 0 else "女性")
    al = st.number_input("AL - 眼軸長", min_value=20.0, max_value=30.0, value=24.0, format="%.2f")

with col3:
    lt = st.number_input("LT - 水晶体厚", min_value=2.0, max_value=6.0, value=4.0, format="%.2f")
    acd = st.number_input("ACD - 前房深度", min_value=2.0, max_value=5.0, value=3.0, format="%.2f")

user_input = {
    '年齢': age,
    '性別': gender,
    'K（AVG）': k_avg,
    'AL': al,
    'LT': lt,
    'ACD': acd
}

# 予測
if st.button("🔮 予測を実行", type="primary", use_container_width=True):
    with st.spinner("予測中..."):
        try:
            if model_choice == 'Ensemble（推奨）':
                result = models['Ensemble'].predict_with_details(user_input)
                is_ensemble = True
            else:
                result = models[model_choice].predict_with_details(user_input)
                is_ensemble = False
            
            st.markdown("## 📊 予測結果")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("予測値 (SE_p)", f"{result['prediction']:.4f}")
            with col2:
                st.metric("95%信頼区間（下限）", f"{result['confidence_interval_95']['lower']:.4f}")
            with col3:
                st.metric("95%信頼区間（上限）", f"{result['confidence_interval_95']['upper']:.4f}")
            
            if is_ensemble:
                st.markdown("### アンサンブル詳細")
                individual_preds = result['individual_predictions']
                pred_df = pd.DataFrame({
                    'モデル': list(individual_preds.keys()),
                    '予測値': [f"{v:.4f}" for v in individual_preds.values()]
                })
                st.dataframe(pred_df, use_container_width=True)
            
            if result['validation']['warnings']:
                st.warning("⚠️ " + "\n".join(result['validation']['warnings']))
            
        except Exception as e:
            st.error(f"予測エラー: {e}")

# フッター
st.markdown("---")
with st.expander("ℹ️ モデル情報"):
    st.markdown("""
    ### 使用モデル
    | モデル | R² Score | RMSE |
    |--------|----------|------|
    | MLP | 0.9150 ± 0.0116 | 0.7830 |
    | ExtraTrees | 0.9145 ± 0.0135 | 0.7846 |
    | CatBoost | 0.9107 ± 0.0131 | 0.8027 |
    """)
