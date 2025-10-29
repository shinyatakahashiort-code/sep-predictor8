"""SE_p予測 - Streamlit Webアプリケーション"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ページ設定
st.set_page_config(page_title="SE_p予測", page_icon="👁️", layout="wide")

st.markdown("# 👁️ SE_p予測システム")
st.markdown("眼科検査データから**SE_p（球面等価屈折度）**を予測します。")

# モデルの読み込み
@st.cache_resource
def load_models():
    try:
        # predictor.pyをインポート
        from predictor import SEPredictor, ModelEnsemble
        
        mlp = SEPredictor(model_name='MLP')
        extra_trees = SEPredictor(model_name='ExtraTrees')
        catboost = SEPredictor(model_name='CatBoost')
        ensemble = ModelEnsemble()
        
        return {
            'MLP': mlp,
            'ExtraTrees': extra_trees,
            'CatBoost': catboost,
            'Ensemble': ensemble
        }
    except Exception as e:
        st.error(f"❌ モデル読み込みエラー: {e}")
        st.error(f"エラー詳細: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())
        return None

# モデルを読み込み
with st.spinner("モデルを読み込み中..."):
    models = load_models()

if models is None:
    st.stop()

st.success("✅ モデルの読み込み完了！")

# サイドバー設定
st.sidebar.header("⚙️ 設定")
model_choice = st.sidebar.selectbox(
    "予測モデルを選択",
    ['Ensemble（推奨）', 'MLP', 'ExtraTrees', 'CatBoost'],
    help="Ensembleは3つのモデルの加重平均です"
)

# 入力フォーム
st.markdown("## 📝 入力データ")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("年齢", min_value=0, max_value=100, value=50)
    k_avg = st.number_input("K（AVG）- 角膜曲率", min_value=40.0, max_value=8.5, value=6.0, step=0.1, format="%.2f")

with col2:
    gender = st.selectbox("性別", [0, 1], format_func=lambda x: "男性" if x == 0 else "女性")
    al = st.number_input("AL - 眼軸長 (mm)", min_value=20.0, max_value=30.0, value=24.0, step=0.1, format="%.2f")

with col3:
    lt = st.number_input("LT - 水晶体厚 (mm)", min_value=2.0, max_value=6.0, value=4.0, step=0.1, format="%.2f")
    acd = st.number_input("ACD - 前房深度 (mm)", min_value=2.0, max_value=5.0, value=3.0, step=0.1, format="%.2f")

# 入力データ
user_input = {
    '年齢': age,
    '性別': gender,
    'K（AVG）': k_avg,
    'AL': al,
    'LT': lt,
    'ACD': acd
}

# 予測ボタン
st.markdown("---")

if st.button("🔮 予測を実行", type="primary", use_container_width=True):
    with st.spinner("予測中..."):
        try:
            # モデルの選択と予測
            if model_choice == 'Ensemble（推奨）':
                result = models['Ensemble'].predict_with_details(user_input)
                is_ensemble = True
            else:
                result = models[model_choice].predict_with_details(user_input)
                is_ensemble = False
            
            # 結果表示
            st.markdown("## 📊 予測結果")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="予測値 (SE_p)",
                    value=f"{result['prediction']:.4f}",
                    help="球面等価屈折度の予測値"
                )
            
            with col2:
                st.metric(
                    label="95%信頼区間（下限）",
                    value=f"{result['confidence_interval_95']['lower']:.4f}"
                )
            
            with col3:
                st.metric(
                    label="95%信頼区間（上限）",
                    value=f"{result['confidence_interval_95']['upper']:.4f}"
                )
            
            # アンサンブル詳細
            if is_ensemble:
                st.markdown("### アンサンブル詳細")
                
                individual_preds = result['individual_predictions']
                weights = result['weights']
                
                pred_df = pd.DataFrame({
                    'モデル': list(individual_preds.keys()),
                    '予測値': [f"{v:.4f}" for v in individual_preds.values()],
                    '重み': [f"{weights[k]:.3f}" for k in individual_preds.keys()]
                })
                
                st.dataframe(pred_df, use_container_width=True)
                
                # グラフ表示
                fig = px.bar(
                    x=list(individual_preds.keys()),
                    y=list(individual_preds.values()),
                    labels={'x': 'モデル', 'y': '予測値'},
                    title='各モデルの予測値比較'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"📌 予測のばらつき (標準偏差): {result['ensemble_std']:.4f}")
            
            else:
                # 単一モデルの詳細
                st.markdown("### モデル性能")
                
                perf = result['model_performance']
                err = result['expected_error']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("R² Score", f"{perf['r2_mean']:.4f}")
                
                with col2:
                    st.metric("R² Std", f"{perf['r2_std']:.4f}")
                
                with col3:
                    st.metric("Expected MAE", f"{err['mae']:.4f}")
                
                with col4:
                    st.metric("Expected RMSE", f"{err['rmse']:.4f}")
            
            # 警告表示
            validation = result['validation']
            
            if validation['warnings']:
                st.warning("⚠️ **警告**")
                for warning in validation['warnings']:
                    st.write(f"- {warning}")
            
            if validation['errors']:
                st.error("❌ **エラー**")
                for error in validation['errors']:
                    st.write(f"- {error}")
            
            # 入力データの確認
            with st.expander("📋 入力データの確認"):
                input_df = pd.DataFrame([user_input])
                st.dataframe(input_df.T, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ 予測エラー: {e}")
            import traceback
            st.code(traceback.format_exc())

# フッター
st.markdown("---")

with st.expander("ℹ️ モデル情報とパフォーマンス"):
    st.markdown("""
    ### 使用モデル
    
    反復付きネスト化クロスバリデーション（5-fold outer, 3-fold inner, 3 repeats）で評価：
    
    | モデル | R² Score | RMSE | MAE |
    |--------|----------|------|-----|
    | **MLP** (Neural Network) | 0.9150 ± 0.0116 | 0.7830 ± 0.0342 | 0.6042 ± 0.0271 |
    | **Extra Trees** | 0.9145 ± 0.0135 | 0.7846 ± 0.0439 | 0.5766 ± 0.0291 |
    | **CatBoost** | 0.9107 ± 0.0131 | 0.8027 ± 0.0410 | 0.6213 ± 0.0340 |
    
    **Ensemble**は3つのモデルの性能に基づく加重平均により、より安定した予測を提供します。
    
    ### 特徴量
    
    - **年齢**: 患者の年齢（歳）
    - **性別**: 0=男性, 1=女性
    - **K(AVG)**: 角膜曲率半径の平均（ジオプトリー）
    - **AL**: 眼軸長（mm）
    - **LT**: 水晶体厚（mm）
    - **ACD**: 前房深度（mm）
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 データ情報")
st.sidebar.info(f"""
**トレーニングデータ**: 1,483 samples  
**評価方法**: Repeated Nested CV  
**作成日**: 2025-10-29
""")
