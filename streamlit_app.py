"""SE_p予測 - Streamlit Webアプリケーション"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

# ページ設定
st.set_page_config(page_title="SE_p予測", page_icon="👁️", layout="wide")

st.markdown("# 👁️ SE_p予測システム")
st.markdown("眼科検査データから**SE_p（球面等価屈折度）**を予測します。")

# モデルの読み込み
@st.cache_resource
def load_models():
    try:
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
        import traceback
        st.code(traceback.format_exc())
        return None

with st.spinner("モデルを読み込み中..."):
    models = load_models()

if models is None:
    st.stop()

st.success("✅ モデルの読み込み完了！")

# サイドバー設定
st.sidebar.header("⚙️ 設定")

# 予測モード選択
prediction_mode = st.sidebar.radio(
    "予測モード",
    ["単一予測", "CSV一括予測"],
    help="単一の症例または複数症例のCSVファイルを選択"
)

model_choice = st.sidebar.selectbox(
    "予測モデルを選択",
    ['Ensemble（推奨）', 'MLP', 'ExtraTrees', 'CatBoost'],
    help="Ensembleは3つのモデルの加重平均です"
)

# ========================================
# 単一予測モード
# ========================================
if prediction_mode == "単一予測":
    st.markdown("## 📝 入力データ")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("年齢", min_value=3, max_value=18, value=9)
        k_avg = st.number_input("K-AVG (角膜曲率)", min_value=7.0, max_value=8.7, value=7.4, step=0.1, format="%.2f")
    
    with col2:
        gender = st.selectbox("性別", [0, 1], format_func=lambda x: "男性" if x == 0 else "女性")
        al = st.number_input("AL (眼軸長)", min_value=20.0, max_value=30.0, value=24.0, step=0.1, format="%.2f")
    
    with col3:
        lt = st.number_input("LT (水晶体厚)", min_value=2.0, max_value=6.0, value=4.0, step=0.1, format="%.2f")
        acd = st.number_input("ACD (前房深度)", min_value=2.0, max_value=5.0, value=3.0, step=0.1, format="%.2f")
    
    user_input = {
        '年齢': age,
        '性別': gender,
        'K（AVG）': k_avg,
        'AL': al,
        'LT': lt,
        'ACD': acd
    }
    
    st.markdown("---")
    
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
                    st.metric("95%信頼区間 (下限)", f"{result['confidence_interval_95']['lower']:.4f}")
                
                with col3:
                    st.metric("95%信頼区間 (上限)", f"{result['confidence_interval_95']['upper']:.4f}")
                
                if is_ensemble:
                    st.markdown("### 📈 アンサンブル詳細")
                    
                    individual_preds = result['individual_predictions']
                    weights = result['weights']
                    
                    pred_df = pd.DataFrame({
                        'モデル': list(individual_preds.keys()),
                        '予測値': [f"{v:.4f}" for v in individual_preds.values()],
                        '重み': [f"{weights[k]:.3f}" for k in individual_preds.keys()]
                    })
                    
                    st.dataframe(pred_df, use_container_width=True)
                    
                    fig = px.bar(
                        x=list(individual_preds.keys()),
                        y=list(individual_preds.values()),
                        labels={'x': 'モデル', 'y': '予測値'},
                        title='各モデルの予測値比較'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info(f"📌 予測のばらつき: {result['ensemble_std']:.4f}")
                
                else:
                    st.markdown("### 📊 モデル性能")
                    
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
                
                validation = result['validation']
                
                if validation['warnings']:
                    st.warning("⚠️ 警告")
                    for warning in validation['warnings']:
                        st.write(f"• {warning}")
                
                with st.expander("📋 入力データの確認"):
                    st.dataframe(pd.DataFrame([user_input]).T, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ 予測エラー: {e}")
                import traceback
                st.code(traceback.format_exc())

# ========================================
# CSV一括予測モード
# ========================================
else:
    st.markdown("## 📤 CSVファイルをアップロード")
    
    # CSVテンプレートのダウンロード
    st.markdown("### 📋 CSVフォーマット")
    
    template_data = {
        '年齢': [9, 10, 8],
        '性別': [0, 1, 0],
        'K（AVG）': [7.4, 7.6, 7.2],
        'AL': [24.0, 24.5, 23.8],
        'LT': [4.0, 4.2, 3.9],
        'ACD': [3.0, 3.1, 2.9]
    }
    template_df = pd.DataFrame(template_data)
    
    st.write("**必要な列:**")
    st.dataframe(template_df, use_container_width=True)
    
    # テンプレートダウンロード
    csv_template = template_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 テンプレートをダウンロード",
        data=csv_template,
        file_name="se_prediction_template.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "CSVファイルを選択",
        type=['csv'],
        help="上記のフォーマットに従ったCSVファイルをアップロードしてください"
    )
    
    if uploaded_file is not None:
        try:
            # CSVの読み込み
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            
            st.success(f"✅ ファイル読み込み成功: {len(df)} 件のデータ")
            
            # データプレビュー
            st.markdown("### 📊 データプレビュー")
            st.dataframe(df.head(10), use_container_width=True)
            
            # 必要な列のチェック
            required_columns = ['年齢', '性別', 'K（AVG）', 'AL', 'LT', 'ACD']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ 不足している列: {', '.join(missing_columns)}")
                st.stop()
            
            # 予測実行ボタン
            if st.button("🔮 一括予測を実行", type="primary", use_container_width=True):
                with st.spinner(f"{len(df)} 件のデータを予測中..."):
                    try:
                        # 選択したモデルを取得
                        if model_choice == 'Ensemble（推奨）':
                            model = models['Ensemble']
                        else:
                            model = models[model_choice]
                        
                        # 各行を予測
                        predictions = []
                        lower_bounds = []
                        upper_bounds = []
                        warnings_list = []
                        
                        progress_bar = st.progress(0)
                        
                        for idx, row in df.iterrows():
                            # 入力データの準備
                            input_data = {
                                '年齢': row['年齢'],
                                '性別': row['性別'],
                                'K（AVG）': row['K（AVG）'],
                                'AL': row['AL'],
                                'LT': row['LT'],
                                'ACD': row['ACD']
                            }
                            
                            # 予測実行
                            result = model.predict_with_details(input_data)
                            
                            predictions.append(result['prediction'])
                            lower_bounds.append(result['confidence_interval_95']['lower'])
                            upper_bounds.append(result['confidence_interval_95']['upper'])
                            
                            # 警告を収集
                            if result['validation']['warnings']:
                                warnings_list.append(f"行{idx+1}: " + "; ".join(result['validation']['warnings']))
                            
                            # プログレスバー更新
                            progress_bar.progress((idx + 1) / len(df))
                        
                        progress_bar.empty()
                        
                        # 結果をデータフレームに追加
                        result_df = df.copy()
                        result_df['SE_p予測値'] = predictions
                        result_df['95%CI_下限'] = lower_bounds
                        result_df['95%CI_上限'] = upper_bounds
                        
                        st.success("✅ 予測完了！")
                        
                        # 結果の表示
                        st.markdown("## 📊 予測結果")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # 統計情報
                        st.markdown("### 📈 統計情報")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("症例数", len(result_df))
                        with col2:
                            st.metric("平均予測値", f"{np.mean(predictions):.4f}")
                        with col3:
                            st.metric("最小値", f"{np.min(predictions):.4f}")
                        with col4:
                            st.metric("最大値", f"{np.max(predictions):.4f}")
                        
                        # 予測値の分布
                        st.markdown("### 📊 予測値の分布")
                        fig = px.histogram(
                            x=predictions,
                            nbins=30,
                            labels={'x': 'SE_p予測値', 'y': '度数'},
                            title='SE_p予測値のヒストグラム'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 散布図
                        st.markdown("### 📊 特徴量との関係")
                        feature_choice = st.selectbox(
                            "表示する特徴量を選択",
                            ['年齢', 'K（AVG）', 'AL', 'LT', 'ACD']
                        )
                        
                        fig2 = px.scatter(
                            x=result_df[feature_choice],
                            y=predictions,
                            labels={'x': feature_choice, 'y': 'SE_p予測値'},
                            title=f'{feature_choice} vs SE_p予測値',
                            trendline="lowess"
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # 警告の表示
                        if warnings_list:
                            with st.expander(f"⚠️ 警告 ({len(warnings_list)} 件)"):
                                for warning in warnings_list:
                                    st.write(f"• {warning}")
                        
                        # 結果のダウンロード
                        st.markdown("### 💾 結果のダウンロード")
                        
                        # CSV形式
                        csv_result = result_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 予測結果をCSVでダウンロード",
                            data=csv_result,
                            file_name=f"se_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Excel形式
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            result_df.to_excel(writer, index=False, sheet_name='予測結果')
                        excel_data = output.getvalue()
                        
                        st.download_button(
                            label="📥 予測結果をExcelでダウンロード",
                            data=excel_data,
                            file_name=f"se_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ 予測エラー: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        except Exception as e:
            st.error(f"❌ ファイル読み込みエラー: {e}")
            st.write("CSVファイルのエンコーディングや形式を確認してください。")

# フッター
st.markdown("---")

with st.expander("ℹ️ モデル情報"):
    st.markdown("""
    ### 使用モデル
    
    | モデル | R² Score | RMSE | MAE |
    |--------|----------|------|-----|
    | **MLP** | 0.9150 ± 0.0116 | 0.7830 ± 0.0342 | 0.6042 ± 0.0271 |
    | **Extra Trees** | 0.9145 ± 0.0135 | 0.7846 ± 0.0439 | 0.5766 ± 0.0291 |
    | **CatBoost** | 0.9107 ± 0.0131 | 0.8027 ± 0.0410 | 0.6213 ± 0.0340 |
    
    ### CSV一括予測の使い方
    
    1. **テンプレートをダウンロード**して、Excelなどで編集
    2. 必要な列: 年齢、性別、K（AVG）、AL、LT、ACD
    3. CSVファイルをアップロード
    4. 「一括予測を実行」をクリック
    5. 結果をCSVまたはExcelでダウンロード
    """)

st.sidebar.markdown("---")
st.sidebar.info("""
**トレーニングデータ**: 1,483 samples  
**評価方法**: Repeated Nested CV  
**作成日**: 2025-10-29
""")
