"""SE_p予測 - Streamlit Webアプリケーション"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

# ページ設定
st.set_page_config(page_title="SE_p予測", page_icon="👁️", layout="wide")

st.markdown("# 👁️ SE予測システム")
st.markdown("眼科検査データから**調節麻痺後の球面等価屈折度**を予測します。")

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
    ["単一予測", "ファイル一括予測"],
    help="単一の症例または複数症例のExcel/CSVファイルを選択"
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
        age = st.number_input("年齢 (age)", min_value=3, max_value=18, value=9)
        k_avg = st.number_input("K (角膜曲率)", min_value=7.0, max_value=8.7, value=7.4, step=0.1, format="%.2f")
    
    with col2:
        gender = st.selectbox("性別 (sex)", [0, 1], format_func=lambda x: "男性" if x == 0 else "女性")
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
                    display_input = {
                        'age': age,
                        'sex': gender,
                        'K': k_avg,
                        'AL': al,
                        'LT': lt,
                        'ACD': acd
                    }
                    st.dataframe(pd.DataFrame([display_input]).T, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ 予測エラー: {e}")
                import traceback
                st.code(traceback.format_exc())

# ========================================
# ファイル一括予測モード
# ========================================
else:
    st.markdown("## 📤 Excel/CSVファイルをアップロード")
    
    # テンプレートのダウンロード
    st.markdown("### 📋 ファイルフォーマット")
    
    template_data = {
        'age': [9, 10, 8],
        'sex': [0, 1, 0],
        'K': [7.4, 7.6, 7.2],
        'AL': [24.0, 24.5, 23.8],
        'LT': [4.0, 4.2, 3.9],
        'ACD': [3.0, 3.1, 2.9]
    }
    template_df = pd.DataFrame(template_data)
    
    st.write("**必要な列:**")
    st.dataframe(template_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    # CSVテンプレートダウンロード
    with col1:
        csv_template = template_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSVテンプレートをダウンロード",
            data=csv_template,
            file_name="se_prediction_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Excelテンプレートダウンロード
    with col2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            template_df.to_excel(writer, index=False, sheet_name='データ')
        excel_template = excel_buffer.getvalue()
        
        st.download_button(
            label="📥 Excelテンプレートをダウンロード",
            data=excel_template,
            file_name="se_prediction_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "Excel または CSV ファイルを選択",
        type=['csv', 'xlsx', 'xls'],
        help="上記のフォーマットに従ったファイルをアップロードしてください"
    )
    
    if uploaded_file is not None:
        try:
            # ファイルの種類に応じて読み込み
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error("サポートされていないファイル形式です")
                st.stop()
            
            st.success(f"✅ ファイル読み込み成功: {len(df)} 件のデータ")
            
            # データプレビュー
            st.markdown("### 📊 データプレビュー")
            st.dataframe(df.head(10), use_container_width=True)
            
            # 列名のマッピング（大文字小文字を無視）
            column_mapping = {
                'age': '年齢',
                'sex': '性別',
                'K': 'K（AVG）',
                'AL': 'AL',
                'LT': 'LT',
                'ACD': 'ACD'
            }
            
            # 列名を小文字に変換してチェック
            df_columns_lower = {col.lower(): col for col in df.columns}
            
            # 必要な列のチェックと変換
            required_columns = ['age', 'sex', 'k', 'al', 'lt', 'acd']
            missing_columns = []
            renamed_df = df.copy()
            
            for req_col in required_columns:
                if req_col not in df_columns_lower:
                    missing_columns.append(req_col)
                else:
                    # 元の列名を取得
                    original_col = df_columns_lower[req_col]
                    # 内部用の列名に変換
                    internal_col = column_mapping[req_col]
                    renamed_df[internal_col] = df[original_col]
            
            if missing_columns:
                st.error(f"❌ 不足している列: {', '.join(missing_columns)}")
                st.info("必要な列: age, sex, K, AL, LT, ACD")
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
                        
                        for idx, row in renamed_df.iterrows():
                            # 入力データの準備（内部用の列名）
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
                        
                        # 結果を元のデータフレームに追加
                        result_df = df.copy()
                        result_df['SE_p_predicted'] = predictions
                        result_df['CI_95_lower'] = lower_bounds
                        result_df['CI_95_upper'] = upper_bounds
                        
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
                        
                        # 元のファイルの列名を使用
                        available_features = []
                        for req_col in required_columns:
                            if req_col in df_columns_lower:
                                available_features.append(df_columns_lower[req_col])
                        
                        feature_choice = st.selectbox(
                            "表示する特徴量を選択",
                            available_features
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
                        
                        col1, col2 = st.columns(2)
                        
                        # CSV形式
                        with col1:
                            csv_result = result_df.to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                label="📥 CSVでダウンロード",
                                data=csv_result,
                                file_name=f"se_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        # Excel形式
                        with col2:
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                result_df.to_excel(writer, index=False, sheet_name='予測結果')
                            excel_data = output.getvalue()
                            
                            st.download_button(
                                label="📥 Excelでダウンロード",
                                data=excel_data,
                                file_name=f"se_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        
                    except Exception as e:
                        st.error(f"❌ 予測エラー: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        except Exception as e:
            st.error(f"❌ ファイル読み込みエラー: {e}")
            st.write("ファイルの形式を確認してください。")
            import traceback
            st.code(traceback.format_exc())

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
    
    ### ファイル一括予測の使い方
    
    1. **テンプレートをダウンロード**（Excel または CSV）
    2. 必要な列: **age, sex, K, AL, LT, ACD**
    3. ファイルをアップロード
    4. 「一括予測を実行」をクリック
    5. 結果をダウンロード
    
    ### 列の説明
    
    - **age**: 年齢（3～18歳）
    - **sex**: 性別（0=男性, 1=女性）
    - **K**: 角膜曲率（7.0～8.7）
    - **AL**: 眼軸長（mm）
    - **LT**: 水晶体厚（mm）
    - **ACD**: 前房深度（mm）
    """)

st.sidebar.markdown("---")
st.sidebar.info("""
**トレーニングデータ**: 1,483 samples  
**評価方法**: Repeated Nested CV  
**作成日**: 2025-10-29
""")
