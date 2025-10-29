"""SE_p予測 - Streamlit Webアプリケーション（デバッグ版）"""
import streamlit as st
import os
from pathlib import Path

st.set_page_config(page_title="SE_p予測", page_icon="👁️", layout="wide")
st.markdown("# 👁️ SE_p予測システム - デバッグモード")

# デバッグ情報を表示
st.markdown("## 🔍 ファイル構造の確認")

# 現在のディレクトリ
current_dir = os.getcwd()
st.write(f"**現在のディレクトリ:** `{current_dir}`")

# ルートディレクトリのファイル一覧
st.write("**ルートディレクトリのファイル:**")
root_files = os.listdir('.')
for f in sorted(root_files):
    file_path = Path(f)
    if file_path.is_dir():
        st.write(f"📁 {f}/")
    else:
        size = file_path.stat().st_size / 1024
        st.write(f"📄 {f} ({size:.1f} KB)")

# saved_models フォルダの確認
st.write("---")
st.write("**saved_models フォルダの確認:**")

if os.path.exists('saved_models'):
    st.success("✅ saved_models フォルダは存在します")
    
    saved_models_files = os.listdir('saved_models')
    st.write(f"**ファイル数:** {len(saved_models_files)}")
    
    if saved_models_files:
        st.write("**含まれるファイル:**")
        for f in sorted(saved_models_files):
            file_path = Path('saved_models') / f
            if file_path.is_file():
                size = file_path.stat().st_size / 1024
                st.write(f"  📄 {f} ({size:.1f} KB)")
    else:
        st.error("❌ saved_models フォルダは空です！")
else:
    st.error("❌ saved_models フォルダが見つかりません！")

# metadata.json の確認
st.write("---")
st.write("**metadata.json の確認:**")

metadata_path = Path('saved_models/metadata.json')
if metadata_path.exists():
    st.success(f"✅ metadata.json が見つかりました: {metadata_path}")
    
    # 内容を読み込んで表示
    import json
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        st.write("**メタデータの内容:**")
        st.json(metadata)
    except Exception as e:
        st.error(f"読み込みエラー: {e}")
else:
    st.error(f"❌ metadata.json が見つかりません: {metadata_path}")
    
    # 代替パスを試す
    st.write("**代替パスを確認中...**")
    alt_paths = [
        'metadata.json',
        './saved_models/metadata.json',
        '../saved_models/metadata.json',
    ]
    for alt_path in alt_paths:
        if Path(alt_path).exists():
            st.success(f"✅ 見つかりました: {alt_path}")
        else:
            st.write(f"❌ {alt_path}")

st.write("---")
st.info("このデバッグ情報をコピーして教えてください！")
