"""
SE_p予測モデル用ユーティリティクラス
GitHub/Streamlitで使用
"""

import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SEPredictor:
    """
    SE_p予測のための統一インターフェース
    
    使用例:
        predictor = SEPredictor(model_name='MLP')
        result = predictor.predict({
            '年齢': 45, 
            '性別': 0, 
            'K（AVG）': 44.5, 
            'AL': 23.5, 
            'LT': 2.8, 
            'ACD': 3.2
        })
    """
    
    def __init__(self, model_name='MLP', model_dir='saved_models'):
        """
        Parameters:
        -----------
        model_name : str
            使用するモデル名 ('MLP', 'ExtraTrees', 'CatBoost')
        model_dir : str
            モデルが保存されているディレクトリのパス
        """
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        
        # メタデータの読み込み
        metadata_path = self.model_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"メタデータが見つかりません: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # データ統計情報の読み込み
        stats_path = self.model_dir / 'data_stats.json'
        with open(stats_path, 'r', encoding='utf-8') as f:
            self.data_stats = json.load(f)
        
        # モデル情報の取得
        if model_name not in self.metadata['models']:
            available_models = list(self.metadata['models'].keys())
            raise ValueError(f"モデル '{model_name}' が見つかりません。利用可能: {available_models}")
        
        self.model_info = self.metadata['models'][model_name]
        self.feature_columns = self.metadata['feature_columns']
        
        # モデルの読み込み
        model_file = self.model_info['model_file']
        model_path = self.model_dir / model_file
        self.model = joblib.load(model_path)
        
        # スケーラーの読み込み（必要な場合）
        self.scaler = None
        if self.model_info['needs_scaling'] and self.model_info['scaler_file']:
            scaler_path = self.model_dir / self.model_info['scaler_file']
            self.scaler = joblib.load(scaler_path)
    
    def predict(self, input_data):
        """
        単一のデータポイントを予測
        
        Parameters:
        -----------
        input_data : dict
            予測用の入力データ
            例: {'年齢': 45, '性別': 0, 'K（AVG）': 44.5, 'AL': 23.5, 'LT': 2.8, 'ACD': 3.2}
        
        Returns:
        --------
        float : 予測値
        """
        # DataFrameに変換
        X = pd.DataFrame([input_data])[self.feature_columns]
        
        # 欠損値チェック
        if X.isnull().any().any():
            missing_cols = X.columns[X.isnull().any()].tolist()
            raise ValueError(f"入力データに欠損値が含まれています: {missing_cols}")
        
        # スケーリング（必要な場合）
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)
        else:
            prediction = self.model.predict(X)
        
        return float(prediction[0])
    
    def predict_with_details(self, input_data):
        """
        予測値と詳細情報を返す
        
        Parameters:
        -----------
        input_data : dict
            予測用の入力データ
        
        Returns:
        --------
        dict : 予測値と統計情報
        """
        # 入力検証
        validation = self.validate_input(input_data)
        
        # 予測
        prediction = self.predict(input_data)
        
        # 性能指標から信頼区間を推定
        perf = self.model_info['performance']
        rmse = perf['outer_rmse_mean']
        mae = perf['outer_mae_mean']
        
        # 95%信頼区間（正規分布を仮定）
        ci_lower = prediction - 1.96 * rmse
        ci_upper = prediction + 1.96 * rmse
        
        return {
            'prediction': round(prediction, 4),
            'confidence_interval_95': {
                'lower': round(ci_lower, 4),
                'upper': round(ci_upper, 4)
            },
            'expected_error': {
                'mae': round(mae, 4),
                'rmse': round(rmse, 4)
            },
            'model_performance': {
                'r2_mean': round(perf['outer_r2_mean'], 4),
                'r2_std': round(perf['outer_r2_std'], 4)
            },
            'validation': validation,
            'model_name': self.model_name
        }
    
    def validate_input(self, input_data):
        """
        入力データの妥当性をチェック
        
        Parameters:
        -----------
        input_data : dict
            チェックする入力データ
        
        Returns:
        --------
        dict : バリデーション結果
        """
        result = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # 必須フィールドのチェック
        for col in self.feature_columns:
            if col not in input_data:
                result['is_valid'] = False
                result['errors'].append(f"必須フィールド '{col}' がありません")
        
        if not result['is_valid']:
            return result
        
        # 値の範囲チェック
        for col in self.feature_columns:
            value = input_data[col]
            stats = self.data_stats['features'][col]
            min_val = stats['min']
            max_val = stats['max']
            
            if value < min_val * 0.5 or value > max_val * 1.5:
                result['warnings'].append(
                    f"{col}={value} はトレーニングデータの範囲外です "
                    f"(通常範囲: {min_val:.2f} - {max_val:.2f})"
                )
        
        return result
    
    def get_feature_importance(self):
        """
        特徴量重要度を取得（利用可能な場合）
        
        Returns:
        --------
        pd.DataFrame or None : 特徴量重要度
        """
        importance_file = f"{self.model_name.lower()}_feature_importance.csv"
        importance_path = self.model_dir / importance_file
        
        if importance_path.exists():
            return pd.read_csv(importance_path)
        else:
            return None
    
    def get_data_stats(self):
        """
        データ統計情報を取得
        
        Returns:
        --------
        dict : データ統計情報
        """
        return self.data_stats
    
    def get_model_info(self):
        """
        モデルの詳細情報を取得
        
        Returns:
        --------
        dict : モデル情報
        """
        return {
            'model_name': self.model_name,
            'feature_columns': self.feature_columns,
            'performance': self.model_info['performance'],
            'needs_scaling': self.model_info['needs_scaling'],
            'evaluation_method': self.metadata['evaluation']['method'],
            'total_evaluations': self.metadata['evaluation']['total_evaluations']
        }


class ModelEnsemble:
    """
    複数モデルのアンサンブル予測
    """
    
    def __init__(self, model_names=None, model_dir='saved_models'):
        """
        Parameters:
        -----------
        model_names : list, optional
            使用するモデル名のリスト。Noneの場合は全モデルを使用
        model_dir : str
            モデルが保存されているディレクトリのパス
        """
        # メタデータから利用可能なモデルを取得
        metadata_path = Path(model_dir) / 'metadata.json'
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        available_models = list(metadata['models'].keys())
        
        if model_names is None:
            model_names = available_models
        
        # 各モデルの予測器を初期化
        self.predictors = {}
        for name in model_names:
            if name in available_models:
                self.predictors[name] = SEPredictor(model_name=name, model_dir=model_dir)
        
        # 性能に基づく重みの計算（R²スコアを使用）
        r2_scores = {
            name: predictor.model_info['performance']['outer_r2_mean']
            for name, predictor in self.predictors.items()
        }
        total_r2 = sum(r2_scores.values())
        self.weights = {
            name: r2 / total_r2
            for name, r2 in r2_scores.items()
        }
    
    def predict(self, input_data):
        """
        加重平均によるアンサンブル予測
        
        Parameters:
        -----------
        input_data : dict
            予測用の入力データ
        
        Returns:
        --------
        float : アンサンブル予測値
        """
        predictions = {}
        for name, predictor in self.predictors.items():
            predictions[name] = predictor.predict(input_data)
        
        # 加重平均
        ensemble_prediction = sum(
            predictions[name] * self.weights[name]
            for name in predictions
        )
        
        return float(ensemble_prediction)
    
    def predict_with_details(self, input_data):
        """
        各モデルの予測値とアンサンブル結果を返す
        
        Parameters:
        -----------
        input_data : dict
            予測用の入力データ
        
        Returns:
        --------
        dict : 詳細な予測結果
        """
        individual_predictions = {}
        individual_details = {}
        
        for name, predictor in self.predictors.items():
            individual_predictions[name] = predictor.predict(input_data)
            individual_details[name] = predictor.predict_with_details(input_data)
        
        ensemble_prediction = sum(
            individual_predictions[name] * self.weights[name]
            for name in individual_predictions
        )
        
        # アンサンブルの不確実性を計算
        predictions_array = np.array(list(individual_predictions.values()))
        ensemble_std = np.std(predictions_array)
        
        return {
            'ensemble_prediction': round(ensemble_prediction, 4),
            'ensemble_std': round(ensemble_std, 4),
            'confidence_interval_95': {
                'lower': round(ensemble_prediction - 1.96 * ensemble_std, 4),
                'upper': round(ensemble_prediction + 1.96 * ensemble_std, 4)
            },
            'individual_predictions': individual_predictions,
            'individual_details': individual_details,
            'weights': self.weights,
            'models_used': list(self.predictors.keys())
        }


# テスト用コード
if __name__ == '__main__':
    print("=" * 80)
    print("SE予測モデル - テストスクリプト")
    print("=" * 80)
    
    # サンプル入力データを読み込み
    try:
        with open('saved_models/sample_inputs.json', 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        sample_input = samples['sample_1']
        
        print("\nテスト用入力データ:")
        for key, value in sample_input.items():
            print(f"  {key}: {value}")
        
        # 単一モデルのテスト
        print("\n" + "-" * 80)
        print("単一モデル（MLP）による予測:")
        print("-" * 80)
        
        predictor = SEPredictor(model_name='MLP')
        result = predictor.predict_with_details(sample_input)
        
        print(f"\n予測値: {result['prediction']}")
        print(f"95%信頼区間: [{result['confidence_interval_95']['lower']}, "
              f"{result['confidence_interval_95']['upper']}]")
        print(f"モデル性能 (R²): {result['model_performance']['r2_mean']} "
              f"± {result['model_performance']['r2_std']}")
        
        # アンサンブルのテスト
        print("\n" + "-" * 80)
        print("アンサンブルモデルによる予測:")
        print("-" * 80)
        
        ensemble = ModelEnsemble()
        ensemble_result = ensemble.predict_with_details(sample_input)
        
        print(f"\nアンサンブル予測値: {ensemble_result['ensemble_prediction']}")
        print(f"予測のばらつき (std): {ensemble_result['ensemble_std']}")
        print(f"\n各モデルの予測:")
        for model, pred in ensemble_result['individual_predictions'].items():
            weight = ensemble_result['weights'][model]
            print(f"  {model}: {pred:.4f} (重み: {weight:.3f})")
        
        print("\n✅ テスト完了")
        
    except FileNotFoundError as e:
        print(f"\n⚠ ファイルが見つかりません: {e}")
        print("先に colab_save_models.py を実行してモデルを保存してください")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
