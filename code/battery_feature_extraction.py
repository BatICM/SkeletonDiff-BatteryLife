"""
电池健康指标(HIs)提取与融合式特征选择模块
基于论文: "Battery health prediction using fusion-based feature selection and machine learning"

优化版V5：
- 正确识别原始特征名
- 限制进入包裹式筛选的特征数量
- 随机种子保证结果可复现
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# 定义所有原始特征名称（用于正确识别）
ORIGINAL_FEATURE_NAMES = [
    'IR', 'IR_diff', 'IR_cumsum', 'IR_ma10', 'IR_ewm',
    'QC', 'QC_diff',
    'QD', 'QD_diff', 'QD_decay',
    'CoulombicEfficiency', 'Q_diff',
    'Tavg', 'Tavg_diff', 'Tmin', 'Tmax', 'Trange', 'T_amplitude', 'T_deviation',
    'chargetime', 'chargetime_diff', 'chargetime_cumsum', 'chargetime_norm', 'chargetime_ma5',
    'charge_rate', 'energy_efficiency',
    'QD_decay_rate', 'QD_curvature', 'QD_slope',
    'IR_growth_rate', 'Composite_HI'
]


class BatteryFeatureExtractor:
    """
    电池健康指标提取器
    从电池循环数据中提取健康指标(HIs)
    """
    
    def __init__(self, battery_data):
        """
        初始化特征提取器
        
        Parameters:
        -----------
        battery_data : dict
            包含电池数据的字典
        """
        self.data = battery_data
        self.summary = battery_data.get('summary', {})
        self.cycles = battery_data.get('cycles', {})
        
    def extract_all_features(self):
        """提取所有健康指标"""
        features = {}
        
        # 1. 提取内阻相关特征
        features.update(self._extract_ir_features())
        
        # 2. 提取容量相关特征
        features.update(self._extract_capacity_features())
        
        # 3. 提取温度相关特征
        features.update(self._extract_temperature_features())
        
        # 4. 提取时间相关特征
        features.update(self._extract_time_features())
        
        # 5. 提取充放电效率特征
        features.update(self._extract_efficiency_features())
        
        # 6. 提取衰减趋势特征
        features.update(self._extract_degradation_features())
        
        # 转换为DataFrame
        df = pd.DataFrame(features)
        
        # 添加循环编号和SOH
        if 'cycle' in self.summary:
            df['cycle'] = self.summary['cycle']
        if 'SOH' in self.summary:
            df['SOH'] = self.summary['SOH']
            
        return df
    
    def _extract_ir_features(self):
        """提取内阻相关特征"""
        features = {}
        IR = self.summary.get('IR')
        
        if IR is not None:
            features['IR'] = IR
            ir_diff = np.diff(IR, prepend=IR[0])
            features['IR_diff'] = ir_diff
            features['IR_cumsum'] = np.cumsum(ir_diff)
            
            if len(IR) >= 10:
                features['IR_ma10'] = np.convolve(IR, np.ones(10)/10, mode='same')
            else:
                features['IR_ma10'] = IR
            
            alpha = 0.1
            ir_ewm = np.zeros_like(IR)
            ir_ewm[0] = IR[0]
            for i in range(1, len(IR)):
                ir_ewm[i] = alpha * IR[i] + (1 - alpha) * ir_ewm[i-1]
            features['IR_ewm'] = ir_ewm
            
        return features
    
    def _extract_capacity_features(self):
        """提取容量相关特征"""
        features = {}
        QC = self.summary.get('QC')
        QD = self.summary.get('QD')
        
        if QC is not None:
            features['QC'] = QC
            features['QC_diff'] = np.diff(QC, prepend=QC[0])
            
        if QD is not None:
            features['QD'] = QD
            features['QD_diff'] = np.diff(QD, prepend=QD[0])
            
            if len(QD) > 0 and QD[0] != 0:
                features['QD_decay'] = (QD[0] - QD) / QD[0]
            else:
                features['QD_decay'] = np.zeros_like(QD)
        
        if QC is not None and QD is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ce = np.where(QC != 0, QD / QC, 0)
            features['CoulombicEfficiency'] = np.clip(ce, 0, 1.5)
            features['Q_diff'] = QC - QD
            
        return features
    
    def _extract_temperature_features(self):
        """提取温度相关特征"""
        features = {}
        Tavg = self.summary.get('Tavg')
        Tmin = self.summary.get('Tmin')
        Tmax = self.summary.get('Tmax')
        
        if Tavg is not None:
            features['Tavg'] = Tavg
            features['Tavg_diff'] = np.diff(Tavg, prepend=Tavg[0])
            
        if Tmin is not None:
            features['Tmin'] = Tmin
            
        if Tmax is not None:
            features['Tmax'] = Tmax
            
        if Tmax is not None and Tmin is not None:
            features['Trange'] = Tmax - Tmin
            features['T_amplitude'] = (Tmax - Tmin) / 2
            
        if Tavg is not None and Tmax is not None:
            features['T_deviation'] = Tmax - Tavg
            
        return features
    
    def _extract_time_features(self):
        """提取时间相关特征"""
        features = {}
        chargetime = self.summary.get('chargetime')
        
        if chargetime is not None:
            features['chargetime'] = chargetime
            features['chargetime_diff'] = np.diff(chargetime, prepend=chargetime[0])
            features['chargetime_cumsum'] = np.cumsum(features['chargetime_diff'])
            
            if chargetime[0] != 0:
                features['chargetime_norm'] = chargetime / chargetime[0]
            else:
                features['chargetime_norm'] = np.ones_like(chargetime)
            
            if len(chargetime) >= 5:
                features['chargetime_ma5'] = np.convolve(chargetime, np.ones(5)/5, mode='same')
            else:
                features['chargetime_ma5'] = chargetime
                
        return features
    
    def _extract_efficiency_features(self):
        """提取效率相关特征"""
        features = {}
        QC = self.summary.get('QC')
        QD = self.summary.get('QD')
        chargetime = self.summary.get('chargetime')
        
        if QC is not None and chargetime is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                charge_rate = np.where(chargetime != 0, QC / chargetime, 0)
            features['charge_rate'] = charge_rate
            
        if QD is not None and chargetime is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                energy_eff = np.where(chargetime != 0, QD / chargetime, 0)
            features['energy_efficiency'] = energy_eff
            
        return features
    
    def _extract_degradation_features(self):
        """提取衰减趋势特征"""
        features = {}
        QD = self.summary.get('QD')
        IR = self.summary.get('IR')
        
        if QD is not None and len(QD) > 10:
            window = 10
            decay_rate = np.zeros_like(QD)
            for i in range(window, len(QD)):
                if QD[i-window] != 0:
                    decay_rate[i] = (QD[i-window] - QD[i]) / QD[i-window] / window
            features['QD_decay_rate'] = decay_rate
            
            d1 = np.gradient(QD)
            d2 = np.gradient(d1)
            features['QD_curvature'] = d2
            features['QD_slope'] = d1
            
        if IR is not None and len(IR) > 10:
            window = 10
            ir_growth = np.zeros_like(IR)
            for i in range(window, len(IR)):
                if IR[i-window] != 0:
                    ir_growth[i] = (IR[i] - IR[i-window]) / IR[i-window] / window
            features['IR_growth_rate'] = ir_growth
            
        if QD is not None and IR is not None:
            QD_range = np.max(QD) - np.min(QD)
            IR_range = np.max(IR) - np.min(IR)
            
            QD_norm = (QD - np.min(QD)) / (QD_range + 1e-10)
            IR_norm = (IR - np.min(IR)) / (IR_range + 1e-10)
            features['Composite_HI'] = 0.7 * (1 - QD_norm) + 0.3 * IR_norm
            
        return features


def get_original_feature_name(stat_feature_name):
    """
    从统计量特征名提取原始特征名
    
    例如：
    - 'IR_mean' -> 'IR'
    - 'energy_efficiency_max' -> 'energy_efficiency'
    - 'T_deviation_slope' -> 'T_deviation'
    """
    # 按长度降序排序，优先匹配长名称
    for orig_name in sorted(ORIGINAL_FEATURE_NAMES, key=len, reverse=True):
        if stat_feature_name.startswith(orig_name + '_') or stat_feature_name == orig_name:
            return orig_name
    
    # 如果都不匹配，返回第一个下划线前的部分
    parts = stat_feature_name.split('_')
    if len(parts) >= 2:
        return parts[0]
    return stat_feature_name


class FusionFeatureSelector:
    """
    融合式特征选择器（优化版V5）
    结合过滤式(Filter)和包裹式(Wrapper)方法
    
    特点：
    1. 正确识别原始特征名
    2. 限制进入包裹式筛选的特征数量
    3. 随机种子保证可复现
    """
    
    ESTIMATOR_RIDGE = 'ridge'
    ESTIMATOR_RF = 'rf'
    ESTIMATOR_MLP = 'mlp'
    ESTIMATOR_GBM = 'gradient_boosting'
    
    def __init__(self, min_features=3, max_iterations=20, 
                 max_samples=5000, random_state=42, estimator='ridge'):
        """
        Parameters:
        -----------
        min_features : int
            最少保留的特征数量
        max_iterations : int
            最大迭代次数
        max_samples : int
            大数据集采样上限
        random_state : int
            随机种子
        estimator : str
            评估器类型: 'ridge', 'rf', 'mlp', 'gradient_boosting'
        """
        self.min_features = min_features
        self.max_iterations = max_iterations
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimator_type = estimator
        self.selected_features = None
        self.correlation_scores = None
        self.selection_history = []
        self.selected_original_features = []
    
    def _get_estimator(self):
        """返回评估器实例"""
        if self.estimator_type == self.ESTIMATOR_RIDGE:
            return Ridge(alpha=1.0, random_state=self.random_state), "Ridge回归（线性，快速）"
        elif self.estimator_type == self.ESTIMATOR_RF:
            return RandomForestRegressor(
                n_estimators=50, max_depth=10, n_jobs=-1,
                random_state=self.random_state
            ), "随机森林（非线性，较慢）"
        elif self.estimator_type == self.ESTIMATOR_MLP:
            return MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=500,
                random_state=self.random_state, early_stopping=True
            ), "MLP神经网络（非线性，适中）"
        elif self.estimator_type == self.ESTIMATOR_GBM:
            return GradientBoostingRegressor(
                n_estimators=50, max_depth=5,
                random_state=self.random_state
            ), "梯度提升（非线性，较慢）"
        else:
            return Ridge(alpha=1.0, random_state=self.random_state), "Ridge回归（默认）"
    
    def fit(self, X, y, verbose=True):
        """执行融合式特征选择"""
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_array = X.values
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X_array = X
        
        y = np.array(y).ravel()
        
        if verbose:
            print("=" * 60)
            print("开始融合式特征选择")
            print("=" * 60)
            print(f"原始特征数量: {len(feature_names)}")
            print(f"样本数量: {len(y)}")
        
        # 计算相关系数
        correlations = []
        for i in range(X_array.shape[1]):
            corr, p_value = stats.pearsonr(X_array[:, i], y)
            if np.isnan(corr):
                corr = 0
            correlations.append(abs(corr))
        
        self.correlation_scores = dict(zip(feature_names, correlations))
        sorted_corr = sorted(self.correlation_scores.items(), key=lambda x: x[1], reverse=True)
        
        max_corr = sorted_corr[0][1] if sorted_corr else 0
        
        # 计算每个原始特征的最高相关系数（正确识别）
        original_feature_importance = {}
        for stat_feature, corr in self.correlation_scores.items():
            orig_feature = get_original_feature_name(stat_feature)
            
            if orig_feature not in original_feature_importance:
                original_feature_importance[orig_feature] = corr
            else:
                original_feature_importance[orig_feature] = max(original_feature_importance[orig_feature], corr)
        
        # 按原始特征重要性排序
        sorted_orig_features = sorted(original_feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True)
        
        if verbose:
            print(f"\n【原始特征重要性排序】（取各统计量最高相关系数）")
            for i, (name, corr) in enumerate(sorted_orig_features[:10]):
                print(f"  {i+1}. {name}: {corr:.4f}")
            if len(sorted_orig_features) > 10:
                print(f"  ... 共{len(sorted_orig_features)}个原始特征")
        
        # 根据相关系数水平决定筛选策略
        if max_corr < 0.5:
            max_filter_features = 20
            if verbose:
                print(f"\n【第一阶段】相关系数较低（最高{max_corr:.4f}<0.5）")
                print(f"取前{max_filter_features}个高相关统计量特征进入包裹式筛选")
        elif max_corr < 0.6:
            max_filter_features = 30
            if verbose:
                print(f"\n【第一阶段】相关系数中等（最高{max_corr:.4f}<0.6）")
                print(f"取前{max_filter_features}个高相关统计量特征进入包裹式筛选")
        else:
            max_filter_features = 50
            if verbose:
                print(f"\n【第一阶段】取前{max_filter_features}个高相关统计量特征进入包裹式筛选")
        
        # 取前N个高相关特征
        filter_selected = [f[0] for f in sorted_corr[:max_filter_features]]
        
        if verbose:
            print(f"进入包裹式筛选的特征数: {len(filter_selected)}")
        
        # 包裹式筛选
        if verbose:
            print(f"\n【第二阶段】包裹式特征筛选")
            _, estimator_name = self._get_estimator()
            print(f"评估器: {estimator_name}")
        
        filter_indices = [feature_names.index(name) for name in filter_selected]
        X_filtered = X_array[:, filter_indices]
        
        wrapper_selected_indices, history = self._sequential_backward_search(
            X_filtered, y, filter_selected, verbose
        )
        
        self.selection_history = history
        self.selected_features = [filter_selected[i] for i in wrapper_selected_indices]
        
        # 提取选中的原始特征（正确识别）
        self.selected_original_features = list(set([
            get_original_feature_name(f) for f in self.selected_features
        ]))
        
        if verbose:
            print(f"\n最终选择的特征数: {len(self.selected_features)}")
            print(f"\n选中的原始特征 ({len(self.selected_original_features)}个):")
            for feat in self.selected_original_features:
                max_corr_feat = original_feature_importance.get(feat, 0)
                print(f"  - {feat} (最高相关系数: {max_corr_feat:.4f})")
        
        return self
    
    def _sequential_backward_search(self, X, y, feature_names, verbose=True):
        """序列后向搜索"""
        n_features = X.shape[1]
        current_features = list(range(n_features))
        history = []
        
        estimator, _ = self._get_estimator()
        
        def evaluate_features(feature_indices):
            if len(feature_indices) == 0:
                return float('inf')
            
            X_subset = X[:, feature_indices]
            
            if len(y) > self.max_samples:
                rng = np.random.RandomState(self.random_state)
                sample_idx = rng.choice(len(y), self.max_samples, replace=False)
                X_subset = X_subset[sample_idx]
                y_sample = y[sample_idx]
            else:
                y_sample = y
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)
            
            try:
                scores = cross_val_score(
                    estimator, X_scaled, y_sample,
                    cv=3, scoring='neg_mean_squared_error'
                )
                return -np.mean(scores)
            except:
                return float('inf')
        
        if verbose:
            print("正在进行初始评估...")
        
        best_score = evaluate_features(current_features)
        history.append({
            'n_features': len(current_features),
            'score': best_score,
            'features': feature_names.copy()
        })
        
        if verbose:
            print(f"初始特征数: {len(current_features)}, MSE: {best_score:.6f}")
        
        iteration = 0
        max_iter = min(self.max_iterations, n_features - self.min_features)
        
        while len(current_features) > self.min_features and iteration < max_iter:
            iteration += 1
            worst_feature = None
            worst_score_increase = float('inf')
            new_score = best_score
            
            if verbose:
                print(f"\n迭代 {iteration}/{max_iter}: 评估 {len(current_features)} 个候选特征...")
            
            for idx, i in enumerate(current_features):
                temp_features = [f for f in current_features if f != i]
                score = evaluate_features(temp_features)
                score_increase = score - best_score
                
                if score_increase < worst_score_increase:
                    worst_score_increase = score_increase
                    worst_feature = i
                    new_score = score
                
                if verbose and (idx + 1) % 5 == 0:
                    print(f"  已评估 {idx + 1}/{len(current_features)} 个...")
            
            tolerance = 0.05 * best_score
            if worst_score_increase <= tolerance:
                current_features.remove(worst_feature)
                best_score = new_score
                removed_name = feature_names[worst_feature]
                
                history.append({
                    'n_features': len(current_features),
                    'score': best_score,
                    'removed_feature': removed_name
                })
                
                if verbose:
                    print(f"  ✓ 删除 '{removed_name}', 特征数={len(current_features)}, MSE={best_score:.6f}")
            else:
                if verbose:
                    print(f"  提前终止: 删除任何特征都会导致性能显著下降")
                break
        
        return current_features, history
    
    def transform(self, X):
        if self.selected_features is None:
            raise ValueError("请先调用fit()方法")
        
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features]
        else:
            indices = [self.selected_features.index(name) for name in self.selected_features]
            return X[:, indices]
    
    def fit_transform(self, X, y, verbose=True):
        self.fit(X, y, verbose)
        return self.transform(X)
