"""
电池寿命聚类模块
使用论文中的方法：基于 var_△Q100-10 和 peak_△Q100-10 进行K-Means聚类

参考论文: "Battery health estimation with degradation pattern recognition and transfer learning"
- var_△Q100-10: 第100循环与第10循环放电容量差的方差
- peak_△Q100-10: |△Q100-10(V)| 的峰值

这两个特征与电池寿命高度相关（ρ ≈ -0.93）
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.interpolate import interp1d

try:
    from plot_style import apply_thesis_plot_style
except Exception:
    def apply_thesis_plot_style(font_size: float = 10.5):
        plt.rcParams["font.family"] = [
            "Times New Roman",
            "SimSun",
            "STSong",
            "SimHei",
            "DejaVu Serif",
        ]
        plt.rcParams["font.serif"] = [
            "Times New Roman",
            "SimSun",
            "STSong",
            "SimHei",
            "DejaVu Serif",
        ]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["font.size"] = font_size
        plt.rcParams["axes.labelsize"] = font_size
        plt.rcParams["xtick.labelsize"] = font_size
        plt.rcParams["ytick.labelsize"] = font_size
        plt.rcParams["legend.fontsize"] = font_size


apply_thesis_plot_style(font_size=10.5)
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.default"] = "it"


class BatteryLifetimeClustering:
    """
    电池寿命聚类器
    按照论文方法，使用 var_△Q100-10 和 peak_△Q100-10 进行聚类
    """
    
    def __init__(self, n_clusters=3, random_state=42):
        """
        Parameters:
        -----------
        n_clusters : int
            聚类数量，默认为3（短/中/长寿命）
        random_state : int
            随机种子，保证结果可复现
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = None
        self.feature_names = None
        self.cluster_labels = None
        self.cluster_info = {}
        
    def extract_lifetime_features(self, bat_dict, cycle_1=10, cycle_2=100):
        """
        提取论文中的寿命相关特征 (终极修复版：精准剥离放电曲线，动态邻近搜索)
        """
        features_list =[]
        
        def get_discharge_curve(cycles_data, target_cycle):
            # 动态搜索：如果目标圈数据恰好损坏，自动搜索相邻的上下两圈 (如10圈坏了找9或11圈)
            for c in[target_cycle, target_cycle+1, target_cycle-1, target_cycle+2, target_cycle-2]:
                c_str = str(c)
                if c_str not in cycles_data:
                    continue
                    
                V = np.array(cycles_data[c_str].get('V',[]))
                Qd = np.array(cycles_data[c_str].get('Qd',[]))
                
                # 【核心修复】：绝不能混入充电阶段！只保留放电阶段 (Qd 大于 0.01)
                valid = (Qd > 0.01) & ~np.isnan(V) & ~np.isnan(Qd)
                V_valid = V[valid]
                Qd_valid = Qd[valid]
                
                if len(V_valid) < 10:
                    continue
                    
                # 按照电压排序（scipy插值要求x必须单调增加）
                idx = np.argsort(V_valid)
                V_sort = V_valid[idx]
                Qd_sort = Qd_valid[idx]
                
                # 去除因为传感器抖动产生的完全相同的电压坐标
                V_uniq, u_idx = np.unique(V_sort, return_index=True)
                Qd_uniq = Qd_sort[u_idx]
                
                # 确保提取出的曲线有效覆盖了核心电压区间 (至少覆盖 2.8V ~ 3.2V)
                if V_uniq.max() < 3.2 or V_uniq.min() > 2.8:
                    continue
                    
                return V_uniq, Qd_uniq
            return None, None

        for key, battery_data in bat_dict.items():
            cycles_data = battery_data.get('cycles', {})
            
            if not cycles_data or 'cycle_life' not in battery_data:
                continue
                
            # 获取第10圈和第100圈的平滑放电曲线
            V1, Q1 = get_discharge_curve(cycles_data, cycle_1)
            V2, Q2 = get_discharge_curve(cycles_data, cycle_2)
            
            if V1 is None or V2 is None:
                print(f"  ⚠ 警告: 电池 {key} 严重损坏，跳过")
                continue
                
            try:
                # 确定共同电压网格进行安全插值
                V_min = max(V1.min(), V2.min())
                V_max = min(V1.max(), V2.max())
                
                if V_min >= V_max - 0.1: # 几乎没有重叠区间
                    continue
                    
                V_grid = np.linspace(V_min, V_max, 100)
                
                f1 = interp1d(V1, Q1, kind='linear', fill_value='extrapolate')
                f2 = interp1d(V2, Q2, kind='linear', fill_value='extrapolate')
                
                # 严格按照论文公式计算特征: △Q(V)
                delta_Q = f2(V_grid) - f1(V_grid)
                
                if np.isnan(delta_Q).any():
                    continue
                    
                var_delta_Q = float(np.var(delta_Q))
                peak_delta_Q = float(np.max(np.abs(delta_Q)))
                
                cycle_life = int(battery_data['cycle_life'][0][0])
                
                features_list.append({
                    'battery_id': key,
                    'var_delta_Q': var_delta_Q,
                    'peak_delta_Q': peak_delta_Q,
                    'log_var_delta_Q': np.log10(max(var_delta_Q, 1e-10)),
                    'log_peak_delta_Q': np.log10(max(peak_delta_Q, 1e-10)),
                    'cycle_life': cycle_life
                })
            except Exception:
                continue
                
        features_df = pd.DataFrame(features_list)
        return features_df
    
    def fit(self, features_df, output_dir=None, verbose=True):
        """
        训练聚类模型
        
        Parameters:
        -----------
        features_df : DataFrame
            包含 var_delta_Q 和 peak_delta_Q 的数据框
        output_dir : str, optional
            可视化输出目录
        verbose : bool
            是否输出详细信息
        """
        # 使用论文中的特征（对数变换后）
        feature_cols = ['log_var_delta_Q', 'log_peak_delta_Q']
        self.feature_names = feature_cols
        
        X = features_df[feature_cols].values
        
        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # K-Means聚类
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        self.cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # 计算轮廓系数
        if len(np.unique(self.cluster_labels)) > 1:
            silhouette = silhouette_score(X_scaled, self.cluster_labels)
        else:
            silhouette = 0
        
        # 分析聚类结果
        features_df_copy = features_df.copy()
        features_df_copy['cluster'] = self.cluster_labels
        
        # 按平均寿命排序，确定短/中/长寿命标签
        cluster_lifetimes = {}
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            avg_lifetime = features_df_copy.loc[cluster_mask, 'cycle_life'].mean()
            cluster_lifetimes[cluster_id] = avg_lifetime
        
        # 排序：寿命短→长
        sorted_clusters = sorted(cluster_lifetimes.items(), key=lambda x: x[1])
        cluster_mapping = {old: new for new, (old, _) in enumerate(sorted_clusters)}
        
        # 重新映射标签
        self.cluster_labels = np.array([cluster_mapping[c] for c in self.cluster_labels])
        self.kmeans.labels_ = self.cluster_labels
        features_df_copy['cluster'] = self.cluster_labels
        
        # 存储每组的统计信息
        self.cluster_info = {
            'silhouette_score': silhouette,
            'cluster_mapping': cluster_mapping,
            'groups': {}
        }
        
        group_names = ['short', 'medium', 'long']
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            lifetimes = features_df_copy.loc[cluster_mask, 'cycle_life'].values
            battery_ids = features_df_copy.loc[cluster_mask, 'battery_id'].values
            
            self.cluster_info['groups'][cluster_id] = {
                'name': group_names[cluster_id],
                'count': int(np.sum(cluster_mask)),
                'min_lifetime': int(np.min(lifetimes)),
                'max_lifetime': int(np.max(lifetimes)),
                'mean_lifetime': float(np.mean(lifetimes)),
                'std_lifetime': float(np.std(lifetimes)),
                'battery_ids': battery_ids.tolist()
            }
        
        if verbose:
            print("=" * 60)
            print("K-Means聚类结果（论文方法）")
            print("=" * 60)
            print(f"轮廓系数: {silhouette:.4f}")
            print(f"聚类特征: var_△Q, peak_△Q (对数变换)")
            print()
            
            for cluster_id in range(self.n_clusters):
                info = self.cluster_info['groups'][cluster_id]
                print(f"【{info['name']}寿命组】(Cluster {cluster_id})")
                print(f"  电池数量: {info['count']}")
                print(f"  循环寿命范围: {info['min_lifetime']} - {info['max_lifetime']}")
                print(f"  平均循环寿命: {info['mean_lifetime']:.1f} ± {info['std_lifetime']:.1f}")
                print()
        
        # 可视化
        if output_dir:
            self._visualize_clustering(features_df_copy, output_dir)
        
        return self
    
    def predict(self, features_df):
        """
        预测新电池的寿命分组
        """
        X = features_df[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        raw_labels = self.kmeans.predict(X_scaled)
        
        cluster_mapping = self.cluster_info.get('cluster_mapping', {})
        if cluster_mapping:
            labels = np.array([cluster_mapping.get(c, c) for c in raw_labels])
        else:
            labels = raw_labels
        
        return labels
    
    def get_target_seq_length(self, cluster_id, cycle_resolution=10, n_downsamples=3, margin_ratio=0.0):
        """
        根据寿命组确定统一的目标寿命步数。

        当前 1D U-Net 有 3 次 stride=2 下采样，因此输入序列长度只需要能被 2^3=8 整除，
        不要求本身是 2 的 n 次方。这里 SOH 序列长度定义为 `target_length / cycle_resolution`，
        因此 `target_length` 需要是 `cycle_resolution * 2^n_downsamples` 的倍数。
        """
        group_info = self.cluster_info.get('groups', {}).get(cluster_id)
        if group_info is None:
            raise ValueError(f"未找到 cluster {cluster_id} 的分组信息，请先调用 fit()")

        round_to = cycle_resolution * (2 ** n_downsamples)
        max_lifetime = int(group_info['max_lifetime'])
        raw_target = max_lifetime * (1.0 + margin_ratio)
        target_length = int(np.ceil(raw_target / round_to) * round_to)

        return max(round_to, target_length)

    def _visualize_clustering(self, features_df, output_dir):
        """
        聚类结果可视化（在一张图上显示）
        
        - 所有数据点在同一坐标系
        - 不同组用不同颜色区分
        """
        os.makedirs(output_dir, exist_ok=True)
        
        colors = ['#CBDDEB', '#EAF5E2', '#FCF0D5']
        center_colors = ['#4683B4', '#67B3AD', '#F3AF44']
        group_names = ['短寿命组', '中寿命组', '长寿命组']
        
        # 画布宽度固定 15cm
        fig, ax = plt.subplots(figsize=(15.0 / 2.54, 10.0 / 2.54))
        
        # 绘制所有数据点：同组内按“距组中心远近”做浅→深渐变
        center_points = {}
        for cluster_id in range(self.n_clusters):
            mask = features_df['cluster'] == cluster_id
            n_points = np.sum(mask)
            if n_points == 0:
                continue

            x_vals = features_df.loc[mask, 'log_var_delta_Q'].to_numpy(dtype=float)
            y_vals = features_df.loc[mask, 'log_peak_delta_Q'].to_numpy(dtype=float)
            center_x = float(np.mean(x_vals))
            center_y = float(np.mean(y_vals))
            center_points[cluster_id] = (center_x, center_y)

            dist = np.sqrt((x_vals - center_x) ** 2 + (y_vals - center_y) ** 2)
            d_scale = float(np.percentile(dist, 95))
            if d_scale < 1e-8:
                d_scale = 1.0
            closeness = 1.0 - np.clip(dist / d_scale, 0.0, 1.0)

            light_rgb = np.array(mcolors.to_rgb(colors[cluster_id]), dtype=float)
            dark_rgb = np.array(mcolors.to_rgb(center_colors[cluster_id]), dtype=float)
            point_rgb = (
                light_rgb[None, :] * (1.0 - closeness[:, None])
                + dark_rgb[None, :] * closeness[:, None]
            )

            ax.scatter(
                x_vals,
                y_vals,
                c=point_rgb,
                alpha=0.95,
                s=52,
                edgecolors='#FFFFFF',
                linewidths=0.6,
            )
        
        # 标记聚类中心
        for cluster_id, (center_x, center_y) in center_points.items():
            ax.scatter(
                center_x,
                center_y,
                marker='o',
                s=52,
                c=[center_colors[cluster_id]],
                edgecolors='#000000',
                linewidths=1.1,
                zorder=5,
            )
        
        ax.set_xlabel(r'$\log_{10}(\mathit{var}(\Delta Q_{100-10}))$')
        ax.set_ylabel(r'$\log_{10}(\mathit{peak}(\Delta Q_{100-10}))$')

        legend_handles = [
            Line2D(
                [0], [0],
                marker='o', linestyle='',
                markerfacecolor=colors[0],
                markeredgecolor='#FFFFFF',
                markeredgewidth=0.6,
                markersize=8,
                label='短寿命组',
            ),
            Line2D(
                [0], [0],
                marker='o', linestyle='',
                markerfacecolor=colors[1],
                markeredgecolor='#FFFFFF',
                markeredgewidth=0.6,
                markersize=8,
                label='中寿命组',
            ),
            Line2D(
                [0], [0],
                marker='o', linestyle='',
                markerfacecolor=colors[2],
                markeredgecolor='#FFFFFF',
                markeredgewidth=0.6,
                markersize=8,
                label='长寿命组',
            ),
            Line2D(
                [0], [0],
                marker='o', linestyle='',
                markerfacecolor=center_colors[0],
                markeredgecolor='#000000',
                markeredgewidth=1.1,
                markersize=8,
                label='聚类中心',
            ),
        ]
        ax.legend(handles=legend_handles, loc='upper left', frameon=True)
        ax.grid(True, linestyle='--', alpha=0.35)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'clustering_results.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ 聚类可视化已保存: {save_path}")
    
    def save(self, filepath):
        """保存聚类器"""
        data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'cluster_info': self.cluster_info,
            'n_clusters': self.n_clusters,
            'random_state': self.random_state
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ 聚类器已保存: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """加载聚类器"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            n_clusters=data['n_clusters'],
            random_state=data['random_state']
        )
        instance.kmeans = data['kmeans']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance.cluster_info = data['cluster_info']
        
        return instance
