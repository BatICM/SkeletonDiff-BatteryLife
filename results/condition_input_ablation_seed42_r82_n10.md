# 条件输入消融汇总 (seed=42, tag=r82, n10)

## 1) 整体指标

| mode | pred_path | rmse_mean | rmse_std | mae_mean | mae_std | corr_mean | corr_std | eol_error_mean | eol_error_std | valid_eol_ratio | n_valid_eol | n_true_eol_positive | false_cross | missed_cross | group_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | D:\study\毕设\mywork\DDPM_Battery\Output\code2_test_predictions_seed42_n10_ab_full_r82_fair.pkl | 2.60583 | 1.27613 | 1.63332 | 0.81286 | 0.971296 | 0.0277601 | 105.136 | 93.1628 | 1 | 22 | 22 | 0 | 0 | 1 |
| no_life | D:\study\毕设\mywork\DDPM_Battery\Output\code2_test_predictions_seed42_n10_ab_nolife_r82_fair.pkl | 2.07335 | 0.927263 | 1.31599 | 0.576991 | 0.972733 | 0.0296588 | 258.136 | 222.479 | 1 | 22 | 22 | 0 | 0 | 1 |
| life_only | D:\study\毕设\mywork\DDPM_Battery\Output\code2_test_predictions_seed42_n10_ab_lifeonly_r82_fair.pkl | 3.60663 | 1.83044 | 2.4527 | 1.38416 | 0.975533 | 0.0201788 | 121.455 | 96.3218 | 1 | 22 | 22 | 0 | 0 | 1 |

## 2) 分组指标

| mode | group | rmse_mean | mae_mean | corr_mean | eol_error_mean | valid_eol_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| full | 0 | 3.17699 | 1.9844 | 0.962102 | 105.818 | 1 |
| full | 1 | 2.11699 | 1.32585 | 0.980444 | 79.125 | 1 |
| full | 2 | 1.81511 | 1.16594 | 0.98061 | 172 | 1 |
| no_life | 0 | 2.45838 | 1.55599 | 0.97231 | 94.7273 | 1 |
| no_life | 1 | 1.50659 | 0.965987 | 0.97471 | 518 | 1 |
| no_life | 2 | 2.17295 | 1.36935 | 0.969011 | 164.333 | 1 |
| life_only | 0 | 3.99428 | 2.6424 | 0.975423 | 114.727 | 1 |
| life_only | 1 | 3.53285 | 2.60982 | 0.98459 | 113.875 | 1 |
| life_only | 2 | 2.38198 | 1.33814 | 0.951783 | 166.333 | 1 |

## 3) 每组最差样本（RMSE最差 & EOL误差最差）

| mode | group | worst_rmse_id | worst_rmse | worst_eol_id | worst_eol |
| --- | --- | --- | --- | --- | --- |
| full | 0 | b1c19 | 4.85496 | b3c21 | 296 |
| full | 1 | b1c5 | 3.2798 | b3c14 | 163 |
| full | 2 | b3c45 | 3.15989 | b3c17 | 387 |
| no_life | 0 | b2c20 | 3.84346 | b2c20 | 243 |
| no_life | 1 | b3c14 | 2.75594 | b3c14 | 705 |
| no_life | 2 | b3c45 | 3.14161 | b3c17 | 389 |
| life_only | 0 | b1c33 | 6.65375 | b3c21 | 305 |
| life_only | 1 | b3c19 | 6.10233 | b3c34 | 253 |
| life_only | 2 | b3c45 | 2.95871 | b3c17 | 326 |
