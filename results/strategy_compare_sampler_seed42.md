# sampler strategy comparison (seed=42)

## Overall metrics

| strategy | rmse_mean | rmse_std | mae_mean | mae_std | corr_mean | corr_std | eol_error_mean | eol_error_std | valid_eol_ratio | group_acc | path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weighted_random | 7.75076 | 2.30675 | 5.89672 | 2.32175 | 0.926325 | 0.0663464 | 417.955 | 297.997 | 1 | 1 | D:\study\毕设\mywork\DDPM_Battery\Output\code2_test_predictions_seed42_n10_stratA_weighted_r82.pkl |
| shuffle | 9.09694 | 2.52499 | 6.94356 | 2.83582 | 0.869928 | 0.11124 | 465.773 | 320.697 | 1 | 1 | D:\study\毕设\mywork\DDPM_Battery\Output\code2_test_predictions_seed42_n10_stratA_shuffle_r82.pkl |
| sequential | 8.72022 | 2.3785 | 6.62598 | 2.61054 | 0.898271 | 0.0836157 | 455.682 | 314.764 | 1 | 1 | D:\study\毕设\mywork\DDPM_Battery\Output\code2_test_predictions_seed42_n10_stratA_sequential_r82.pkl |

## Worst sample in each group (worst RMSE / worst EOL)

| strategy | group | worst_rmse_id | worst_rmse | worst_eol_id | worst_eol |
| --- | --- | --- | --- | --- | --- |
| weighted_random | 0 | b1c33 | 10.1885 | b3c21 | 323 |
| weighted_random | 1 | b3c14 | 7.77897 | b3c19 | 873 |
| weighted_random | 2 | b3c45 | 12.5261 | b3c45 | 686 |
| shuffle | 0 | b1c33 | 12.8317 | b3c21 | 369 |
| shuffle | 1 | b3c4 | 7.32405 | b3c34 | 955 |
| shuffle | 2 | b3c45 | 13.7692 | b3c45 | 719 |
| sequential | 0 | b1c33 | 11.7557 | b3c21 | 385 |
| sequential | 1 | b1c24 | 7.91552 | b3c19 | 965 |
| sequential | 2 | b3c45 | 13.3772 | b3c45 | 701 |
