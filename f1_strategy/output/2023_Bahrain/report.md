# 2023 Bahrain — Strategy Analysis Report

## Top 10 Strategies

|   Rank |   Stops | Strategy                           |   Total (s) |   Δ to P1 (s) |
|--------|---------|------------------------------------|-------------|---------------|
|      1 |       2 | 17L Medium → 35L Medium → 5L Hard  |      5264.3 |           0   |
|      2 |       2 | 5L Soft → 17L Medium → 35L Medium  |      5264.9 |           0.6 |
|      3 |       2 | 15L Medium → 35L Medium → 7L Hard  |      5265   |           0.7 |
|      4 |       2 | 13L Medium → 35L Medium → 9L Hard  |      5265.9 |           1.6 |
|      5 |       2 | 7L Soft → 15L Medium → 35L Medium  |      5266.1 |           1.9 |
|      6 |       2 | 11L Medium → 35L Medium → 11L Hard |      5267   |           2.8 |
|      7 |       2 | 9L Soft → 13L Medium → 35L Medium  |      5267.7 |           3.4 |
|      8 |       2 | 35L Medium → 11L Hard → 11L Hard   |      5268.4 |           4.2 |
|      9 |       2 | 9L Medium → 35L Medium → 13L Hard  |      5268.6 |           4.4 |
|     10 |       2 | 35L Medium → 9L Hard → 13L Hard    |      5269   |           4.7 |

## Sensitivity Analysis

| Strategy                           |   Nominal (s) |   -15% Deg (s) |   +15% Deg (s) | Type Flip?   |
|------------------------------------|---------------|----------------|----------------|--------------|
| 17L Medium → 35L Medium → 5L Hard  |        5264.3 |         5268.1 |         5260.5 | no           |
| 5L Soft → 17L Medium → 35L Medium  |        5264.9 |         5268.6 |         5261.1 | no           |
| 15L Medium → 35L Medium → 7L Hard  |        5265   |         5268.7 |         5261.3 | no           |
| 13L Medium → 35L Medium → 9L Hard  |        5265.9 |         5269.4 |         5262.3 | no           |
| 7L Soft → 15L Medium → 35L Medium  |        5266.1 |         5269.7 |         5262.6 | no           |
| 11L Medium → 35L Medium → 11L Hard |        5267   |         5270.4 |         5263.6 | no           |
| 9L Soft → 13L Medium → 35L Medium  |        5267.7 |         5271   |         5264.4 | no           |
| 35L Medium → 11L Hard → 11L Hard   |        5268.4 |         5271.6 |         5265.2 | no           |
| 9L Medium → 35L Medium → 13L Hard  |        5268.6 |         5271.8 |         5265.5 | no           |
| 35L Medium → 9L Hard → 13L Hard    |        5269   |         5272.1 |         5265.9 | no           |

## Model vs. Reality

| Driver   | Actual Strategy                           |   Pred. Rank |   Actual Time (s) |   Pred. Time (s) |   Error (s) |   Δ Optimal (s) |   Traffic Loss/Lap (s) |
|----------|-------------------------------------------|--------------|-------------------|------------------|-------------|-----------------|------------------------|
| VER      | 14L Soft → 22L Soft → 21L Hard            |          408 |            5636.7 |           5389.1 |      -247.6 |           124.8 |                  -0.43 |
| PER      | 17L Soft → 17L Soft → 23L Hard            |          387 |            5648.7 |           5395   |      -253.7 |           122.8 |                   0.43 |
| HUL      | 11L Soft → 15L Hard → 14L Hard → 16L Soft |           45 |            5660.6 |           5352.9 |      -307.8 |            25.8 |                   1.72 |
| ALO      | 14L Soft → 20L Hard → 23L Hard            |          332 |            5675.4 |           5416.9 |      -258.5 |           116.8 |                   1.34 |
| ZHO      | 12L Soft → 20L Hard → 22L Soft → 2L Soft  |           97 |            5676.4 |           5409.9 |      -266.5 |            45   |                   5.19 |
| SAI      | 13L Soft → 18L Hard → 26L Hard            |          344 |            5684.8 |           5407.8 |      -277   |           118.1 |                   0    |
| HAM      | 12L Soft → 18L Hard → 27L Hard            |          354 |            5687.7 |           5421.7 |      -266   |           118.9 |                   1.48 |
| STR      | 15L Soft → 15L Hard → 27L Hard            |          355 |            5691.2 |           5447.8 |      -243.4 |           119.3 |                   1.16 |
| RUS      | 13L Soft → 18L Hard → 26L Hard            |          344 |            5692.6 |           5418.4 |      -274.2 |           118.1 |                   0.97 |
| BOT      | 11L Soft → 18L Hard → 28L Hard            |          359 |            5709.4 |           5483.1 |      -226.3 |           119.9 |                   0.22 |
| TSU      | 10L Soft → 16L Hard → 14L Hard → 17L Soft |          366 |            5727.6 |           5544.8 |      -182.8 |           120.8 |                   0    |

## Insights

For the 2023 Bahrain GP, the model's pre-race optimal strategy is a 2-stop: 17L Medium → 35L Medium → 5L Hard (predicted total: 5264.3 s). The optimal strategy is ROBUST to ±15% degradation variation, maintaining its advantage across the sensitivity range. 8 of 11 analyzed drivers matched the model's recommended stop-count (2-stop). Median model error: -258.5 s.
