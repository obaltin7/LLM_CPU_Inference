**Table 2. Quantization comparison**

| Family | Ratio           | Metric        | Replicates                       | Mean  | 95 % CI (Student-t) |
|--------|-----------------|---------------|----------------------------------|-------|---------------------|
| 1B     | Q8_0 / Q4_K_M   | J/token (raw) | 1.542, 1.511, 1.510 (3 sessions) | 1.521 | [1.475; 1.566]      |
| 1B     | Q4_K_M / Q8_0   | Tokens/s      | 1.595, 1.573, 1.573 (3 sessions) | 1.580 | [1.549; 1.612]      |
| 3B     | Q8_0 / Q4_K_M   | J/token (raw) | 1.543, 1.568, 1.562 (3 sessions) | 1.558 | [1.525; 1.591]      |
| 3B     | Q4_K_M / Q8_0   | Tokens/s      | 1.621, 1.646, 1.642 (3 sessions) | 1.636 | [1.603; 1.670]      |
| 8B     | Q8_0 / Q4_K_M   | J/token (raw) | 1.601, 1.603, 1.596 (3 sessions) | 1.600 | [1.591; 1.609]      |
| 8B     | Q4_K_M / Q8_0   | Tokens/s      | 1.680, 1.683, 1.679 (3 sessions) | 1.681 | [1.675; 1.686]      |
| 8B †   | Q5_K_M / Q4_K_M | J/token (raw) | 10 rounds (night_01)             | 1.174 | [1.171; 1.177]      |
| 8B †   | Q4_K_M / Q5_K_M | Tokens/s      | 10 rounds (night_01)             | 1.143 | [1.140; 1.146]      |
| 8B †   | Q6_K / Q4_K_M   | J/token (raw) | 10 rounds (night_01)             | 1.288 | [1.282; 1.294]      |
| 8B †   | Q4_K_M / Q6_K   | Tokens/s      | 10 rounds (night_01)             | 1.308 | [1.303; 1.313]      |
| 8B †   | Q8_0 / Q4_K_M   | J/token (raw) | 10 rounds (night_01)             | 1.594 | [1.591; 1.597]      |
| 8B †   | Q4_K_M / Q8_0   | Tokens/s      | 10 rounds (night_01)             | 1.677 | [1.673; 1.681]      |

*† Night run (night_01): one session, 8 threads, 10 rounds; the ratio is paired within each round, the CI is a Student-t interval over rounds (df = 9). In the same night the Q8_0 / Q4_K_M ratio matches the campaign (1.600).*
