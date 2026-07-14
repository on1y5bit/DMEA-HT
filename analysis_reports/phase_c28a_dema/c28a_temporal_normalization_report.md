# C28-A Temporal Normalization Report

The prior-only baseline is computed on the exact official temporal mask as `exp(log(2) * recency_t) / sum_j exp(log(2) * recency_j)`.

- seed 0: raw/excess/log-ratio count Spearman=`-0.8007323380`/`0.2217034014`/`0.1395811231`; ordered multi-visit stratum trend=`1`.
- seed 42: raw/excess/log-ratio count Spearman=`-0.9150080294`/`0.4465632942`/`0.1971469677`; ordered multi-visit stratum trend=`1`.
- seed 3407: raw/excess/log-ratio count Spearman=`-0.9096045440`/`0.2698532714`/`0.3069286713`; ordered multi-visit stratum trend=`1`.

Normalization decision: `C28A_CONTENT_SCORER_REMAINS_COUNT_ASSOCIATED`
