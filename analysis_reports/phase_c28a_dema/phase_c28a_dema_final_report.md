# Phase C28-A DEMA-HT Final Report

C28-A is a frozen, validation-only attribution audit. It performs no fitting, threshold adjustment, model combination, or checkpoint selection.

- material-damage overlap across temporal variants: `0.5298341148`
- selected-structure shortcut safety: `True`

## Validation AUC By Variant

- V0_official: seed 0 AUC `0.9017655048`/sensitivity `0.8510638298`/inversions `217`/n `94`, seed 42 AUC `0.8714350385`/sensitivity `0.7234042553`/inversions `284`/n `94`, seed 3407 AUC `0.8736985061`/sensitivity `0.8085106383`/inversions `279`/n `94`; mean AUC `0.8822996831`; mean positive probability `0.6895132258`; aggregate C17 TP-to-FN `17`.
- V1_uniform: seed 0 AUC `0.8596650068`/sensitivity `0.7021276596`/inversions `310`/n `94`, seed 42 AUC `0.8574015392`/sensitivity `0.6595744681`/inversions `315`/n `94`, seed 3407 AUC `0.8537799909`/sensitivity `0.7021276596`/inversions `323`/n `94`; mean AUC `0.8569488456`; mean positive probability `0.6158205031`; aggregate C17 TP-to-FN `27`.
- V2_recency_only: seed 0 AUC `0.8755092802`/sensitivity `0.8085106383`/inversions `275`/n `94`, seed 42 AUC `0.8718877320`/sensitivity `0.7234042553`/inversions `283`/n `94`, seed 3407 AUC `0.8587596197`/sensitivity `0.7446808511`/inversions `312`/n `94`; mean AUC `0.8687188773`; mean positive probability `0.6516721523`; aggregate C17 TP-to-FN `18`.
- V3_content_only: seed 0 AUC `0.8755092802`/sensitivity `0.7659574468`/inversions `275`/n `94`, seed 42 AUC `0.8524219104`/sensitivity `0.6808510638`/inversions `326`/n `94`, seed 3407 AUC `0.8718877320`/sensitivity `0.7872340426`/inversions `283`/n `94`; mean AUC `0.8666063075`; mean positive probability `0.6620816528`; aggregate C17 TP-to-FN `24`.
- V4_latest_only: seed 0 AUC `0.8678134903`/sensitivity `0.8936170213`/inversions `292`/n `94`, seed 42 AUC `0.8691715708`/sensitivity `0.8510638298`/inversions `289`/n `94`, seed 3407 AUC `0.8623811679`/sensitivity `0.8510638298`/inversions `304`/n `94`; mean AUC `0.8664554097`; mean positive probability `0.7444867306`; aggregate C17 TP-to-FN `8`.
- V5_history_mean_only: seed 0 AUC `0.6795005203`/sensitivity `0.2580645161`/inversions `308`/n `62`, seed 42 AUC `0.6597294485`/sensitivity `0.1290322581`/inversions `327`/n `62`, seed 3407 AUC `0.6638917794`/sensitivity `0.1612903226`/inversions `323`/n `62`; mean AUC `0.6677072494`; mean positive probability `0.3241314245`; aggregate C17 TP-to-FN `59`.

## Attribution Guard

- Latest-only reduced aggregate C17 TP-to-FN from `17` to `8`, but mean validation AUC changed by `-0.0158442734` and mean inversion count changed by `+35.0000000000`.
- This positive-recall rescue therefore fails the fixed ranking non-worsening guard and cannot authorize a temporal design.

Primary attribution: `C28A_MIXED_OR_INCONCLUSIVE`
Normalization: `C28A_CONTENT_SCORER_REMAINS_COUNT_ASSOCIATED`
C28-B authorization: `C28B_NOT_AUTHORIZED`
`KEEP_DEMA_C17_STRICT_BEST`
`STOP_VTME_TEMPORAL_TUNING`

Current strict best: `DEMA_C17_POSITIVE_PRESERVATION`
