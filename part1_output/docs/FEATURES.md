# SmartWatt feature definitions

Row = one target hour H (index = start of the hour). `y_kwh` = appliance energy during H. All features use only hours that finished before H began.

| Group | Name | Role | Count | Definition | Why |
|---|---|---|---|---|---|
| A | Energy history | primary | 14 | Completed-hour lags of appliance energy (1, 2, 3, 6, 12, 24, 168 h), rolling mean/std of the previous 3/6/24 hours and the last hour-to-hour change. | Captures persistence, recent volatility and daily/weekly routine. |
| B | Calendar | primary | 3 | Hour of the target hour as sin/cos, and a weekend flag, both taken from the household's LOCAL wall-clock time (file clock + 1 h after the 2016-03-27 DST change). | Hour 23 and hour 0 are neighbours, sin/cos keeps that. Known in advance, so no leakage. The local-clock correction is needed because routines move 1 h on the file clock at the DST date (see data_quality_log.json). |
| C | Local environment | primary | 21 | Previous completed hour mean of 9 temperatures and 9 humidities (T6/RH_6 are the on-site OUTDOOR sensors) plus 1-hour change of mean indoor T, mean indoor RH and T6. | Tests whether recorded conditions add predictive information. |
| D | Lighting usage proxy | primary | 2 | Lighting energy (Wh) of the previous completed hour and its 1-hour change. | May reflect household activity. It is a proxy, NOT a verified occupancy label. |
| E | Delayed airport weather (diagnostic) | diagnostic only | 6 | Chievres airport weather, mean of hour H-2 only. | The airport data was hourly and interpolated to 10 minutes, so the newest hour could contain future information. Delaying by 2 h is a safe sensitivity test, never part of the primary model. |
| RV | Random controls (diagnostic) | diagnostic only | 2 | rv1 / rv2 of the previous hour. | Artificial random numbers. A model that finds them important is fitting noise (negative control). |

## Column names per group

**A: Energy history**

`y_lag_1h`, `y_lag_2h`, `y_lag_3h`, `y_lag_6h`, `y_lag_12h`, `y_lag_24h`, `y_lag_168h`, `y_roll_mean_3h`, `y_roll_std_3h`, `y_roll_mean_6h`, `y_roll_std_6h`, `y_roll_mean_24h`, `y_roll_std_24h`, `y_change_1h`

**B: Calendar**

`hour_sin`, `hour_cos`, `is_weekend`

**C: Local environment**

`T1_lag_1h`, `T2_lag_1h`, `T3_lag_1h`, `T4_lag_1h`, `T5_lag_1h`, `T6_lag_1h`, `T7_lag_1h`, `T8_lag_1h`, `T9_lag_1h`, `RH_1_lag_1h`, `RH_2_lag_1h`, `RH_3_lag_1h`, `RH_4_lag_1h`, `RH_5_lag_1h`, `RH_6_lag_1h`, `RH_7_lag_1h`, `RH_8_lag_1h`, `RH_9_lag_1h`, `indoor_T_change_1h`, `indoor_RH_change_1h`, `T6_change_1h`

**D: Lighting usage proxy**

`lights_lag_1h_wh`, `lights_change_1h_wh`

**E: Delayed airport weather (diagnostic)**

`T_out_lag_2h`, `Press_mm_hg_lag_2h`, `RH_out_lag_2h`, `Windspeed_lag_2h`, `Visibility_lag_2h`, `Tdewpoint_lag_2h`

**RV: Random controls (diagnostic)**

`rv1_lag_1h`, `rv2_lag_1h`

## Excluded on purpose

- `rv1`, `rv2` as predictors (artificial random variables; only the diagnostic copies exist).
- Current-hour temperature, humidity, lighting (would be future information at forecast time).
- Airport weather in the primary set (interpolated hourly data can leak the future).
- The raw `Appliances` value of the target hour itself.
