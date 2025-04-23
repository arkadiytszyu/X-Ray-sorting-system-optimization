import pandas as pd

def recommend_md(
    fraction_str,
    f_p2o5,
    f_mgo,
    feed,
    prioritet,
    model_output_cat,
    model_output_xgb,
    model_cp2o5_cat,
    model_cp2o5_xgb,
    model_cmgo_rf,
    model_cmgo_cat
):
    fraction_map = {'20-40': 0, '40-80': 1, '80-130': 2}
    if fraction_str not in fraction_map:
        raise ValueError(f"Неизвестная фракция: {fraction_str}")
    fraction = fraction_map[fraction_str]

    results = []
    for md in range(15, 56):
        row = pd.DataFrame([{
            'Fraction_code': fraction,
            'MD': md,
            'F_P2O5_%': f_p2o5,
            'MER': 100 * f_mgo / (f_p2o5 + 1e-5)
        }])

        # ✅ Прогноз C_output (70% CatBoost + 30% XGBoost)
        pred_out_cat = model_output_cat.predict(row)[0]
        pred_out_xgb = model_output_xgb.predict(row)[0]
        c_output = 0.7 * pred_out_cat + 0.3 * pred_out_xgb
        row['C_output_pred'] = c_output

        # ✅ Прогноз C_P2O5_% (80% CatBoost + 20% XGBoost)
        row_cp2o5 = row[['Fraction_code', 'MD', 'F_P2O5_%', 'MER', 'C_output_pred']]
        pred_p2o5_cat = model_cp2o5_cat.predict(row_cp2o5)[0]
        pred_p2o5_xgb = model_cp2o5_xgb.predict(row_cp2o5)[0]
        c_p2o5 = 0.8 * pred_p2o5_cat + 0.2 * pred_p2o5_xgb
        row['C_P2O5_%_pred'] = c_p2o5

        # ✅ Прогноз C_MgO_% (50% RF + 50% CatBoost)
        row_cmgo = row[['Fraction_code', 'MD', 'F_P2O5_%', 'MER', 'C_output_pred', 'C_P2O5_%_pred']]
        pred_mgo_rf = model_cmgo_rf.predict(row_cmgo)[0]
        pred_mgo_cat = model_cmgo_cat.predict(row_cmgo)[0]
        c_mgo = 0.5 * pred_mgo_rf + 0.5 * pred_mgo_cat

        # Расчёты
        concentrate = c_output * feed
        tails = feed - concentrate

        c_p2o5_t = c_p2o5 * concentrate / 100
        c_mgo_t = c_mgo * concentrate / 100
        f_p2o5_t = f_p2o5 * feed / 100
        f_mgo_t = f_mgo * feed / 100
        t_p2o5_t = f_p2o5_t - c_p2o5_t
        t_mgo_t = f_mgo_t - c_mgo_t
        t_p2o5 = t_p2o5_t / tails * 100 if tails != 0 else 0
        t_mgo = t_mgo_t / tails * 100 if tails != 0 else 0
        extraction = c_p2o5_t / f_p2o5_t * 100 if f_p2o5_t != 0 else 0

        results.append({
            'MD': md,
            'Fraction': fraction_str,
            'Feed': feed,
            'C_output': c_output,
            'C_P2O5_%': c_p2o5,
            'C_MgO_%': c_mgo,
            'Concentrate': concentrate,
            'Tails': tails,
            'Extraction': extraction,
            'T_P2O5_%': t_p2o5,
            'T_MgO_%': t_mgo,
            'C_P2O5_t': c_p2o5_t,
            'C_MgO_t': c_mgo_t,
            'F_P2O5_t': f_p2o5_t,
            'F_MgO_t': f_mgo_t,
            'T_P2O5_t': t_p2o5_t,
            'T_MgO_t': t_mgo_t
        })

    df_results = pd.DataFrame(results)

    # 🔽 Сортировка по приоритету
    if prioritet == 'P2O5':
        df_sorted = df_results.sort_values(by=['C_P2O5_%', 'C_MgO_%', 'C_output'], ascending=[False, True, False])
    elif prioritet == 'MgO':
        df_sorted = df_results.sort_values(by=['C_MgO_%', 'C_P2O5_%', 'C_output'], ascending=[True, False, False])
    elif prioritet == 'Output':
        df_sorted = df_results.sort_values(by=['C_output', 'C_P2O5_%', 'C_MgO_%'], ascending=[False, False, True])
    else:
        raise ValueError("Prioritet должен быть 'P2O5', 'MgO' или 'Output'")

    top5_df = df_sorted.head(5).reset_index(drop=True)
    full_df_sorted = df_results.sort_values(by='MD').reset_index(drop=True)

    return top5_df, full_df_sorted
