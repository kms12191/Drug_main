from pathlib import Path
import io

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="\uc11c\uc6b8\uc2dc \uc57d \uc218\uc694\uc608\uce21 \ubc0f \uc7ac\uace0\uad00\ub9ac", layout="wide")

OUTPUT_DIR = Path("/content/outputs")
FORECAST_PATH = OUTPUT_DIR / "random_forest_2026_demand_forecast.csv"
METRICS_PATH = OUTPUT_DIR / "random_forest_2025_validation_metrics.csv"
TEST_PATH = OUTPUT_DIR / "random_forest_2025_test_predictions.csv"
ALL_FORECAST_PATH = OUTPUT_DIR / "model_forecast_2026_all_methods.csv"
ALL_METRICS_PATH = OUTPUT_DIR / "model_comparison_2025_metrics.csv"
ALL_TEST_PATH = OUTPUT_DIR / "model_comparison_2025_predictions.csv"

COL_DATE = "\uc77c\uc2dc"
COL_DRUG = "\uc57d\ud488\uad6c\ubd84"
COL_DISTRICT = "\uc2dc\uad70\uad6c\uba85\uce6d"
COL_PRED = "\uc608\uce21\uc218\ub7c9"
MODEL_COL = "\ud559\uc2b5\ubaa8\ub378"
METHOD_COL = "\ud559\uc2b5\ubc29\ubc95"
COL_STOCK = "\ud604\uc7ac\uc7ac\uace0"
COL_START_STOCK = "\uc6d4\ucd08\uc7ac\uace0"
COL_BUY = "\uad8c\uc7a5\uad6c\ub9e4\uc218\ub7c9"
COL_AFTER_BUY = "\uad6c\ub9e4\ud6c4\uc7ac\uace0"
COL_END_STOCK = "\uc6d4\ub9d0\uc7ac\uace0"
COL_BUFFER = "\uc548\uc804\uc5ec\uc720\uc728"
ALL_LABEL = "\uc804\uccb4 \ubcf4\uae30"
TOTAL_LABEL = "\uc804\uccb4\ud569\uacc4"

st.title("\uc11c\uc6b8\uc2dc \uc57d \uc218\uc694\uc608\uce21 \ubc0f \uc7ac\uace0\uad00\ub9ac")
st.caption("\uc11c\uc6b8\uc2dc 25\uac1c \uad6c\uc758 \uc57d\ud488\uad6c\ubd84\ubcc4 \uc218\uc694\uc608\uce21 \uacb0\uacfc\uc640 \uc7ac\uace0 \uae30\ubc18 \uad8c\uc7a5 \uad6c\ub9e4\ub7c9\uc744 \ud655\uc778\ud569\ub2c8\ub2e4.")

forecast_source = ALL_FORECAST_PATH if ALL_FORECAST_PATH.exists() else FORECAST_PATH
metrics_source = ALL_METRICS_PATH if ALL_METRICS_PATH.exists() else METRICS_PATH
test_source = ALL_TEST_PATH if ALL_TEST_PATH.exists() else TEST_PATH

if not forecast_source.exists():
    st.error("\uc608\uce21 \uacb0\uacfc CSV\uac00 \uc5c6\uc2b5\ub2c8\ub2e4. Colab \ub178\ud2b8\ubd81\uc5d0\uc11c \ud559\uc2b5/\uc608\uce21/\uc800\uc7a5 \uc140\uc744 \uba3c\uc800 \uc2e4\ud589\ud558\uc138\uc694.")
    st.stop()

forecast = pd.read_csv(forecast_source)
if MODEL_COL not in forecast.columns:
    forecast[MODEL_COL] = "\ub79c\ub364\ud3ec\ub808\uc2a4\ud2b8"
if METHOD_COL not in forecast.columns:
    forecast[METHOD_COL] = "\ubc30\uae45"
forecast[COL_PRED] = pd.to_numeric(forecast[COL_PRED], errors="coerce").fillna(0)
forecast[COL_DATE] = forecast[COL_DATE].astype(str)

model_types = sorted(forecast[MODEL_COL].dropna().unique())
method_types = sorted(forecast[METHOD_COL].dropna().unique())
drug_types = sorted(forecast[COL_DRUG].dropna().unique())
districts = sorted(forecast[COL_DISTRICT].dropna().unique())


def read_csv_with_fallback(uploaded_file):
    raw = uploaded_file.getvalue()
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw))


def normalize_inventory(inventory):
    inventory = inventory.copy()
    inventory.columns = [str(col).strip() for col in inventory.columns]
    rename_map = {}
    for col in inventory.columns:
        lower = col.lower()
        if col == COL_DRUG or lower in ("drug_type", "drug", "type"):
            rename_map[col] = COL_DRUG
        elif col == COL_DISTRICT or lower in ("district", "region", "gu"):
            rename_map[col] = COL_DISTRICT
        elif col in (COL_STOCK, "\uc7ac\uace0\uc218\ub7c9", "\uc7ac\uace0") or lower in ("stock", "inventory", "current_stock", "stock_qty"):
            rename_map[col] = COL_STOCK
    inventory = inventory.rename(columns=rename_map)
    required = [COL_DRUG, COL_DISTRICT, COL_STOCK]
    missing = [col for col in required if col not in inventory.columns]
    if missing:
        raise ValueError("\uc7ac\uace0 CSV\uc5d0 \ud544\uc694\ud55c \uc5f4\uc774 \uc5c6\uc2b5\ub2c8\ub2e4: " + ", ".join(missing))
    inventory = inventory[required].copy()
    inventory[COL_DRUG] = inventory[COL_DRUG].astype(str).str.strip().replace({'ALL_TOTAL': TOTAL_LABEL, 'all_total': TOTAL_LABEL, 'total': TOTAL_LABEL})
    inventory[COL_DISTRICT] = inventory[COL_DISTRICT].astype(str).str.strip().replace({'ALL_TOTAL': TOTAL_LABEL, 'all_total': TOTAL_LABEL, 'total': TOTAL_LABEL})
    inventory[COL_STOCK] = pd.to_numeric(inventory[COL_STOCK], errors="coerce").fillna(0)
    return inventory.groupby([COL_DRUG, COL_DISTRICT], as_index=False)[COL_STOCK].sum()


def build_inventory_matched_forecast(forecast_df, inventory_df):
    base = forecast_df[[MODEL_COL, METHOD_COL, COL_DATE, COL_DRUG, COL_DISTRICT, COL_PRED]].copy()
    base[COL_PRED] = pd.to_numeric(base[COL_PRED], errors="coerce").fillna(0)
    matched_parts = []

    for _, inv_row in inventory_df.iterrows():
        drug_label = inv_row[COL_DRUG]
        district_label = inv_row[COL_DISTRICT]
        part = base.copy()
        if drug_label != TOTAL_LABEL:
            part = part[part[COL_DRUG] == drug_label]
        if district_label != TOTAL_LABEL:
            part = part[part[COL_DISTRICT] == district_label]
        if part.empty:
            continue

        grouped = (
            part.groupby([MODEL_COL, METHOD_COL, COL_DATE], as_index=False)[COL_PRED]
            .sum()
        )
        grouped[COL_DRUG] = drug_label
        grouped[COL_DISTRICT] = district_label
        grouped[COL_STOCK] = float(inv_row[COL_STOCK])
        matched_parts.append(grouped[[MODEL_COL, METHOD_COL, COL_DATE, COL_DRUG, COL_DISTRICT, COL_PRED, COL_STOCK]])

    if not matched_parts:
        return pd.DataFrame(columns=[MODEL_COL, METHOD_COL, COL_DATE, COL_DRUG, COL_DISTRICT, COL_PRED, COL_STOCK])
    return pd.concat(matched_parts, ignore_index=True)


def simulate_inventory(forecast_df, inventory_df, buffer_rate):
    base = build_inventory_matched_forecast(forecast_df, inventory_df)
    if base.empty:
        return pd.DataFrame(columns=[MODEL_COL, METHOD_COL, COL_DATE, COL_DRUG, COL_DISTRICT, COL_START_STOCK, COL_PRED, COL_BUY, COL_AFTER_BUY, COL_END_STOCK])
    base["date_sort"] = pd.to_datetime(base[COL_DATE], format="%Y-%m", errors="coerce")
    base = base.sort_values([MODEL_COL, METHOD_COL, COL_DRUG, COL_DISTRICT, "date_sort"])
    rows = []
    for (model_type, method_type, drug, district), part in base.groupby([MODEL_COL, METHOD_COL, COL_DRUG, COL_DISTRICT], sort=False):
        stock = float(part[COL_STOCK].iloc[0])
        for _, row in part.iterrows():
            predicted = float(row[COL_PRED])
            target = predicted * (1 + buffer_rate)
            buy = max(0.0, target - stock)
            after_buy = stock + buy
            end_stock = max(0.0, after_buy - predicted)
            rows.append({MODEL_COL: model_type, METHOD_COL: method_type, COL_DATE: row[COL_DATE], COL_DRUG: drug, COL_DISTRICT: district, COL_START_STOCK: round(stock), COL_PRED: round(predicted), COL_BUY: round(buy), COL_AFTER_BUY: round(after_buy), COL_END_STOCK: round(end_stock)})
            stock = end_stock
    return pd.DataFrame(rows)


with st.sidebar:
    st.header("\ud544\ud130")
    selected_model_option = st.selectbox("\ud559\uc2b5\ubaa8\ub378 \uc120\ud0dd", [ALL_LABEL] + model_types)
    selected_method_option = st.selectbox("\ud559\uc2b5\ubc29\ubc95 \uc120\ud0dd", [ALL_LABEL] + method_types)
    selected_heatmap_option = st.selectbox("\ud788\ud2b8\ub9f5 \uc57d\ud488\uad6c\ubd84", [ALL_LABEL] + drug_types)
    selected_drugs = st.multiselect("\uc57d\ud488\uad6c\ubd84 \uc120\ud0dd", drug_types, default=drug_types)
    selected_districts = st.multiselect("\uad6c \uc120\ud0dd", districts, default=districts)
    st.divider()
    st.header("\uac00\uc0c1 \uc7ac\uace0")
    inventory_file = st.file_uploader("\uc7ac\uace0 CSV \uc5c5\ub85c\ub4dc", type=["csv"])
    buffer_rate = st.slider("\uc608\uce21 \uc218\uc694\ubcf4\ub2e4 \ub354 \uc900\ube44\ud560 \ube44\uc728", 0, 50, 10, 5) / 100
    template = forecast[[COL_DRUG, COL_DISTRICT]].drop_duplicates().sort_values([COL_DRUG, COL_DISTRICT]).copy()
    template[COL_STOCK] = 0
    st.download_button("\uc7ac\uace0 \ud15c\ud50c\ub9bf \ub2e4\uc6b4\ub85c\ub4dc", template.to_csv(index=False, encoding="utf-8-sig"), "virtual_inventory_template.csv", "text/csv")

selected_models = model_types if selected_model_option == ALL_LABEL else [selected_model_option]
selected_methods = method_types if selected_method_option == ALL_LABEL else [selected_method_option]
heatmap_drugs = drug_types if selected_heatmap_option == ALL_LABEL else [selected_heatmap_option]
filtered = forecast[forecast[MODEL_COL].isin(selected_models) & forecast[METHOD_COL].isin(selected_methods) & forecast[COL_DRUG].isin(selected_drugs) & forecast[COL_DISTRICT].isin(selected_districts)].copy()

tab_validation, tab_forecast, tab_inventory, tab_heatmap = st.tabs(["\uac80\uc99d \ube44\uad50", "2026 \uc608\uce21 \ucd94\uc138", "\uc7ac\uace0 \uc2dc\ubbac\ub808\uc774\uc158", "\ud788\ud2b8\ub9f5"])

with tab_validation:
    if metrics_source.exists():
        st.subheader("2025\ub144 \uac80\uc99d \uc131\ub2a5")
        metrics = pd.read_csv(metrics_source)
        if MODEL_COL not in metrics.columns:
            metrics[MODEL_COL] = "\ub79c\ub364\ud3ec\ub808\uc2a4\ud2b8"
        if METHOD_COL not in metrics.columns:
            metrics[METHOD_COL] = "\ubc30\uae45"
        metrics = metrics[metrics[MODEL_COL].isin(selected_models) & metrics[METHOD_COL].isin(selected_methods)].copy()
        if 'drug_type' in metrics.columns:
            metrics = metrics[metrics['drug_type'].isin(selected_drugs)].copy()
        st.dataframe(metrics, use_container_width=True)
        if {'mape', MODEL_COL, METHOD_COL, 'drug_type'}.issubset(metrics.columns):
            metrics['label'] = metrics[MODEL_COL] + ' / ' + metrics[METHOD_COL]
            fig = px.bar(metrics, x='drug_type', y='mape', color='label', barmode='group', title='MAPE \ube44\uad50')
            st.plotly_chart(fig, use_container_width=True)
    if test_source.exists():
        test = pd.read_csv(test_source)
        if MODEL_COL not in test.columns:
            test[MODEL_COL] = "\ub79c\ub364\ud3ec\ub808\uc2a4\ud2b8"
        if METHOD_COL not in test.columns:
            test[METHOD_COL] = "\ubc30\uae45"
        test = test[test[MODEL_COL].isin(selected_models) & test[METHOD_COL].isin(selected_methods)].copy()
        if 'drug_type' in test.columns:
            test = test[test['drug_type'].isin(selected_drugs)].copy()
        if 'district' in test.columns:
            test = test[test['district'].isin(selected_districts)].copy()
        test['date'] = pd.to_datetime(test['date']).dt.strftime('%Y-%m')
        test_monthly = test.groupby(['date', MODEL_COL, METHOD_COL, 'drug_type'], as_index=False).agg(actual_qty=('qty', 'sum'), predicted_qty=('predicted_qty', 'sum'))
        test_long = test_monthly.melt(id_vars=['date', MODEL_COL, METHOD_COL, 'drug_type'], value_vars=['actual_qty', 'predicted_qty'], var_name='type', value_name='qty')
        test_long['type'] = test_long['type'].map({'actual_qty':'\uc2e4\uc81c \uc218\uc694', 'predicted_qty':'\uc608\uce21 \uc218\uc694'})
        for model_type in selected_models:
            for method_type in selected_methods:
                plot_df = test_long[(test_long[MODEL_COL] == model_type) & (test_long[METHOD_COL] == method_type)]
                if plot_df.empty:
                    continue
                fig = px.line(plot_df, x='date', y='qty', color='type', facet_row='drug_type', markers=True, title=f'{model_type} / {method_type}')
                fig.update_layout(height=600, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)

with tab_forecast:
    monthly = filtered.groupby([COL_DATE, MODEL_COL, METHOD_COL, COL_DRUG], as_index=False).agg(total_predicted_qty=(COL_PRED, 'sum'))
    for model_type in selected_models:
        for method_type in selected_methods:
            method_monthly = monthly[(monthly[MODEL_COL] == model_type) & (monthly[METHOD_COL] == method_type)]
            if method_monthly.empty:
                continue
            fig = px.line(method_monthly, x=COL_DATE, y='total_predicted_qty', color=COL_DRUG, markers=True, title=f'{model_type} / {method_type}')
            fig.update_layout(height=420, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

with tab_inventory:
    if inventory_file is None:
        st.info("\uc0ac\uc774\ub4dc\ubc14\uc5d0\uc11c \uc7ac\uace0 CSV\ub97c \uc5c5\ub85c\ub4dc\ud558\uba74 CSV\uc5d0 \uc788\ub294 \uc57d\ud488/\uad6c \ub610\ub294 \uc804\uccb4\ud569\uacc4 \ub2e8\uc704\ub85c\ub9cc \uad8c\uc7a5 \uad6c\ub9e4\ub7c9\uc774 \uacc4\uc0b0\ub429\ub2c8\ub2e4.")
    else:
        inventory = normalize_inventory(read_csv_with_fallback(inventory_file))
        simulation = simulate_inventory(filtered, inventory, buffer_rate)
        if simulation.empty:
            st.warning("\uc5c5\ub85c\ub4dc\ud55c \uc7ac\uace0 CSV\uc640 \ud604\uc7ac \ud544\ud130\uc5d0 \ub9de\ub294 \uc608\uce21 \ub370\uc774\ud130\uac00 \uc5c6\uc2b5\ub2c8\ub2e4. \uc57d\ud488\uad6c\ubd84/\uc2dc\uad70\uad6c\uba85\uce6d \uac12\uc744 \ud655\uc778\ud558\uc138\uc694.")
            st.stop()
        st.dataframe(simulation, use_container_width=True)
        buy_summary = simulation.groupby([COL_DATE, MODEL_COL, METHOD_COL, COL_DRUG], as_index=False)[COL_BUY].sum()
        buy_summary['label'] = buy_summary[MODEL_COL] + ' / ' + buy_summary[METHOD_COL]
        fig = px.bar(buy_summary, x=COL_DATE, y=COL_BUY, color='label', facet_row=COL_DRUG, barmode='group')
        fig.update_layout(height=720)
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("\uc7ac\uace0 \uc2dc\ubbac\ub808\uc774\uc158 \uacb0\uacfc \ub2e4\uc6b4\ub85c\ub4dc", simulation.to_csv(index=False, encoding='utf-8-sig'), 'inventory_purchase_simulation.csv', 'text/csv')

with tab_heatmap:
    for model_type in selected_models:
        for method_type in selected_methods:
            method_df = forecast[(forecast[MODEL_COL] == model_type) & (forecast[METHOD_COL] == method_type)]
            for selected_drug in heatmap_drugs:
                heatmap_df = method_df[method_df[COL_DRUG] == selected_drug]
                if heatmap_df.empty:
                    continue
                pivot = heatmap_df.pivot_table(index=COL_DISTRICT, columns=COL_DATE, values=COL_PRED, aggfunc='sum')
                fig = px.imshow(pivot, aspect='auto', color_continuous_scale='YlOrRd', title=f'{model_type} / {method_type} / {selected_drug}')
                fig.update_layout(height=620)
                st.plotly_chart(fig, use_container_width=True)
