from pathlib import Path
import re

import numpy as np
import pandas as pd

from data_generation import FESTIVAL_DATES, OUTPUT_DIR, get_meal_time


ITEM_PATTERN = re.compile(r"\d+\.([A-Z0-9_]+)\(([^|]+)\|\$([0-9.]+)\)", flags=re.IGNORECASE)


def get_mode(series):
    mode_values = series.dropna().mode()
    return mode_values.iloc[0] if not mode_values.empty else np.nan


def parse_cart(cart_str, catalog_lookup):
    if not isinstance(cart_str, str):
        return []
    parsed = []
    for item_id, label, price in ITEM_PATTERN.findall(cart_str.upper()):
        if item_id in catalog_lookup:
            info = catalog_lookup[item_id]
            parsed.append(
                {
                    "item_id": item_id,
                    "item_name": info["item_name"],
                    "item_type": info["item_type"],
                    "cuisine": info["cuisine"],
                    "addon_category": info["addon_category"],
                    "diet_type": info["diet_type"],
                    "price": float(price),
                }
            )
    return parsed


def clean_data(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    raw_path = output_dir / "raw_cart_orders.csv"
    catalog_path = output_dir / "item_catalog.csv"

    if not raw_path.exists() or not catalog_path.exists():
        raise FileNotFoundError("Run data_generation.py first to create raw_cart_orders.csv and item_catalog.csv.")

    df = pd.read_csv(raw_path)
    catalog_df = pd.read_csv(catalog_path)
    catalog_lookup = catalog_df.set_index("item_id").to_dict("index")
    initial_count = len(df)

    df = df.dropna(subset=["order_id", "cart_composition"])
    df = df[df["restaurant_id"] != "RES_NULL"].copy()

    cols_to_fix = [
        "discount_clicked",
        "app_opens_since_last_order",
        "time_spent_ordering_minutes",
        "dietary_pref",
        "order_frequency",
        "favorite_addon_category",
        "price_sensitivity",
    ]

    for col in cols_to_fix:
        user_modes = df.groupby("user_id")[col].transform(get_mode)
        df[col] = df[col].fillna(user_modes)
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(get_mode(df[col]))

    df["app_opens_since_last_order"] = df["app_opens_since_last_order"].astype(int)
    df["time_spent_ordering_minutes"] = df["time_spent_ordering_minutes"].astype(float)
    df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce").fillna(df["total_value"].median())
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp"] = df["timestamp"].fillna(df["timestamp"].dropna().median())

    df["order_date"] = df["timestamp"].dt.date.astype(str)
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_friday_night"] = ((df["day_of_week"] == 4) & (df["timestamp"].dt.hour >= 19)).astype(int)
    df["is_festival"] = df["order_date"].isin(FESTIVAL_DATES).astype(int)
    df["meal_time_slot"] = df["timestamp"].dt.hour.apply(get_meal_time)
    df["engagement_score"] = 0.4 * df["app_opens_since_last_order"] + 0.6 * df["time_spent_ordering_minutes"]
    df["is_discounted"] = (~df["discount_clicked"].astype(str).str.lower().isin(["none", "nan", ""])).astype(int)

    parsed = df["cart_composition"].apply(lambda cart: parse_cart(cart, catalog_lookup))
    df["addon_count"] = parsed.apply(lambda items: sum(item["item_type"] == "addon" for item in items))
    df["has_addon"] = (df["addon_count"] > 0).astype(int)
    df["cart_main_count"] = parsed.apply(lambda items: sum(item["item_type"] == "main" for item in items))
    df["cart_addon_value"] = parsed.apply(lambda items: round(sum(item["price"] for item in items if item["item_type"] == "addon"), 2))
    df["attach_rate_order_value_share"] = np.where(df["total_value"] > 0, df["cart_addon_value"] / df["total_value"], 0.0)

    order_item_rows = []
    for row in df.itertuples(index=False):
        for position, item in enumerate(parse_cart(row.cart_composition, catalog_lookup), start=1):
            order_item_rows.append(
                {
                    "order_id": row.order_id,
                    "user_id": row.user_id,
                    "timestamp": row.timestamp,
                    "restaurant_id": row.restaurant_id,
                    "restaurant_cuisine": row.restaurant_cuisine,
                    "meal_time_slot": row.meal_time_slot,
                    "position_in_cart": position,
                    **item,
                }
            )

    order_items = pd.DataFrame(order_item_rows)

    user_features = (
        df.groupby("user_id")
        .agg(
            total_orders=("order_id", "count"),
            avg_order_value=("total_value", "mean"),
            addon_attach_rate=("has_addon", "mean"),
            avg_addons_per_order=("addon_count", "mean"),
            avg_engagement_score=("engagement_score", "mean"),
            discount_affinity=("is_discounted", "mean"),
            weekend_order_share=("is_weekend", "mean"),
            favorite_addon_category=("favorite_addon_category", "first"),
            dietary_pref=("dietary_pref", "first"),
            order_frequency=("order_frequency", "first"),
            price_sensitivity=("price_sensitivity", "first"),
        )
        .reset_index()
    )

    meal_slot_aov = df.pivot_table(index="user_id", columns="meal_time_slot", values="total_value", aggfunc="mean").fillna(0)
    meal_slot_aov.columns = [f"{slot.lower().replace(' ', '_')}_avg_spend" for slot in meal_slot_aov.columns]
    user_features = user_features.merge(meal_slot_aov.reset_index(), on="user_id", how="left")

    df.to_csv(output_dir / "clean_cart_orders.csv", index=False)
    order_items.to_csv(output_dir / "order_items_long.csv", index=False)
    user_features.to_csv(output_dir / "user_feature_store.csv", index=False)

    print("Cleaning complete.")
    print(f"Rows removed: {initial_count - len(df):,}")
    print(f"Cleaned orders: {len(df):,}")
    print(f"Item-level rows: {len(order_items):,}")
    print(f"Users in feature store: {len(user_features):,}")
    return df, order_items, user_features


if __name__ == "__main__":
    clean_data()
