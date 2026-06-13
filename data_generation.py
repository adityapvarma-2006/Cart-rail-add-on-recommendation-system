from pathlib import Path
import random

import numpy as np
import pandas as pd


SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "cart_add_on_outputs"

CUISINES = ["North Indian", "Chinese", "Italian", "South Indian", "Continental", "Fast Food"]
CITY_ZONES = ["Mumbai_West", "Delhi_South", "Bangalore_East", "Hyd_Central", "Pune_Central"]
DISCOUNTS = ["SAVE20", "WELCOME50", "BOGO", "FREE_DELIV", "FESTIVE10", "None"]
FESTIVAL_DATES = {"2025-01-14", "2025-01-26", "2025-02-14", "2025-02-26", "2025-03-14"}


MAIN_DISHES = {
    "North Indian": ["Paneer Butter Masala", "Butter Chicken", "Dal Makhani", "Chole Kulche", "Veg Biryani", "Chicken Biryani"],
    "Chinese": ["Veg Hakka Noodles", "Chicken Fried Rice", "Chilli Paneer", "Chicken Manchurian", "Schezwan Noodles", "Dimsum Bowl"],
    "Italian": ["Margherita Pizza", "Farmhouse Pizza", "Penne Alfredo", "Chicken Lasagna", "Arrabbiata Pasta", "Pesto Pasta"],
    "South Indian": ["Masala Dosa", "Idli Sambar", "Ghee Roast Dosa", "Uttapam", "Curd Rice", "Chicken Chettinad"],
    "Continental": ["Grilled Sandwich", "Caesar Salad", "Roast Chicken Plate", "Veg Steak Bowl", "English Breakfast", "Herb Rice Bowl"],
    "Fast Food": ["Classic Burger", "Cheese Burger", "Peri Peri Wrap", "Chicken Nuggets", "Loaded Nachos", "Veggie Sub"],
}


ADD_ON_SPECS = [
    ("Masala Chaas", "Beverage", ["North Indian", "South Indian"], 2.2, "Veg"),
    ("Coke", "Beverage", CUISINES, 1.8, "Veg"),
    ("Fresh Lime Soda", "Beverage", CUISINES, 2.0, "Veg"),
    ("Cold Coffee", "Beverage", ["Italian", "Continental", "Fast Food"], 3.2, "Veg"),
    ("Iced Tea", "Beverage", ["Chinese", "Italian", "Continental", "Fast Food"], 2.6, "Veg"),
    ("Gulab Jamun", "Dessert", ["North Indian", "South Indian"], 2.4, "Veg"),
    ("Chocolate Brownie", "Dessert", ["Italian", "Continental", "Fast Food"], 3.5, "Veg"),
    ("Tiramisu Cup", "Dessert", ["Italian", "Continental"], 4.2, "Veg+Egg"),
    ("Ice Cream Cup", "Dessert", CUISINES, 2.8, "Veg"),
    ("Rasmalai", "Dessert", ["North Indian"], 3.1, "Veg"),
    ("Fries", "Side", ["Fast Food", "Continental"], 2.5, "Veg"),
    ("Garlic Bread", "Side", ["Italian", "Continental", "Fast Food"], 2.9, "Veg"),
    ("Papad", "Side", ["North Indian", "South Indian"], 1.2, "Veg"),
    ("Spring Roll", "Side", ["Chinese"], 3.1, "Veg"),
    ("Momo Chutney", "Condiment", ["Chinese"], 0.9, "Veg"),
    ("Mint Chutney", "Condiment", ["North Indian"], 0.7, "Veg"),
    ("Extra Sambar", "Side", ["South Indian"], 1.0, "Veg"),
    ("Cheese Dip", "Condiment", ["Italian", "Fast Food", "Continental"], 1.1, "Veg"),
    ("Chicken Wings", "Side", ["Fast Food", "Continental"], 4.5, "Non-Veg"),
    ("Egg Mayo Dip", "Condiment", ["Fast Food", "Continental"], 1.4, "Veg+Egg"),
]


def diet_allows(user_pref, item_diet):
    if user_pref == "Non-Veg":
        return True
    if user_pref == "Veg+Egg":
        return item_diet in {"Veg", "Veg+Egg"}
    return item_diet == "Veg"


def get_meal_time(hour):
    if 6 <= hour < 11:
        return "Breakfast"
    if 11 <= hour < 16:
        return "Lunch"
    if 16 <= hour < 19:
        return "Snacks"
    if 19 <= hour < 23:
        return "Dinner"
    return "Late Night"


def build_catalog():
    rows = []
    for cuisine, dishes in MAIN_DISHES.items():
        for index, dish in enumerate(dishes, start=1):
            diet = "Non-Veg" if "chicken" in dish.lower() else "Veg"
            rows.append(
                {
                    "item_id": f"MAIN_{cuisine[:3].upper()}_{index:03d}".replace(" ", ""),
                    "item_name": dish,
                    "item_type": "main",
                    "cuisine": cuisine,
                    "addon_category": "Main",
                    "base_price": round(np.random.uniform(6.0, 18.0), 2),
                    "diet_type": diet,
                }
            )

    for index, (name, category, cuisines, price, diet) in enumerate(ADD_ON_SPECS, start=1):
        rows.append(
            {
                "item_id": f"ADD_{category[:3].upper()}_{index:03d}",
                "item_name": name,
                "item_type": "addon",
                "cuisine": " | ".join(cuisines),
                "addon_category": category,
                "base_price": price,
                "diet_type": diet,
            }
        )
    return pd.DataFrame(rows)


def compatible_main_ids(catalog_df, catalog_lookup, cuisine, user_pref):
    ids = catalog_df[(catalog_df["item_type"] == "main") & (catalog_df["cuisine"] == cuisine)]["item_id"].tolist()
    allowed = [item for item in ids if diet_allows(user_pref, catalog_lookup[item]["diet_type"])]
    return allowed if allowed else ids


def compatible_addon_ids(catalog_lookup, addon_ids, cuisine, user_pref):
    return [
        item_id
        for item_id in addon_ids
        if cuisine in catalog_lookup[item_id]["cuisine"] and diet_allows(user_pref, catalog_lookup[item_id]["diet_type"])
    ]


def sample_hour():
    slot = np.random.choice(["Breakfast", "Lunch", "Snacks", "Dinner", "Late Night"], p=[0.12, 0.29, 0.16, 0.34, 0.09])
    ranges = {
        "Breakfast": range(7, 11),
        "Lunch": range(12, 16),
        "Snacks": range(16, 19),
        "Dinner": range(19, 23),
        "Late Night": [0, 1, 2, 23],
    }
    return int(np.random.choice(list(ranges[slot])))


def choose_addon(catalog_lookup, candidates, user, cuisine, meal_slot, discount_clicked, is_festival):
    weights_by_slot = {
        "Breakfast": {"Beverage": 0.45, "Side": 0.30, "Dessert": 0.08, "Condiment": 0.17},
        "Lunch": {"Beverage": 0.34, "Side": 0.36, "Dessert": 0.18, "Condiment": 0.12},
        "Snacks": {"Beverage": 0.35, "Side": 0.24, "Dessert": 0.29, "Condiment": 0.12},
        "Dinner": {"Beverage": 0.31, "Side": 0.34, "Dessert": 0.25, "Condiment": 0.10},
        "Late Night": {"Beverage": 0.30, "Side": 0.28, "Dessert": 0.34, "Condiment": 0.08},
    }[meal_slot].copy()

    weights_by_slot[user["favorite_addon_category"]] += 0.16
    if discount_clicked != "None":
        weights_by_slot["Dessert"] += 0.08
        weights_by_slot["Beverage"] += 0.04
    if is_festival:
        weights_by_slot["Dessert"] += 0.16
    if cuisine in {"Chinese", "Fast Food"}:
        weights_by_slot["Beverage"] += 0.08

    weights = []
    for item_id in candidates:
        info = catalog_lookup[item_id]
        weight = weights_by_slot.get(info["addon_category"], 0.05)
        if user["price_sensitivity"] == "High" and info["base_price"] <= 2.6:
            weight += 0.10
        if user["price_sensitivity"] == "Low" and info["base_price"] >= 3.0:
            weight += 0.07
        weights.append(max(weight, 0.001))

    probs = np.array(weights, dtype=float)
    probs = probs / probs.sum()
    return str(np.random.choice(candidates, p=probs))


def generate_data(num_users=6000, num_restaurants=450, output_dir=OUTPUT_DIR):
    random.seed(SEED)
    np.random.seed(SEED)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog_df = build_catalog()
    catalog_lookup = catalog_df.set_index("item_id").to_dict("index")
    addon_ids = catalog_df.loc[catalog_df["item_type"] == "addon", "item_id"].tolist()

    restaurants = pd.DataFrame(
        {
            "restaurant_id": [f"RES_{i:03d}" for i in range(1, num_restaurants + 1)],
            "cuisine": np.random.choice(CUISINES, num_restaurants, p=[0.23, 0.17, 0.14, 0.18, 0.12, 0.16]),
            "city_zone": np.random.choice(CITY_ZONES, num_restaurants),
            "restaurant_rating": np.round(np.random.uniform(3.4, 4.8, num_restaurants), 2),
        }
    )

    users = pd.DataFrame(
        {
            "user_id": [f"USR_{i:05d}" for i in range(1, num_users + 1)],
            "dietary_pref": np.random.choice(["Veg", "Non-Veg", "Veg+Egg", "Jain"], num_users, p=[0.34, 0.46, 0.15, 0.05]),
            "order_frequency": np.random.choice(["Daily", "Weekly", "Bi-Weekly", "Monthly"], num_users, p=[0.09, 0.42, 0.30, 0.19]),
            "city_zone": np.random.choice(CITY_ZONES, num_users),
            "favorite_addon_category": np.random.choice(["Beverage", "Dessert", "Side", "Condiment"], num_users, p=[0.36, 0.23, 0.31, 0.10]),
            "price_sensitivity": np.random.choice(["Low", "Medium", "High"], num_users, p=[0.22, 0.55, 0.23]),
        }
    )

    rows = []
    order_number = 100000
    freq_map = {"Daily": 10, "Weekly": 4, "Bi-Weekly": 2, "Monthly": 1}

    for user in users.to_dict("records"):
        n_orders = max(1, np.random.poisson(freq_map[user["order_frequency"]]))
        for _ in range(n_orders):
            order_number += 1
            restaurant = restaurants.sample(1, random_state=np.random.randint(0, 10_000_000)).iloc[0]
            cuisine = restaurant["cuisine"]
            hour = sample_hour()
            timestamp = pd.Timestamp("2025-01-01") + pd.Timedelta(
                days=int(np.random.randint(0, 90)), hours=hour, minutes=int(np.random.randint(0, 60))
            )
            meal_slot = get_meal_time(timestamp.hour)
            is_festival = int(str(timestamp.date()) in FESTIVAL_DATES)

            main_candidates = compatible_main_ids(catalog_df, catalog_lookup, cuisine, user["dietary_pref"])
            main_count = int(np.random.choice([1, 2, 3], p=[0.64, 0.28, 0.08]))
            chosen_mains = list(np.random.choice(main_candidates, min(main_count, len(main_candidates)), replace=False))

            app_opens = int(np.random.randint(1, 25))
            time_spent = round(float(np.random.uniform(1.5, 21.0)), 2)
            discount_clicked = str(np.random.choice(DISCOUNTS, p=[0.11, 0.09, 0.05, 0.10, 0.06, 0.59]))

            addon_probability = min(0.55 + 0.015 * min(app_opens, 14) + (0.10 if discount_clicked != "None" else 0) + (0.08 if is_festival else 0), 0.88)
            chosen_addons = []
            addon_candidates = compatible_addon_ids(catalog_lookup, addon_ids, cuisine, user["dietary_pref"])
            if addon_candidates and np.random.random() < addon_probability:
                addon_count = int(np.random.choice([1, 2, 3], p=[0.74, 0.22, 0.04]))
                for _addon_idx in range(addon_count):
                    remaining = [item for item in addon_candidates if item not in chosen_addons]
                    if remaining:
                        chosen_addons.append(choose_addon(catalog_lookup, remaining, user, cuisine, meal_slot, discount_clicked, is_festival))

            cart_entries = []
            total_value = 0.0
            for position, item_id in enumerate(chosen_mains + chosen_addons, start=1):
                info = catalog_lookup[item_id]
                price = round(float(info["base_price"] * np.random.uniform(0.92, 1.12)), 2)
                label = info["cuisine"] if info["item_type"] == "main" else info["addon_category"]
                cart_entries.append(f"{position}.{item_id}({label}|${price})")
                total_value += price

            rows.append(
                {
                    "user_id": user["user_id"],
                    "order_id": f"ORD_{order_number}",
                    "restaurant_id": restaurant["restaurant_id"],
                    "restaurant_cuisine": cuisine,
                    "timestamp": timestamp,
                    "cart_composition": " >> ".join(cart_entries),
                    "total_items": len(cart_entries),
                    "total_value": round(total_value, 2),
                    "discount_clicked": discount_clicked,
                    "app_opens_since_last_order": app_opens,
                    "time_spent_ordering_minutes": time_spent,
                }
            )

    orders = pd.DataFrame(rows).merge(users.drop(columns=["city_zone"]), on="user_id", how="left")
    orders.loc[orders.sample(frac=0.04, random_state=SEED).index, "restaurant_id"] = "RES_NULL"
    orders.loc[orders.sample(frac=0.08, random_state=SEED + 1).index, "cart_composition"] = orders["cart_composition"].str.upper()
    orders.loc[orders.sample(frac=0.05, random_state=SEED + 2).index, ["app_opens_since_last_order", "time_spent_ordering_minutes"]] = np.nan
    orders["timestamp"] = orders["timestamp"].astype("object")
    orders.loc[orders.sample(frac=0.02, random_state=SEED + 3).index, "timestamp"] = "TIMESTAMP_ERROR"
    orders.loc[orders.sample(frac=0.01, random_state=SEED + 4).index, "discount_clicked"] = np.nan

    orders.to_csv(output_dir / "raw_cart_orders.csv", index=False)
    catalog_df.to_csv(output_dir / "item_catalog.csv", index=False)
    users.to_csv(output_dir / "users.csv", index=False)
    restaurants.to_csv(output_dir / "restaurants.csv", index=False)

    print(f"Generated raw orders: {len(orders):,}")
    print(f"Users: {num_users:,} | Restaurants: {num_restaurants:,} | Catalog items: {len(catalog_df):,}")
    print(f"Saved files under: {output_dir}")
    return orders, catalog_df, users, restaurants


if __name__ == "__main__":
    generate_data()
