from collections import Counter
from pathlib import Path
import math

import numpy as np
import pandas as pd

from cleaning import clean_data, parse_cart
from data_generation import OUTPUT_DIR, diet_allows, generate_data
from visualization import create_visualizations


SEED = 42


def build_sessions(orders, catalog_df):
    catalog_lookup = catalog_df.set_index("item_id").to_dict("index")
    rows = []
    for row in orders.sort_values("timestamp").itertuples(index=False):
        items = parse_cart(row.cart_composition, catalog_lookup)
        context = []
        for item in items:
            if item["item_type"] == "addon" and context:
                cart_total = sum(catalog_lookup[item_id]["base_price"] for item_id in context if item_id in catalog_lookup)
                rows.append(
                    {
                        "user_id": row.user_id,
                        "order_id": row.order_id,
                        "timestamp": row.timestamp,
                        "restaurant_id": row.restaurant_id,
                        "restaurant_cuisine": row.restaurant_cuisine,
                        "meal_time_slot": row.meal_time_slot,
                        "discount_clicked": row.discount_clicked,
                        "is_festival": row.is_festival,
                        "is_weekend": row.is_weekend,
                        "dietary_pref": row.dietary_pref,
                        "favorite_addon_category": row.favorite_addon_category,
                        "price_sensitivity": row.price_sensitivity,
                        "app_opens": row.app_opens_since_last_order,
                        "time_spent": row.time_spent_ordering_minutes,
                        "cart_context": tuple(context),
                        "cart_total_so_far": float(cart_total),
                        "target_item": item["item_id"],
                    }
                )
            context.append(item["item_id"])
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


class CartRailRecommender:
    def __init__(self, catalog_df):
        self.catalog = catalog_df.set_index("item_id").to_dict("index")
        self.addon_item_ids = catalog_df.loc[catalog_df["item_type"] == "addon", "item_id"].tolist()

    def fit(self, train_df):
        self.global_item = Counter(train_df["target_item"])
        self.cuisine_item = Counter(zip(train_df["restaurant_cuisine"], train_df["target_item"]))
        self.slot_item = Counter(zip(train_df["meal_time_slot"], train_df["target_item"]))
        self.user_item = Counter(zip(train_df["user_id"], train_df["target_item"]))
        self.restaurant_item = Counter(zip(train_df["restaurant_id"], train_df["target_item"]))
        self.user_category = Counter((row.user_id, self.catalog[row.target_item]["addon_category"]) for row in train_df.itertuples())
        self.cuisine_category = Counter((row.restaurant_cuisine, self.catalog[row.target_item]["addon_category"]) for row in train_df.itertuples())
        self.slot_category = Counter((row.meal_time_slot, self.catalog[row.target_item]["addon_category"]) for row in train_df.itertuples())
        self.cooccurrence = Counter()
        for row in train_df.itertuples():
            for context_item in row.cart_context:
                self.cooccurrence[(context_item, row.target_item)] += 1

        self.max_global = max(self.global_item.values(), default=1)
        self.max_cuisine_item = max(self.cuisine_item.values(), default=1)
        self.max_slot_item = max(self.slot_item.values(), default=1)
        self.max_user_item = max(self.user_item.values(), default=1)
        self.max_restaurant_item = max(self.restaurant_item.values(), default=1)
        self.max_cooccurrence = max(self.cooccurrence.values(), default=1)
        return self

    @staticmethod
    def _norm(counter, key, max_value):
        return math.log1p(counter.get(key, 0)) / math.log1p(max_value) if max_value > 0 else 0.0

    def score_candidate(self, session, candidate_item):
        info = self.catalog[candidate_item]
        category = info["addon_category"]
        price = float(info["base_price"])
        prior = self._norm(self.global_item, candidate_item, self.max_global)
        cuisine_item = self._norm(self.cuisine_item, (session["restaurant_cuisine"], candidate_item), self.max_cuisine_item)
        slot_item = self._norm(self.slot_item, (session["meal_time_slot"], candidate_item), self.max_slot_item)
        user_item = self._norm(self.user_item, (session["user_id"], candidate_item), self.max_user_item)
        restaurant_item = self._norm(self.restaurant_item, (session["restaurant_id"], candidate_item), self.max_restaurant_item)
        user_category = min(self.user_category.get((session["user_id"], category), 0) / 8.0, 1.0)
        cuisine_category = min(self.cuisine_category.get((session["restaurant_cuisine"], category), 0) / 200.0, 1.0)
        slot_category = min(self.slot_category.get((session["meal_time_slot"], category), 0) / 400.0, 1.0)
        favorite_match = 1.0 if session.get("favorite_addon_category") == category else 0.0

        cooc_count = sum(self.cooccurrence.get((context_item, candidate_item), 0) for context_item in session["cart_context"])
        cooc = math.log1p(cooc_count) / math.log1p(self.max_cooccurrence) if cooc_count else 0.0

        cart_total = max(float(session.get("cart_total_so_far", 0.0)), 1.0)
        price_fit = math.exp(-abs((price / cart_total) - 0.18) * 4.5)
        if session.get("price_sensitivity") == "High" and price <= 2.6:
            price_fit += 0.10
        if session.get("price_sensitivity") == "Low" and price >= 3.0:
            price_fit += 0.08

        context_bonus = 0.0
        if session.get("discount_clicked") != "None" and category in {"Dessert", "Beverage"}:
            context_bonus += 0.05
        if int(session.get("is_festival", 0)) == 1 and category == "Dessert":
            context_bonus += 0.08
        if session.get("meal_time_slot") == "Breakfast" and category == "Beverage":
            context_bonus += 0.04
        if session.get("meal_time_slot") in {"Lunch", "Dinner"} and category == "Side":
            context_bonus += 0.04

        return float(
            0.18 * prior
            + 0.16 * cuisine_item
            + 0.13 * slot_item
            + 0.12 * user_item
            + 0.12 * cooc
            + 0.07 * restaurant_item
            + 0.08 * user_category
            + 0.05 * cuisine_category
            + 0.04 * slot_category
            + 0.03 * favorite_match
            + 0.02 * min(price_fit, 1.0)
            + context_bonus
        )

    def candidate_items(self, session):
        context = set(session["cart_context"])
        user_pref = session.get("dietary_pref", "Non-Veg")
        candidates = []
        for item_id in self.addon_item_ids:
            if item_id in context:
                continue
            info = self.catalog[item_id]
            if diet_allows(user_pref, info["diet_type"]):
                candidates.append(item_id)
        return candidates if candidates else [item for item in self.addon_item_ids if item not in context]

    def recommend(self, session, top_n=8):
        rows = []
        for item_id in self.candidate_items(session):
            info = self.catalog[item_id]
            rows.append(
                {
                    "item_id": item_id,
                    "item_name": info["item_name"],
                    "addon_category": info["addon_category"],
                    "price": info["base_price"],
                    "score": self.score_candidate(session, item_id),
                }
            )
        recs = pd.DataFrame(rows).sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
        recs["rank"] = np.arange(1, len(recs) + 1)
        recs["score"] = recs["score"].round(4)
        return recs[["rank", "item_id", "item_name", "addon_category", "price", "score"]]


def binary_auc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def build_scored_pairs(model, source_sessions, negatives_per_positive=1, max_sessions=4500, seed=SEED):
    rng = np.random.default_rng(seed)
    if len(source_sessions) > max_sessions:
        source_sessions = source_sessions.sample(max_sessions, random_state=seed)
    rows = []
    for session in source_sessions.to_dict("records"):
        candidates = [item for item in model.candidate_items(session) if item != session["target_item"]]
        if not candidates:
            continue
        neg_count = min(negatives_per_positive, len(candidates))
        negatives = list(rng.choice(candidates, size=neg_count, replace=False))
        for item_id, label in [(session["target_item"], 1)] + [(neg, 0) for neg in negatives]:
            rows.append({"label": label, "score": model.score_candidate(session, item_id)})
    return pd.DataFrame(rows)


def best_threshold(scored_pairs):
    thresholds = np.quantile(scored_pairs["score"], np.linspace(0.05, 0.95, 80))
    best_accuracy, best_cutoff = 0.0, 0.5
    labels = scored_pairs["label"].to_numpy()
    scores = scored_pairs["score"].to_numpy()
    for threshold in thresholds:
        accuracy = ((scores >= threshold).astype(int) == labels).mean()
        if accuracy > best_accuracy:
            best_accuracy, best_cutoff = float(accuracy), float(threshold)
    return best_accuracy, best_cutoff


def evaluate_ranking(model, source_sessions, max_sessions=1500, seed=SEED):
    if len(source_sessions) > max_sessions:
        source_sessions = source_sessions.sample(max_sessions, random_state=seed)
    hit3 = hit8 = 0
    reciprocal_ranks = []
    ranks = []
    for session in source_sessions.to_dict("records"):
        candidates = model.candidate_items(session)
        if session["target_item"] not in candidates:
            candidates.append(session["target_item"])
        scored = sorted(((item, model.score_candidate(session, item)) for item in candidates), key=lambda x: x[1], reverse=True)
        ranked_items = [item for item, _ in scored]
        target_rank = ranked_items.index(session["target_item"]) + 1
        ranks.append(target_rank)
        reciprocal_ranks.append(1 / target_rank)
        hit3 += int(target_rank <= 3)
        hit8 += int(target_rank <= 8)
    n = len(ranks)
    return {
        "sessions_evaluated": n,
        "hit_rate_at_3": hit3 / n,
        "hit_rate_at_8": hit8 / n,
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "median_target_rank": float(np.median(ranks)),
    }


def rationale_for(row, session):
    category = row["addon_category"]
    if category == session.get("favorite_addon_category"):
        return "matches user's add-on affinity"
    if category == "Beverage" and session.get("meal_time_slot") in {"Breakfast", "Lunch", "Dinner"}:
        return "fills a common beverage gap"
    if category == "Dessert" and (session.get("discount_clicked") != "None" or int(session.get("is_festival", 0)) == 1):
        return "strong dessert context"
    if row["price"] <= 2.6:
        return "low-friction impulse add-on"
    return "high contextual co-purchase score"


def run_recommendation_pipeline(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    if not (output_dir / "raw_cart_orders.csv").exists():
        generate_data(output_dir=output_dir)
    orders, _, _ = clean_data(output_dir=output_dir)
    create_visualizations(output_dir=output_dir)

    catalog_df = pd.read_csv(output_dir / "item_catalog.csv")
    sessions = build_sessions(orders, catalog_df)
    split_index = int(len(sessions) * 0.80)
    train_sessions = sessions.iloc[:split_index].reset_index(drop=True)
    test_sessions = sessions.iloc[split_index:].reset_index(drop=True)

    model = CartRailRecommender(catalog_df).fit(train_sessions)
    train_pairs = build_scored_pairs(model, train_sessions, seed=SEED)
    test_pairs = build_scored_pairs(model, test_sessions, seed=SEED + 1)
    train_accuracy, threshold = best_threshold(train_pairs)
    test_predictions = (test_pairs["score"].to_numpy() >= threshold).astype(int)
    test_accuracy = float((test_predictions == test_pairs["label"].to_numpy()).mean())
    ranking_metrics = evaluate_ranking(model, test_sessions)

    metrics = {
        "raw_orders": len(pd.read_csv(output_dir / "raw_cart_orders.csv")),
        "clean_orders": len(orders),
        "recommendation_sessions": len(sessions),
        "train_sessions": len(train_sessions),
        "test_sessions": len(test_sessions),
        "train_start": str(train_sessions["timestamp"].min()),
        "train_end": str(train_sessions["timestamp"].max()),
        "test_start": str(test_sessions["timestamp"].min()),
        "test_end": str(test_sessions["timestamp"].max()),
        "train_auc": binary_auc(train_pairs["label"], train_pairs["score"]),
        "test_auc": binary_auc(test_pairs["label"], test_pairs["score"]),
        "train_pair_accuracy": train_accuracy,
        "test_pair_accuracy": test_accuracy,
        **ranking_metrics,
    }

    demo_session = None
    for candidate in test_sessions.sample(min(len(test_sessions), 300), random_state=SEED).to_dict("records"):
        demo_recs = model.recommend(candidate, top_n=8)
        if candidate["target_item"] in set(demo_recs["item_id"]):
            demo_session = candidate
            break
    if demo_session is None:
        demo_session = test_sessions.iloc[0].to_dict()

    recommendations = model.recommend(demo_session, top_n=8)
    recommendations["rationale"] = recommendations.apply(lambda row: rationale_for(row, demo_session), axis=1)
    recommendations.to_csv(output_dir / "sample_recommendations.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "model_summary_metrics.csv", index=False)

    print("\nTemporal split")
    print(f"Train sessions: {len(train_sessions):,} | {metrics['train_start']} to {metrics['train_end']}")
    print(f"Test sessions: {len(test_sessions):,} | {metrics['test_start']} to {metrics['test_end']}")
    print("\nEvaluation")
    print(f"Test AUC: {metrics['test_auc']:.2%}")
    print(f"Test pair accuracy: {metrics['test_pair_accuracy']:.2%}")
    print(f"Hit Rate @ 8: {metrics['hit_rate_at_8']:.2%}")
    print(f"MRR: {metrics['mean_reciprocal_rank']:.2%}")
    print("\nSample cart rail recommendations")
    print(recommendations.to_string(index=False))
    return model, metrics, recommendations


if __name__ == "__main__":
    run_recommendation_pipeline()
