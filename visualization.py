from pathlib import Path

import pandas as pd

from data_generation import OUTPUT_DIR


def escape_svg(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bar_svg(series, title, width=860, height=360, color="#2F6F73"):
    series = series.dropna()
    labels = [str(x) for x in series.index]
    values = series.astype(float).values
    max_value = max(values.max(), 1.0)
    left, right, top, bottom = 80, 30, 46, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    gap = 12
    bar_w = max(12, (plot_w - gap * (len(values) - 1)) / max(len(values), 1))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="font-family: Arial, sans-serif;">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width/2}" y="26" text-anchor="middle" font-size="18" font-weight="700" fill="#1f2933">{escape_svg(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#9aa5b1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#9aa5b1"/>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + i * (bar_w + gap)
        bar_h = plot_h * value / max_value
        y = top + plot_h - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="11" fill="#334e68">{value:.0f}</text>')
        parts.append(
            f'<text x="{x + bar_w/2:.1f}" y="{top + plot_h + 18}" text-anchor="end" '
            f'transform="rotate(-35 {x + bar_w/2:.1f},{top + plot_h + 18})" font-size="11" fill="#334e68">{escape_svg(label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def heatmap_svg(matrix, title, width=760, height=430):
    rows = [str(x) for x in matrix.index]
    cols = [str(x) for x in matrix.columns]
    values = matrix.fillna(0).values.astype(float)
    max_value = max(values.max(), 1e-9)
    left, top = 150, 58
    cell_w = (width - left - 30) / len(cols)
    cell_h = (height - top - 50) / len(rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="font-family: Arial, sans-serif;">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18" font-weight="700" fill="#1f2933">{escape_svg(title)}</text>',
    ]
    for j, col in enumerate(cols):
        x = left + j * cell_w + cell_w / 2
        parts.append(f'<text x="{x:.1f}" y="{top - 14}" text-anchor="middle" font-size="12" fill="#334e68">{escape_svg(col)}</text>')
    for i, row in enumerate(rows):
        y = top + i * cell_h
        parts.append(f'<text x="{left - 12}" y="{y + cell_h/2 + 4:.1f}" text-anchor="end" font-size="12" fill="#334e68">{escape_svg(row)}</text>')
        for j, value in enumerate(values[i]):
            intensity = value / max_value
            r = int(239 - 135 * intensity)
            g = int(246 - 72 * intensity)
            b = int(249 - 47 * intensity)
            x = left + j * cell_w
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w - 3:.1f}" height="{cell_h - 3:.1f}" fill="rgb({r},{g},{b})"/>')
            parts.append(f'<text x="{x + cell_w/2:.1f}" y="{y + cell_h/2 + 4:.1f}" text-anchor="middle" font-size="11" fill="#1f2933">{value:.1%}</text>')
    parts.append("</svg>")
    return "".join(parts)


def write_svg(svg, path):
    path.write_text(svg, encoding="utf-8")
    print(f"Saved {path.name}")


def create_visualizations(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    clean_path = output_dir / "clean_cart_orders.csv"
    item_path = output_dir / "order_items_long.csv"
    user_path = output_dir / "user_feature_store.csv"

    if not clean_path.exists() or not item_path.exists() or not user_path.exists():
        raise FileNotFoundError("Run cleaning.py first to create cleaned data files.")

    orders = pd.read_csv(clean_path)
    order_items = pd.read_csv(item_path)
    user_features = pd.read_csv(user_path)

    segments = pd.cut(
        user_features["avg_order_value"],
        bins=[-float("inf"), user_features["avg_order_value"].quantile(0.33), user_features["avg_order_value"].quantile(0.66), float("inf")],
        labels=["Budget", "Medium", "Premium"],
    ).value_counts().reindex(["Budget", "Medium", "Premium"])

    meal_slots = orders["meal_time_slot"].value_counts().reindex(["Breakfast", "Lunch", "Snacks", "Dinner", "Late Night"])
    cuisines = orders["restaurant_cuisine"].value_counts().head(8)
    addon_items = order_items[order_items["item_type"] == "addon"].copy()
    addon_mix = addon_items["addon_category"].value_counts().reindex(["Beverage", "Side", "Dessert", "Condiment"]).dropna()
    top_addons = addon_items["item_name"].value_counts().head(10)
    attach_rate = (orders.groupby("meal_time_slot")["has_addon"].mean().reindex(["Breakfast", "Lunch", "Snacks", "Dinner", "Late Night"]) * 100).round(1)
    category_slot = pd.crosstab(addon_items["addon_category"], addon_items["meal_time_slot"], normalize="columns")
    category_slot = category_slot.reindex(index=["Beverage", "Side", "Dessert", "Condiment"], columns=["Breakfast", "Lunch", "Snacks", "Dinner", "Late Night"]).fillna(0)

    write_svg(bar_svg(segments, "User Segments by Average Order Value", color="#2F6F73"), output_dir / "user_segments.svg")
    write_svg(bar_svg(meal_slots, "Order Traffic by Meal Time Slot", color="#7B8794"), output_dir / "meal_slot_traffic.svg")
    write_svg(bar_svg(cuisines, "Cuisine Preference by Order Count", color="#D64550"), output_dir / "cuisine_preference.svg")
    write_svg(bar_svg(addon_mix, "Add-On Category Mix", color="#3E7C59"), output_dir / "addon_category_mix.svg")
    write_svg(bar_svg(top_addons, "Top Add-On Items", color="#9F580A"), output_dir / "top_addons.svg")
    write_svg(bar_svg(attach_rate, "Add-On Attach Rate by Meal Slot (%)", color="#52606D"), output_dir / "attach_rate_by_slot.svg")
    write_svg(heatmap_svg(category_slot, "Add-On Category Share Within Each Meal Slot"), output_dir / "category_slot_heatmap.svg")
    return True


if __name__ == "__main__":
    create_visualizations()
