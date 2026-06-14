# Cart-rail-add-on-recommendation-system
Personalized cart add-on recommendation system for food delivery, with synthetic data generation, visual analytics, temporal train-test evaluation, and a live cart rail recommender.

This project generates synthetic food delivery cart data, cleans and enriches it, creates visual analytics, and trains a personalized recommendation system that suggests relevant add-ons such as beverages, desserts, sides, and condiments while a user is building their cart.

The project is intentionally built with simple, portable Python so it can run easily in a local environment, GitHub Codespaces, Kaggle, or Google Colab without requiring heavy machine learning frameworks.

## Project Objective

Food delivery carts often contain a main dish but miss high-intent add-ons such as drinks, desserts, dips, or sides. A cart rail recommender can improve the checkout experience by surfacing relevant add-ons at the right moment.

This project answers the question:

> Given a user's current cart, restaurant context, meal time, discount behavior, and historical preferences, which add-ons should be shown in the cart rail?

The recommender is evaluated using a temporal train-test split so the model is tested on future cart sessions instead of randomly mixed historical data.

## Repository Structure

```text
Cart_Add_On_Recommender/
│
├── data_generation.py      # Generates synthetic food delivery users, restaurants, catalog, and raw cart orders
├── cleaning.py             # Cleans messy records, parses carts, and creates feature tables
├── visualization.py        # Creates SVG visualizations from cleaned data
├── recommendation.py       # Builds recommendation sessions, trains/evaluates the model, and prints sample recommendations
└── README.md               # Project documentation
```

When the pipeline is run, generated files are saved under:

```text
cart_add_on_outputs/
```

This folder is created automatically and contains CSV datasets, SVG charts, model metrics, and sample recommendations.

## Features

- Synthetic food delivery order generation
- User profile simulation
- Restaurant and cuisine metadata
- Main item and add-on catalog
- Cart composition strings with item order preserved
- Messy data simulation, including null IDs, missing values, uppercase carts, and timestamp errors
- Cleaning and imputation logic
- Temporal feature engineering
- User feature store generation
- Item-level long table generation
- SVG visualizations without requiring plotting libraries
- Personalized cart add-on ranking model
- Temporal train-test evaluation
- Live sample cart rail recommendation output

## Dataset Generated

The pipeline creates a realistic synthetic dataset with:

- Users
- Restaurants
- Main dishes
- Add-ons
- Orders
- Cart compositions
- Discounts
- Session engagement features
- Meal time slots
- Weekend and festival flags

During a verified run, the project generated approximately:

| Dataset Component | Count |
|---|---:|
| Raw orders | 21,164 |
| Cleaned orders | 20,317 |
| Item-level rows | 48,951 |
| Users in feature store | 5,916 |
| Recommendation sessions | 19,675 |

Exact numbers may vary slightly if generation settings are changed, but the default random seed keeps the output reproducible.

## Synthetic Data Design

The generated data includes several entities:

### Users

Each user has:

- `user_id`
- `dietary_pref`
- `order_frequency`
- `favorite_addon_category`
- `price_sensitivity`

### Restaurants

Each restaurant has:

- `restaurant_id`
- `cuisine`
- `city_zone`
- `restaurant_rating`

### Catalog

The catalog contains:

- Main dishes
- Beverages
- Desserts
- Sides
- Condiments

Each item has:

- `item_id`
- `item_name`
- `item_type`
- `cuisine`
- `addon_category`
- `base_price`
- `diet_type`

### Orders

Each order contains:

- User and restaurant IDs
- Timestamp
- Cart composition
- Total items
- Total value
- Discount clicked
- App opens since last order
- Time spent ordering

The cart composition preserves item order, which allows the recommender to model "next add-on" behavior.

Example cart:

```text
1.MAIN_NOR_002(North Indian|$12.40) >> 2.MAIN_NOR_004(North Indian|$8.30) >> 3.ADD_BEV_003(Beverage|$2.00)
```

## Data Cleaning

The cleaning step handles:

- Missing order or cart data
- Invalid restaurant IDs
- Missing discount values
- Missing session engagement values
- Timestamp parsing errors
- Numeric type conversion
- User-level mode imputation
- Global fallback imputation

It also creates useful features such as:

- `meal_time_slot`
- `is_weekend`
- `is_friday_night`
- `is_festival`
- `engagement_score`
- `is_discounted`
- `addon_count`
- `has_addon`
- `cart_main_count`
- `cart_addon_value`
- `attach_rate_order_value_share`

## Visualizations

The visualization script creates SVG charts, including:

- User segments by average order value
- Order traffic by meal time slot
- Cuisine preference by order count
- Add-on category mix
- Top add-on items
- Add-on attach rate by meal slot
- Add-on category share by meal slot

SVG is used so the project does not require `matplotlib`, `seaborn`, or other plotting libraries.

## Recommendation Approach

The recommender is a personalized ranking system designed for a horizontal cart rail.

It scores eligible add-ons using:

- Global add-on popularity
- Cuisine-level add-on behavior
- Meal-slot behavior
- User-level add-on affinity
- Restaurant-level signals
- Cart item co-occurrence
- Favorite add-on category
- Price fit
- Discount context
- Festival context
- Dietary compatibility

The model ranks candidate add-ons and returns the top recommendations for the current cart.

## Why This Is Not a Simple Popularity Model

The recommender does not only suggest globally popular items. It adjusts recommendations based on:

- What the user tends to add
- What pairs well with the current cart
- The restaurant cuisine
- The current meal time
- The user's dietary preference
- The user's price sensitivity
- Whether a discount was clicked
- Whether the order is around a festival

This makes the recommendations context-aware and personalized.

## Train-Test Split

The recommendation sessions are sorted by timestamp.

The first 80% of sessions are used for training, and the final 20% are used for testing.

This simulates a realistic production scenario where the model is trained on past behavior and evaluated on future cart sessions.

During a verified run:

| Split | Sessions | Date Range |
|---|---:|---|
| Train | 15,740 | 2025-01-01 to 2025-03-13 |
| Test | 3,935 | 2025-03-13 to 2025-03-31 |

This avoids future data leakage that would happen with a random train-test split.

## Evaluation Metrics

The project reports both pairwise ranking quality and cart rail quality.

### Test AUC

Measures how well the model scores the true next add-on higher than sampled alternatives.

### Test Pair Accuracy

Measures accuracy on balanced pairs where the model compares the true next add-on against one eligible negative add-on.

### Hit Rate @ 8

Measures whether the actual next add-on appears in the top 8 recommended items.

This is the most relevant metric for a cart rail because users usually see multiple add-on cards in a horizontal list.

### Mean Reciprocal Rank

Measures how high the correct add-on appears in the ranked list.

During a verified run:

| Metric | Result |
|---|---:|
| Test AUC | 84.35% |
| Test pair accuracy | 75.86% |
| Hit Rate @ 8 | 96.27% |
| Mean Reciprocal Rank | 39.68% |

## How to Run

Install the required Python packages:

```bash
pip install pandas numpy
```

Run the full pipeline:

```bash
python recommendation.py
```

This automatically runs:

1. Data generation, if raw data does not already exist
2. Cleaning and feature engineering
3. Visualization generation
4. Recommendation session creation
5. Model training
6. Temporal test evaluation
7. Sample cart rail recommendation output

## Running Individual Steps

You can also run each file independently:

```bash
python data_generation.py
python cleaning.py
python visualization.py
python recommendation.py
```

Recommended order:

1. `data_generation.py`
2. `cleaning.py`
3. `visualization.py`
4. `recommendation.py`

## Output Files

After running the pipeline, the following files are created under `cart_add_on_outputs/`:

```text
raw_cart_orders.csv
clean_cart_orders.csv
item_catalog.csv
users.csv
restaurants.csv
order_items_long.csv
user_feature_store.csv
model_summary_metrics.csv
sample_recommendations.csv
user_segments.svg
meal_slot_traffic.svg
cuisine_preference.svg
addon_category_mix.svg
top_addons.svg
attach_rate_by_slot.svg
category_slot_heatmap.svg
```

## Example Recommendation Output

Example current cart:

```text
Butter Chicken + Chole Kulche
```

Example cart rail recommendations:

| Rank | Item | Category | Rationale |
|---:|---|---|---|
| 1 | Coke | Beverage | matches user's add-on affinity |
| 2 | Fresh Lime Soda | Beverage | matches user's add-on affinity |
| 3 | Ice Cream Cup | Dessert | strong dessert context |
| 4 | Masala Chaas | Beverage | matches user's add-on affinity |
| 5 | Gulab Jamun | Dessert | strong dessert context |
| 6 | Papad | Side | low-friction impulse add-on |
| 7 | Extra Sambar | Side | low-friction impulse add-on |
| 8 | Iced Tea | Beverage | matches user's add-on affinity |

## Tech Stack

- Python
- pandas
- numpy
- Standard library collections and math utilities
- SVG generation for charts

No TensorFlow, PyTorch, XGBoost, or scikit-learn is required.

## Project Strengths

- Runs end to end without external data
- Uses a temporal split instead of random leakage-prone evaluation
- Produces meaningful cart rail metrics
- Keeps code modular across four Python files
- Uses simple dependencies
- Generates explainable recommendations
- Suitable for a GitHub portfolio project

## Limitations

This project uses synthetic data, so the results should not be interpreted as real-world business performance.

The recommender is a strong lightweight baseline, but a production system would require:

- Real transaction logs
- Real impression and click data
- Online A/B testing
- Cold-start handling
- Inventory and availability checks
- Diversity constraints
- Latency-aware serving
- Monitoring for drift and bias

## Future Improvements

Possible next steps:

- Add a learned ranking model
- Add item embeddings
- Add collaborative filtering
- Add cart rail diversity constraints
- Add a FastAPI inference endpoint
- Add unit tests
- Add a Streamlit demo
- Add configurable data generation settings
- Add support for real anonymized order data

## Summary

This project demonstrates an end-to-end recommendation workflow for food delivery cart add-ons:

1. Generate realistic synthetic order data
2. Clean and enrich the data
3. Visualize customer and cart behavior
4. Train a context-aware ranking recommender
5. Evaluate it on future sessions
6. Produce explainable cart rail recommendations

It is designed to be understandable, reproducible, and easy to extend.

## Contributors

- @adityapvarma-2006
- @iota-me24b079
