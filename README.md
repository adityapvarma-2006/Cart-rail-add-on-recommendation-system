# Cart-rail-add-on-recommendation-system
Personalized cart add-on recommendation system for food delivery, with synthetic data generation, visual analytics, temporal train-test evaluation, and a live cart rail recommender.

## Project Overview

The goal is to recommend relevant add-ons such as beverages, desserts, sides, and condiments while a user is building their cart.

Example:

> Current cart: Butter Chicken + Chole Kulche  
> Recommended add-ons: Fresh Lime Soda, Papad, Coke, Masala Chaas, Ice Cream Cup

The notebook is designed to run end-to-end without requiring external datasets.

## What This Notebook Contains

- Synthetic food delivery order data generation
- User profiles, restaurant metadata, item catalog, and cart composition
- Messy data simulation and cleaning
- Feature engineering for users, sessions, carts, discounts, festivals, and meal slots
- Exploratory visualizations
- Cart add-on recommendation model
- Temporal train-test evaluation
- Live cart rail recommendation demo

## Dataset Size

The generated dataset contains:

| Dataset Component | Count |
|---|---:|
| Raw synthetic orders | 20,946 |
| Cleaned orders | 20,108 |
| Long item-level rows | 48,326 |
| Users in feature store | 5,926 |
| Recommendation sessions | 19,429 |

## Train-Test Split

The model uses a temporal split to avoid future data leakage.

| Split | Sessions | Date Range |
|---|---:|---|
| Train | 15,543 | 2025-01-01 to 2025-03-13 |
| Test | 3,886 | 2025-03-13 to 2025-03-31 |

The first 80% of sessions by timestamp are used for training, and the final 20% are used for testing.

## Model Approach

The recommender is a lightweight personalized ranking system using:

- Global add-on popularity
- Cuisine-level add-on behavior
- Meal-slot behavior
- User-level add-on affinity
- Restaurant-level signals
- Cart co-occurrence patterns
- Price fit
- Discount and festival context
- Dietary compatibility

This avoids heavy dependencies like TensorFlow, XGBoost, or scikit-learn, making the notebook easy to run on GitHub, Colab, or Kaggle.

## Evaluation Results

| Metric | Result |
|---|---:|
| Test AUC | 83.84% |
| Test pair accuracy | 75.55% |
| Hit Rate @ 8 | 96.73% |
| Mean Reciprocal Rank | 39.75% |
| Median target rank | 3 |

Hit Rate @ 8 measures whether the actual next add-on appears inside the top 8 recommended cart rail items.

## How to Run

1. Clone the repository.
2. Open the notebook:

```bash
Cart_Add_On_Recommender.ipynb
