# T20 World Cup 2026 Predictions

A machine learning project that predicts the outcomes of the 2026 ICC Men's T20 World Cup. 

> **[View the Live Visualizer on Vercel](https://t20wc-one.vercel.app/)**

## How It Works

The prediction pipeline uses a Logistic Regression model trained on historical T20 International cricket matches stretching back over a decade.

The model evaluates four key features to determine match outcomes:
1. **Elo Rating:** A strength metric that rewards points based on the opponent's ranking.
2. **Recent Form:** A rolling win-rate evaluating the momentum of both teams in their last 5 fixtures.
3. **Head-to-Head:** Historical win percentages between two specific teams.
4. **Toss Impact:** The measurable advantage of winning the coin toss in recent games.

To ground historical predictions in current reality, the model output is blended with the official ICC Men's T20I Team Rankings (70% ML prediction / 30% ICC ranking).
