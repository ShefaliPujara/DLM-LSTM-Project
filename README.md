# LSTM-Based Stock Price Prediction

## Contributors:
- **Shefali Pujara (055044)**
- **Vandana Jain (055058)**

## Introduction & Objective
Stock market prediction remains a challenging endeavor due to the inherent volatility and non-linearity of financial data. Traditional statistical models, like ARIMA, often struggle to capture the complex relationships and dependencies within time series data, leading to limited accuracy in forecasting future stock prices.

This project addresses these limitations by leveraging the power of Long Short-Term Memory (LSTM) networks, a type of recurrent neural network specifically designed to handle sequential data. The objective is to develop an LSTM-based model capable of accurately predicting stock prices for **Bank of India** and the **NIFTY 50 index**. The project encompasses data preprocessing, LSTM model development, rigorous training and evaluation, and a comparative analysis with the traditional ARIMA model to assess the LSTM's effectiveness in capturing stock market dynamics.

---

## Data Collection
Stock price data for **Bank of India (BANKINDIA.NS)** and the **NIFTY 50 index (^NSEI)** was obtained using the `yfinance` library, a popular Python package for downloading financial data from Yahoo Finance.

The dataset retrieved covers a historical period from **September 17, 2007, to March 25, 2025** (or the most recent available data). It includes the following essential features for each trading day:

- **Open:** The price at which the stock opened for trading.
- **High:** The highest price reached during the trading day.
- **Low:** The lowest price reached during the trading day.
- **Close:** The price at which the stock closed for trading.
- **Volume:** The number of shares traded during the day.

---

## Exploratory Data Analysis (EDA)
### Summary Statistics & Visualizations
Before model development, a thorough exploratory data analysis was conducted to gain insights into the characteristics and patterns of the stock price data.

- **Descriptive Statistics:** Summary statistics such as mean, standard deviation, minimum, maximum, and quartiles were calculated for the stock prices to understand the distribution and variability of the data.
- **Line Charts:** Line charts were plotted to visualize the historical trends and patterns in the closing prices of both Bank of India and the NIFTY 50 index over time.
- **Correlation Heatmap:** A correlation heatmap was generated to analyze the relationships between different stock features (open, high, low, close, volume).

---

## Feature Engineering
To enhance the predictive power of the LSTM model, several feature engineering techniques were applied:

- **Lag Features:** Previous stock prices were included as features to allow the model to learn from past price movements.
- **Relative Strength Index (RSI):** A momentum indicator calculated to assess recent price changes and identify overbought or oversold conditions.
- **Moving Averages:** 10-day and 50-day moving averages were computed to smooth out price fluctuations and identify trend directions.

---

## Data Preprocessing & Train-Test Split
The following preprocessing steps were performed:

- **Missing Values Handling:** Missing values were addressed using imputation or removal.
- **Data Normalization:** The data was normalized using `MinMaxScaler` from `sklearn.preprocessing`.
- **Train-Test Split:** The dataset was divided into training, validation, and test sets for a robust evaluation of the model.

---

## LSTM Model Training
An LSTM-based neural network was constructed with the following architecture:

- **LSTM Layers:** One or more LSTM layers were used to capture temporal dependencies.
- **Dropout Layers:** Added to prevent overfitting.
- **Dense Layers:** Used for the output and intermediate layers.
- **Adam Optimizer:** Employed for efficient weight updates.
- **Mean Squared Error (MSE):** Used as the loss function.
- **Training Process:** The model was trained over multiple epochs, monitoring training and validation losses.

---

## Model Performance Evaluation
The LSTM model was evaluated using the following metrics:

- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **Mean Absolute Percentage Error (MAPE)**
- **Visualization:** Line charts comparing actual vs. predicted stock prices.

---

## Comparison with ARIMA
A traditional **ARIMA model** was implemented as a baseline to benchmark performance. The comparison focused on:

- **Model Performance:** The RMSE and MAPE values of both models were compared.
- **Non-linear Dependencies:** LSTM outperformed ARIMA in capturing complex, non-linear relationships.
- **Short-term vs. Long-term Trends:** ARIMA was better at short-term fluctuations, while LSTM captured long-term trends effectively.

---

## Final Predictions & Visualizations
### Predicted vs. Actual Stock Prices
Line charts were generated to compare predicted stock prices against actual stock prices.

### Residual Plot
A residual plot was created to analyze biases and systematic patterns in prediction errors.

---

## Observations
### Dataset Preparation
- **Normalization with MinMaxScaler** improved model convergence speed and stability.

### Model Performance
- **Trend Capture:** The LSTM model captured general trends effectively.
- **Prediction Accuracy:** Achieved low RMSE and MAE values, signifying good accuracy.

### Comparison with ARIMA
- **Non-linear Dependencies:** LSTM outperformed ARIMA in capturing non-linear relationships.
- **Short-term Trends:** ARIMA was better suited for short-term fluctuations.

### Prediction Accuracy
- **General Trends:** LSTM accurately captured general trends.
- **Extreme Fluctuations:** The model struggled during highly volatile periods.

---

## Managerial Insights
### Data-Driven Decision Making
- **Predictive Insights:** Helps investors make informed investment decisions.
- **Buy/Sell Strategies:** Identifies potential entry and exit points.

### Risk Management
- **Sudden Price Changes:** Incorporating macroeconomic indicators and sentiment analysis could improve robustness.
- **Risk Mitigation:** Predictions should be used alongside diversification and other risk management strategies.

### Portfolio Optimization
- **Balance & Diversification:** Helps in constructing well-balanced portfolios.

### Automation in Financial Analysis
- **Efficiency & Scalability:** AI-driven models significantly reduce manual effort in stock analysis and trading.

---

## Future Enhancements
### Hybrid Models
- **Sentiment Analysis & Macroeconomic Indicators:** Integrating these as input features could improve prediction accuracy.
- **Combining with Other Models:** Exploring hybrid models (LSTM + ARIMA/GARCH) for better forecasting.

### Hyperparameter Tuning
- **Optimization & Adaptability:** Using Grid Search or Bayesian Optimization for fine-tuning.

---

## Conclusion
This study demonstrates the effectiveness of LSTM networks for stock price prediction, particularly in stable market conditions. The model effectively captures trends and provides valuable insights for data-driven decision-making in financial markets.

However, the model’s limitations during extreme fluctuations highlight the need for additional external factors. Future enhancements like hybrid modeling and hyperparameter tuning could further refine performance.

This project contributes to AI-driven stock market analysis, offering valuable insights for investors leveraging advanced technologies for informed decision-making and portfolio management.
