# Hands-On Haiti: Food Price Modeling

This project explores Haiti's food economy through data. After initial EDA highlighted price variance and skew, a Ridge regression model (with log-transformed prices) was trained using market, commodity, and regional features. Tuning the regularization parameter improved stability across price levels, with RMSE =0.47 USD and R² ~0.48 on test data. Final residuals showed no major bias, confirming a well-fit, interpretable model for small-scale forecasting. Finally, using Folium, I mapped median prices across Haiti’s markets to explore regional disparities. These interactive maps provided geographic context for commodity trends and helped surface structural inequalities in Haiti's food access.

