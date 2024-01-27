import pandas as pd
import statsmodels.api as sm

# Read CSV file
data = pd.read_csv('/home/xc217/fintech545/repo/Week02/Project/problem3.csv')

# Assume time series data is in a column named 'x'
ts = data['x']

best_aic = float('inf')
best_order = None
best_model = None

# Store AIC values for each model
aic_values = []

# Loop through AR(1) to AR(3) and MA(1) to MA(3) models
for p in range(1, 4):
   
    # Fit AR model
    try:
        model = sm.tsa.ARIMA(ts, order=(p, 0, 0))  # AR model, d=0 indicates non-seasonal differencing
        results = model.fit()
            
        # Compare AIC values
        current_aic = results.aic
        aic_values.append((f"AR{p}", current_aic))  # Store model and corresponding AIC value
        
    except Exception as e:
        # Handle exception for models that cannot be fit
        print(f"Error fitting AR({p}): {e}")

for q in range(1, 4):
    # Fit MA model
    try:
        model = sm.tsa.ARIMA(ts, order=(0, 0, q))  # MA model, d=0 indicates non-seasonal differencing
        results = model.fit()
        
        # Compare AIC values
        current_aic = results.aic
        aic_values.append((f"MA{q}", current_aic))  # Store model and corresponding AIC value
    
    except Exception as e:
        # Handle exception for models that cannot be fit
        print(f"Error fitting MA({q}): {e}")

# Print AIC values for each model
for model_order, aic_value in aic_values:
    print(f"{model_order}: AIC = {aic_value}")

# Loop through AR(1) to AR(3) and MA(1) to MA(3) models
for p in range(1, 4):
    for q in range(1, 4):
        # Fit AR model
        try:
            model = sm.tsa.ARIMA(ts, order=(p, 0, 0))  # AR model, d=0 indicates non-seasonal differencing
            results = model.fit()
            
            # Compare AIC values
            current_aic = results.aic
            if current_aic < best_aic:
                best_aic = current_aic
                best_order = (p, 0, 0)
                best_model = results.summary()
        
        except Exception as e:
            # Handle exception for models that cannot be fit
            print(f"Error fitting AR({p}): {e}")

        # Fit MA model
        try:
            model = sm.tsa.ARIMA(ts, order=(0, 0, q))  # MA model, d=0 indicates non-seasonal differencing
            results = model.fit()
            
            # Compare AIC values
            current_aic = results.aic
            if current_aic < best_aic:
                best_aic = current_aic
                best_order = (0, 0, q)
                best_model = results.summary()
        
        except Exception as e:
            # Handle exception for models that cannot be fit
            print(f"Error fitting MA({q}): {e}")

# Print information about the best model
print(f"Best Model Order: ARIMA{best_order}")
print(f"Best AIC: {best_aic}")
print("Best Model Summary:")
print(best_model)

best_bic = float('inf')
best_order = None
best_model = None

# Store BIC values for each model
bic_values = []

# Loop through AR(1) to AR(3) and MA(1) to MA(3) models
for p in range(1, 4):
   
    # Fit AR model
    try:
        model = sm.tsa.ARIMA(ts, order=(p, 0, 0))  # AR model, d=0 indicates non-seasonal differencing
        results = model.fit()
            
        # Compare BIC values
        current_bic = results.bic
        bic_values.append((f"AR{p}", current_bic))  # Store model and corresponding BIC value
        
    except Exception as e:
        # Handle exception for models that cannot be fit
        print(f"Error fitting AR({p}): {e}")

for q in range(1, 4):
    # Fit MA model
    try:
        model = sm.tsa.ARIMA(ts, order=(0, 0, q))  # MA model, d=0 indicates non-seasonal differencing
        results = model.fit()
        
        # Compare BIC values
        current_bic = results.bic
        bic_values.append((f"MA{q}", current_bic))  # Store model and corresponding BIC value
    
    except Exception as e:
        # Handle exception for models that cannot be fit
        print(f"Error fitting MA({q}): {e}")

# Print BIC values for each model
for model_order, bic_value in bic_values:
    print(f"{model_order}: BIC = {bic_value}")

# Loop through AR(1) to AR(3) and MA(1) to MA(3) models
for p in range(1, 4):
    for q in range(1, 4):
        # Fit AR model
        try:
            model = sm.tsa.ARIMA(ts, order=(p, 0, 0))  # AR model, d=0 indicates non-seasonal differencing
            results = model.fit()
            
            # Compare BIC values
            current_bic = results.bic
            if current_bic < best_bic:
                best_bic = current_bic
                best_order = (p, 0, 0)
                best_model = results.summary()
        
        except Exception as e:
            # Handle exception for models that cannot be fit
            print(f"Error fitting AR({p}): {e}")

        # Fit MA model
        try:
            model = sm.tsa.ARIMA(ts, order=(0, 0, q))  # MA model, d=0 indicates non-seasonal differencing
            results = model.fit()
            
            # Compare BIC values
            current_bic = results.bic
            if current_bic < best_bic:
                best_bic = current_bic
                best_order = (0, 0, q)
                best_model = results.summary()
        
        except Exception as e:
            # Handle exception for models that cannot be fit
            print(f"Error fitting MA({q}): {e}")

# Print information about the best model
print(f"Best Model Order: ARIMA{best_order}")
print(f"Best BIC: {best_bic}")
print("Best Model Summary:")
print(best_model)
