import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Try to import TensorFlow with fallback
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
    # Set TensorFlow to use CPU only to avoid GPU issues
    tf.config.set_visible_devices([], 'GPU')
except ImportError as e:
    st.warning("⚠️ TensorFlow not available. LSTM model will be disabled.")
    tf = None
except Exception as e:
    st.warning(f"⚠️ TensorFlow import error: {str(e)}. LSTM model will be disabled.")
    tf = None

# Set page configuration
st.set_page_config(
    page_title="📈 Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for ticker {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def create_lag_features(data, n_lags=5):
    """Create lag features for Linear Regression"""
    df = data.copy()
    for i in range(1, n_lags + 1):
        df[f'Close_lag_{i}'] = df['Close'].shift(i)
    
    # Add technical indicators
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['Volume_MA'] = df['Volume'].rolling(window=5).mean()
    df['Price_Change'] = df['Close'].pct_change()
    
    return df.dropna()

def prepare_lstm_data(data, look_back=60):
    """Prepare data for LSTM model"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    """Build LSTM model architecture"""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not available")
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_linear_regression(data, target_col='Close'):
    """Train Linear Regression model"""
    # Prepare features
    feature_cols = [col for col in data.columns if col != target_col and 'Close' not in col]
    feature_cols.extend([col for col in data.columns if 'Close_lag' in col or 'MA_' in col])
    
    X = data[feature_cols].fillna(method='ffill').fillna(method='bfill')
    y = data[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred, X_train.columns

def train_lstm_model(data, look_back=60):
    """Train LSTM model"""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not available for LSTM training")
    
    X, y, scaler = prepare_lstm_data(data, look_back)
    
    # Reshape for LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Train-test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Build and train model
    model = build_lstm_model((X_train.shape[1], 1))
    
    with st.spinner("Training LSTM model... This may take a few minutes."):
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_test, y_test),
            verbose=0
        )
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_actual = scaler.inverse_transform(y_pred).flatten()
    
    return model, y_test_actual, y_pred_actual, scaler, history

def predict_future_prices(model, data, model_type, days_ahead, scaler=None, look_back=60):
    """Predict future stock prices"""
    if model_type == "Linear Regression":
        # Use last available data point for prediction
        last_data = data.tail(1)
        predictions = []
        
        for _ in range(days_ahead):
            # Create features for prediction
            feature_cols = [col for col in data.columns if 'Close_lag' in col or 'MA_' in col]
            features = last_data[feature_cols].fillna(method='ffill').fillna(method='bfill')
            
            pred = model.predict(features)[0]
            predictions.append(pred)
            
            # Update lag features (simplified approach)
            # In practice, you'd want to update all lag features properly
            
        return predictions
    
    else:  # LSTM
        # Get last 'look_back' days of data
        last_sequence = data['Close'].tail(look_back).values
        scaled_sequence = scaler.transform(last_sequence.reshape(-1, 1))
        
        predictions = []
        current_sequence = scaled_sequence.copy()
        
        for _ in range(days_ahead):
            # Reshape for prediction
            X_pred = current_sequence.reshape((1, look_back, 1))
            pred_scaled = model.predict(X_pred, verbose=0)
            
            # Inverse transform
            pred_actual = scaler.inverse_transform(pred_scaled)[0, 0]
            predictions.append(pred_actual)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[1:], pred_scaled)
        
        return predictions

def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return mae, rmse, mape

def plot_predictions(data, y_test, y_pred, future_dates, future_predictions, ticker):
    """Create interactive plot of predictions"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Historical Prices',
        line=dict(color='blue')
    ))
    
    # Test predictions
    test_dates = data.index[-len(y_test):]
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=y_test,
        mode='lines',
        name='Actual (Test)',
        line=dict(color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=y_pred,
        mode='lines',
        name='Predicted (Test)',
        line=dict(color='red', dash='dash')
    ))
    
    # Future predictions
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        mode='lines+markers',
        name='Future Predictions',
        line=dict(color='orange', dash='dot'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f'{ticker} Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        template='plotly_white',
        height=600
    )
    
    return fig

# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">📈 AI Stock Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("Predict stock prices using **Linear Regression** or **LSTM** models")
    
    # Sidebar Configuration
    st.sidebar.header("🎛️ Configuration Panel")
    
    # Stock ticker input
    ticker = st.sidebar.text_input(
        "Stock Ticker Symbol",
        value="AAPL",
        help="Enter stock ticker (e.g., AAPL, TSLA, GOOGL, INFY)"
    ).upper()
    
    # Date selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365*2),
            max_value=datetime.now()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    # Model selection - conditional based on TensorFlow availability
    if TENSORFLOW_AVAILABLE:
        model_options = ["Linear Regression", "LSTM"]
    else:
        model_options = ["Linear Regression"]
        st.sidebar.info("💡 LSTM requires TensorFlow. Only Linear Regression is available.")
    
    model_type = st.sidebar.selectbox(
        "Choose Model Type",
        model_options,
        help="Linear Regression: Fast, good for short-term trends\nLSTM: Advanced neural network, better for complex patterns"
    )
    
    # Prediction horizon
    prediction_days = st.sidebar.slider(
        "Prediction Horizon (Days)",
        min_value=1,
        max_value=30,
        value=7,
        help="Number of days to predict into the future"
    )
    
    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        if model_type == "LSTM" and TENSORFLOW_AVAILABLE:
            look_back = st.slider("LSTM Look-back Period", 30, 120, 60)
        else:
            n_lags = st.slider("Number of Lag Features", 3, 10, 5)
    
    # Run prediction button
    if st.sidebar.button("🚀 Run Prediction", type="primary"):
        
        # Validate inputs
        if start_date >= end_date:
            st.error("Start date must be before end date!")
            return
        
        if not ticker:
            st.error("Please enter a valid stock ticker!")
            return
        
        # Main content area
        with st.spinner(f"Fetching data for {ticker}..."):
            stock_data = fetch_stock_data(ticker, start_date, end_date)
        
        if stock_data is None or stock_data.empty:
            st.error("Could not fetch stock data. Please check the ticker symbol.")
            return
        
        # Display basic stock info
        st.subheader(f"📊 {ticker} Stock Analysis")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${stock_data['Close'][-1]:.2f}")
        with col2:
            price_change = stock_data['Close'][-1] - stock_data['Close'][-2]
            st.metric("Daily Change", f"${price_change:.2f}", f"{price_change:.2f}")
        with col3:
            st.metric("Volume", f"{stock_data['Volume'][-1]:,.0f}")
        with col4:
            volatility = stock_data['Close'].pct_change().std() * np.sqrt(252) * 100
            st.metric("Annual Volatility", f"{volatility:.1f}%")
        
        # Model training and prediction
        st.subheader(f"🤖 {model_type} Model Training")
        
        try:
            if model_type == "Linear Regression":
                # Prepare data with lag features
                processed_data = create_lag_features(stock_data, n_lags)
                
                with st.spinner("Training Linear Regression model..."):
                    model, X_test, y_test, y_pred, feature_cols = train_linear_regression(processed_data)
                
                # Calculate metrics
                mae, rmse, mape = calculate_metrics(y_test, y_pred)
                
                # Future predictions
                future_predictions = predict_future_prices(
                    model, processed_data, model_type, prediction_days
                )
                
            elif model_type == "LSTM" and TENSORFLOW_AVAILABLE:
                model, y_test, y_pred, scaler, history = train_lstm_model(stock_data, look_back)
                
                # Calculate metrics
                mae, rmse, mape = calculate_metrics(y_test, y_pred)
                
                # Future predictions
                future_predictions = predict_future_prices(
                    model, stock_data, model_type, prediction_days, scaler, look_back
                )
            else:
                st.error("LSTM model is not available. TensorFlow installation required.")
                return
            
            # Display model performance
            st.subheader("📈 Model Performance")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Mean Absolute Error</h4>
                    <h2>${mae:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Root Mean Square Error</h4>
                    <h2>${rmse:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Mean Absolute Percentage Error</h4>
                    <h2>{mape:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Future predictions display
            st.subheader("🔮 Future Price Predictions")
            
            # Create future dates
            last_date = stock_data.index[-1]
            future_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
            
            # Display predictions table
            predictions_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': [f"${price:.2f}" for price in future_predictions]
            })
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(predictions_df, use_container_width=True)
            
            with col2:
                avg_prediction = np.mean(future_predictions)
                current_price = stock_data['Close'][-1]
                price_trend = "📈 Upward" if avg_prediction > current_price else "📉 Downward"
                change_pct = ((avg_prediction - current_price) / current_price) * 100
                
                st.metric(
                    "Predicted Trend",
                    price_trend,
                    f"{change_pct:+.1f}%"
                )
            
            # Interactive plot
            st.subheader("📊 Price Prediction Visualization")
            fig = plot_predictions(stock_data, y_test, y_pred, future_dates, future_predictions, ticker)
            st.plotly_chart(fig, use_container_width=True)
            
            # Model insights
            if model_type == "LSTM" and TENSORFLOW_AVAILABLE:
                st.subheader("🧠 LSTM Training Progress")
                
                # Plot training history
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    y=history.history['loss'],
                    name='Training Loss',
                    line=dict(color='blue')
                ))
                fig_loss.add_trace(go.Scatter(
                    y=history.history['val_loss'],
                    name='Validation Loss',
                    line=dict(color='red')
                ))
                fig_loss.update_layout(
                    title='Model Training Progress',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    template='plotly_white'
                )
                st.plotly_chart(fig_loss, use_container_width=True)
            
            # Download predictions
            st.subheader("💾 Download Predictions")
            predictions_csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=predictions_csv,
                file_name=f"{ticker}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
            st.info("Please try with different parameters or check your data.")
    
    # Information section
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        ### How It Works
        
        **Linear Regression Model:**
        - Uses historical price data and technical indicators
        - Creates lag features (previous days' prices)
        - Fast training and prediction
        - Good for identifying linear trends
        
        **LSTM Model:**
        - Deep learning neural network
        - Learns complex patterns in sequential data
        - Better for capturing non-linear relationships
        - Requires more computational time
        
        ### Disclaimer
        ⚠️ **This app is for educational purposes only. Stock price predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with financial professionals and do your own research.**
        
        ### Data Source
        Stock data is fetched from Yahoo Finance using the `yfinance` library.
        """)

if __name__ == "__main__":
    main()
