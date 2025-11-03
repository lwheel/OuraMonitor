"""
Enhanced Automated Oura Ring Flare Prediction System
Includes weather data, detailed HRV analysis, and condition-specific markers
No manual input required - learns from physiological patterns alone
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

class EnhancedFlarePredictor:
    """
    Fully automated flare prediction using Oura Ring data + weather
    Tailored for UCTD, POTS, Fibromyalgia, HSD, and CRMO
    """
    
    def __init__(self, personal_access_token, latitude=39.2904, longitude=-76.6122):
        """
        Initialize predictor
        
        Args:
            personal_access_token: Oura API token
            latitude: Your location latitude (default: Baltimore)
            longitude: Your location longitude (default: Baltimore)
        """
        self.oura_base_url = "https://api.ouraring.com/v2/usercollection"
        self.headers = {"Authorization": f"Bearer {personal_access_token}"}
        self.latitude = latitude
        self.longitude = longitude
        self.scaler = StandardScaler()
        self.model = None
        
    def _make_request(self, endpoint, params=None):
        """Make Oura API request with error handling"""
        url = f"{self.oura_base_url}/{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {endpoint}: {e}")
            return None
    
    def get_date_range(self, days_back=300):
        """Generate date range for API requests"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        return start_date.isoformat(), end_date.isoformat()
    
    def fetch_weather_data(self, start_date, end_date):
        """
        Fetch historical weather data including barometric pressure
        Uses Open-Meteo API (free, no key required)
        """
        print("  Fetching weather data...")
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'start_date': start_date,
            'end_date': end_date,
            'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean,'
                     'relative_humidity_2m_mean,pressure_msl_mean,wind_speed_10m_max,'
                     'precipitation_sum',
            'timezone': 'America/New_York'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            weather_df = pd.DataFrame({
                'date': pd.to_datetime(data['daily']['time']),
                'temp_max': data['daily']['temperature_2m_max'],
                'temp_min': data['daily']['temperature_2m_min'],
                'temp_mean': data['daily']['temperature_2m_mean'],
                'humidity': data['daily']['relative_humidity_2m_mean'],
                'pressure': data['daily']['pressure_msl_mean'],  # Barometric pressure
                'wind_speed': data['daily']['wind_speed_10m_max'],
                'precipitation': data['daily']['precipitation_sum']
            })
            
            print(f"    ✓ {len(weather_df)} days of weather data")
            return weather_df
            
        except Exception as e:
            print(f"    ✗ Could not fetch weather data: {e}")
            return pd.DataFrame()
    
    def fetch_all_oura_data(self, days_back=300):
        """Fetch all relevant Oura data types"""
        start_date, end_date = self.get_date_range(days_back)
        print(f"Fetching Oura data from {start_date} to {end_date}...")
        
        endpoints = {
            'sleep': 'daily_sleep',
            'sleep_time_series': 'sleep',  # 5-min sleep data with detailed HRV
            'readiness': 'daily_readiness',
            'activity': 'daily_activity',
            'stress': 'daily_stress',
            'heart_rate': 'heartrate'  # Continuous HR data
        }
        
        data = {}
        for name, endpoint in endpoints.items():
            print(f"  Fetching {name}...")
            params = {"start_date": start_date, "end_date": end_date}
            result = self._make_request(endpoint, params)
            if result and 'data' in result:
                data[name] = pd.DataFrame(result['data'])
                print(f"    ✓ {len(data[name])} records")
            else:
                data[name] = pd.DataFrame()
                print(f"    ✗ No data")
        
        # Fetch weather data
        weather = self.fetch_weather_data(start_date, end_date)
        if not weather.empty:
            data['weather'] = weather
        
        return data
    
    def extract_hrv_features(self, sleep_time_series_df):
        """
        Extract detailed HRV metrics from sleep data
        Critical for fibromyalgia, POTS, and autonomic dysfunction detection
        """
        if sleep_time_series_df.empty:
            return pd.DataFrame()
        
        hrv_features = []
        
        for _, row in sleep_time_series_df.iterrows():
            try:
                # Extract HRV from heart rate data if available
                if 'heart_rate' in row and row['heart_rate'] is not None:
                    hr_data = row['heart_rate']
                    
                    # Calculate HRV proxies from HR data
                    if isinstance(hr_data, dict) and 'items' in hr_data:
                        hr_values = hr_data['items']
                        if hr_values:
                            hr_array = np.array(hr_values)
                            
                            hrv_features.append({
                                'day': row['day'] if 'day' in row else None,
                                'hr_mean': np.mean(hr_array),
                                'hr_std': np.std(hr_array),  # Proxy for HRV
                                'hr_min': np.min(hr_array),
                                'hr_max': np.max(hr_array),
                                'hr_range': np.max(hr_array) - np.min(hr_array),
                                'hr_cv': np.std(hr_array) / np.mean(hr_array) if np.mean(hr_array) > 0 else 0
                            })
            except Exception as e:
                continue
        
        if hrv_features:
            return pd.DataFrame(hrv_features)
        return pd.DataFrame()
    
    def extract_features(self, raw_data):
        """Extract comprehensive ML features from all data sources"""
        print("\nExtracting enhanced features...")
        
        if raw_data['readiness'].empty:
            raise ValueError("No readiness data available")
        
        # Start with readiness data
        df = raw_data['readiness'][['day', 'score', 'temperature_deviation', 
                                     'temperature_trend_deviation']].copy()
        df.columns = ['date', 'readiness_score', 'temp_deviation', 'temp_trend']
        
        # Convert date to datetime immediately
        df['date'] = pd.to_datetime(df['date'])
        
        # Add sleep features
        if not raw_data['sleep'].empty:
            sleep = raw_data['sleep'][['day', 'score']].copy()
            sleep.columns = ['date', 'sleep_score']
            sleep['date'] = pd.to_datetime(sleep['date'])
            
            # Extract nested contributors
            if 'contributors' in raw_data['sleep'].columns:
                contributors = pd.json_normalize(raw_data['sleep']['contributors'])
                for col in contributors.columns:
                    sleep[f'sleep_{col}'] = contributors[col].values
            
            df = df.merge(sleep, on='date', how='left')
        
        # Add activity features
        if not raw_data['activity'].empty:
            activity = raw_data['activity'][['day', 'score']].copy()
            activity.columns = ['date', 'activity_score']
            activity['date'] = pd.to_datetime(activity['date'])
            df = df.merge(activity, on='date', how='left')
        
        # Add stress features (critical for all conditions)
        if not raw_data['stress'].empty:
            stress = raw_data['stress'][['day', 'stress_high', 'recovery_high']].copy()
            stress.columns = ['date', 'stress_high', 'recovery_high']
            stress['date'] = pd.to_datetime(stress['date'])
            df = df.merge(stress, on='date', how='left')
        
        # Add detailed HRV features (critical for fibromyalgia & POTS)
        if not raw_data['sleep_time_series'].empty:
            hrv_data = self.extract_hrv_features(raw_data['sleep_time_series'])
            if not hrv_data.empty:
                hrv_data.columns = ['date'] + [f'hrv_{col}' if col != 'day' else col 
                                               for col in hrv_data.columns if col != 'day']
                hrv_data['date'] = pd.to_datetime(hrv_data['date'])
                df = df.merge(hrv_data, on='date', how='left')
        
        # Add heart rate throughout day (POTS indicator)
        if not raw_data['heart_rate'].empty:
            # Aggregate daily HR stats
            hr_daily = self._aggregate_heart_rate(raw_data['heart_rate'])
            if not hr_daily.empty:
                # Ensure date is datetime
                hr_daily['date'] = pd.to_datetime(hr_daily['date'])
                df = df.merge(hr_daily, on='date', how='left')
        
        # Add weather data (critical for all autoimmune conditions)
        if 'weather' in raw_data and not raw_data['weather'].empty:
            weather = raw_data['weather'].copy()
            weather['date'] = pd.to_datetime(weather['date'])
            df = df.merge(weather, on='date', how='left')
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Create derived features
        df = self._create_derived_features(df)
        
        # Create condition-specific risk markers
        df = self._create_condition_markers(df)
        
        print(f"  ✓ Created dataset with {len(df)} days and {len(df.columns)} features")
        return df
    
    def _aggregate_heart_rate(self, hr_df):
        """Aggregate continuous heart rate data to daily stats"""
        if hr_df.empty:
            return pd.DataFrame()
        
        # Ensure timestamp is datetime
        if 'timestamp' in hr_df.columns:
            hr_df['timestamp'] = pd.to_datetime(hr_df['timestamp'])
        else:
            return pd.DataFrame()
        
        daily_hr = []
        
        for date in hr_df['timestamp'].dt.date.unique():
            day_data = hr_df[hr_df['timestamp'].dt.date == date]
            
            if not day_data.empty and 'bpm' in day_data.columns:
                hr_values = day_data['bpm'].dropna()
                if len(hr_values) > 0:
                    daily_hr.append({
                        'date': pd.Timestamp(date),
                        'hr_daytime_mean': hr_values.mean(),
                        'hr_daytime_max': hr_values.max(),
                        'hr_daytime_min': hr_values.min(),
                        'hr_daytime_std': hr_values.std(),
                        'hr_spike_count': (hr_values > hr_values.mean() + hr_values.std()).sum()
                    })
        
        return pd.DataFrame(daily_hr) if daily_hr else pd.DataFrame()
    
    def _create_derived_features(self, df):
        """Create rolling averages, trends, and weather change features"""
        
        feature_cols = [col for col in df.columns if col not in ['date']]
        
        for col in feature_cols:
            if df[col].dtype in [np.float64, np.int64]:
                # Rolling averages
                df[f'{col}_3d_avg'] = df[col].rolling(window=3, min_periods=1).mean()
                df[f'{col}_7d_avg'] = df[col].rolling(window=7, min_periods=1).mean()
                
                # Rate of change
                df[f'{col}_change'] = df[col].diff()
                df[f'{col}_change_3d'] = df[col].diff(3)
                
                # Deviation from baseline
                if not df[col].isna().all():
                    baseline = df[col].mean()
                    df[f'{col}_deviation'] = df[col] - baseline
        
        # Weather-specific features (critical for autoimmune flares)
        if 'pressure' in df.columns:
            # Barometric pressure drop (major trigger)
            df['pressure_drop_24h'] = -df['pressure'].diff()
            df['pressure_drop_48h'] = -df['pressure'].diff(2)
            df['pressure_rapid_change'] = df['pressure_drop_24h'].abs() > 5  # >5 hPa change
            
        if 'temp_mean' in df.columns:
            # Temperature swings
            df['temp_swing_24h'] = df['temp_mean'].diff().abs()
            df['temp_swing_48h'] = df['temp_mean'].diff(2).abs()
            
        if 'humidity' in df.columns:
            # Humidity changes
            df['humidity_change'] = df['humidity'].diff()
        
        # Day of week and season patterns
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['season'] = df['month'].apply(lambda x: (x%12 + 3)//3)  # 1=winter, 2=spring, etc.
        
        return df
    
    def _infer_menstrual_phase(self, df):
        """
        Infer menstrual cycle phase from temperature patterns
        Based on Oura's method: temp drops in follicular, rises in luteal
        """
        if 'temp_deviation' not in df.columns:
            return df
        
        # Calculate 7-day rolling average of temperature
        df['temp_7d_avg'] = df['temp_deviation'].rolling(window=7, min_periods=3).mean()
        
        # Detect phase based on temperature trend
        # Positive trend = luteal phase (progesterone raises temp)
        # Negative trend = follicular phase (estrogen lowers temp)
        df['temp_trend_7d'] = df['temp_7d_avg'].diff(3)
        
        # Infer cycle phase (0=unknown, 1=follicular, 2=luteal)
        df['inferred_cycle_phase'] = 0
        df.loc[df['temp_trend_7d'] < -0.1, 'inferred_cycle_phase'] = 1  # Follicular
        df.loc[df['temp_trend_7d'] > 0.1, 'inferred_cycle_phase'] = 2   # Luteal
        
        # Estimate cycle day (rough approximation)
        # Look for temperature drops (potential period start)
        temp_drops = (df['temp_deviation'].diff() < -0.2) & (df['temp_deviation'] < -0.2)
        
        # Initialize cycle day counter
        df['estimated_cycle_day'] = 0
        current_day = 0
        
        for idx in df.index:
            if temp_drops.loc[idx]:
                current_day = 1  # Reset on likely period start
            else:
                current_day += 1
                if current_day > 35:  # Cap at typical max cycle length
                    current_day = 1
            
            df.loc[idx, 'estimated_cycle_day'] = current_day
        
        return df
    
    def _create_condition_markers(self, df):
        """
        Create condition-specific risk markers based on research
        """
        
        # Infer menstrual cycle phase first
        df = self._infer_menstrual_phase(df)
        
        # Menstrual cycle markers (impacts all conditions)
        if 'inferred_cycle_phase' in df.columns:
            # Luteal phase often worsens autoimmune symptoms
            df['luteal_phase'] = df['inferred_cycle_phase'] == 2
            # Pre-menstrual (days 25-28 of typical cycle)
            df['premenstrual'] = (df['estimated_cycle_day'] >= 25) & (df['estimated_cycle_day'] <= 28)
        
        # POTS markers: Orthostatic HR increase, autonomic dysfunction
        if 'hr_daytime_mean' in df.columns and 'hrv_hr_mean' in df.columns:
            # High daytime HR relative to sleep HR
            df['pots_hr_elevation'] = (df['hr_daytime_mean'] - df['hrv_hr_mean']) > 30
            df['pots_hr_spike_risk'] = df['hr_spike_count'] > 10  # Frequent HR spikes
        
        # Fibromyalgia markers: Low HRV, sympathetic dominance
        if 'hrv_hr_std' in df.columns:
            # Low HRV indicates autonomic dysfunction
            hrv_baseline = df['hrv_hr_std'].median()
            df['fibro_low_hrv'] = df['hrv_hr_std'] < (hrv_baseline * 0.7)
        
        # UCTD/Autoimmune markers: Temperature, inflammation
        if 'temp_deviation' in df.columns:
            df['autoimmune_inflammation'] = df['temp_deviation'] > 0.3
            df['autoimmune_fever_pattern'] = df['temp_deviation'] > 0.5
        
        # HSD/EDS markers: Activity intolerance, fatigue
        if 'activity_score' in df.columns and 'sleep_score' in df.columns:
            df['hsd_overexertion'] = (df['activity_score'] > 85) & (df['sleep_score'] < 70)
            df['hsd_poor_recovery'] = (df['readiness_score'] < 70) & (df['sleep_total_sleep'] > 7)
        
        # CRMO markers: Stress/inflammation patterns
        if 'stress_high' in df.columns:
            df['crmo_high_stress'] = df['stress_high'] > 0.6
        
        # Weather sensitivity (all conditions)
        if 'pressure_drop_24h' in df.columns:
            df['weather_sensitive_flare'] = df['pressure_drop_24h'] > 5
        
        return df
    
    def create_risk_labels(self, df):
        """
        Create multi-condition risk labels from physiological markers
        """
        print("\nCreating condition-specific risk labels...")
        
        # Define flare conditions for each disease
        conditions = []
        
        # General markers
        if 'readiness_score' in df.columns:
            conditions.append(df['readiness_score'] < 70)
        if 'sleep_score' in df.columns:
            conditions.append(df['sleep_score'] < 70)
        if 'temp_deviation' in df.columns:
            conditions.append(df['temp_deviation'] > 0.3)
        if 'stress_high' in df.columns:
            conditions.append(df['stress_high'] > 0.5)
        
        # Weather triggers
        if 'pressure_drop_24h' in df.columns:
            conditions.append(df['pressure_drop_24h'] > 5)
        if 'temp_swing_24h' in df.columns:
            conditions.append(df['temp_swing_24h'] > 10)
        
        # HRV markers (fibromyalgia, POTS)
        if 'hrv_hr_std' in df.columns:
            hrv_baseline = df['hrv_hr_std'].median()
            conditions.append(df['hrv_hr_std'] < (hrv_baseline * 0.7))
        
        # HR elevation (POTS)
        if 'hr_daytime_mean' in df.columns:
            hr_baseline = df['hr_daytime_mean'].median()
            conditions.append(df['hr_daytime_mean'] > (hr_baseline * 1.15))
        
        # Combine conditions (3+ markers = likely flare)
        if conditions:
            condition_count = sum([c.astype(int) for c in conditions])
            df['likely_flare_day'] = (condition_count >= 3).astype(int)
            
            flare_count = df['likely_flare_day'].sum()
            print(f"  ✓ Identified {flare_count} likely flare days ({flare_count/len(df)*100:.1f}%)")
        else:
            df['likely_flare_day'] = 0
            print("  ⚠ Could not create risk labels - insufficient data")
        
        return df
    
    def train_model(self, df):
        """Train both anomaly detector and classification model"""
        print("\nTraining prediction models...")
        
        # Select features for modeling
        feature_cols = [col for col in df.columns 
                       if col not in ['date', 'likely_flare_day'] 
                       and df[col].dtype in [np.float64, np.int64]]
        
        X = df[feature_cols].fillna(df[feature_cols].mean())
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest for anomaly detection
        anomaly_model = IsolationForest(
            contamination=0.15,
            random_state=42,
            n_estimators=100
        )
        
        predictions = anomaly_model.fit_predict(X_scaled)
        anomaly_scores = anomaly_model.score_samples(X_scaled)
        
        df['anomaly_score'] = anomaly_scores
        df['is_anomaly'] = (predictions == -1).astype(int)
        
        # Train Random Forest for flare prediction if we have labels
        if 'likely_flare_day' in df.columns and df['likely_flare_day'].sum() > 10:
            y = df['likely_flare_day']
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            rf_model.fit(X_scaled, y)
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n  Top 10 predictive features:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
            
            self.model = {'anomaly': anomaly_model, 'classification': rf_model}
        else:
            self.model = {'anomaly': anomaly_model}
        
        anomaly_count = df['is_anomaly'].sum()
        print(f"\n  ✓ Detected {anomaly_count} anomalous days ({anomaly_count/len(df)*100:.1f}%)")
        
        return df, feature_cols
    
    def predict_next_days(self, df, feature_cols, days_ahead=3):
        """Enhanced prediction with condition-specific insights"""
        print(f"\nGenerating {days_ahead}-day forecast...")
        
        recent = df.tail(14).copy()  # Look at past 2 weeks
        
        # Current status indicators
        current_readiness = recent['readiness_score'].iloc[-1] if 'readiness_score' in recent.columns else None
        current_temp = recent['temp_deviation'].iloc[-1] if 'temp_deviation' in recent.columns else None
        current_pressure = recent['pressure'].iloc[-1] if 'pressure' in recent.columns else None
        
        # Calculate trends
        trends = {}
        for col in ['readiness_score', 'sleep_score', 'temp_deviation', 'pressure']:
            if col in recent.columns:
                values = recent[col].dropna()
                if len(values) >= 3:
                    trend = values.iloc[-1] - values.iloc[-3]
                    trends[col] = trend
        
        # Recent anomalies
        recent_anomalies = recent['is_anomaly'].sum() if 'is_anomaly' in recent.columns else 0
        
        # Calculate risk score
        risk_score = 0
        risk_factors = []
        condition_alerts = []
        
        # Readiness risk
        if current_readiness and current_readiness < 70:
            risk_score += 25
            risk_factors.append(f"Low readiness ({current_readiness:.0f}/100)")
        
        # Temperature/inflammation
        if current_temp and current_temp > 0.3:
            risk_score += 30
            risk_factors.append(f"Elevated temperature (+{current_temp:.2f}°C)")
            condition_alerts.append("⚠️ UCTD/Autoimmune inflammation detected")
        
        # Declining trends
        if 'readiness_score' in trends and trends['readiness_score'] < -5:
            risk_score += 20
            risk_factors.append("Declining readiness trend")
        
        # Weather: Pressure drops
        if 'pressure_drop_24h' in recent.columns:
            recent_pressure_drops = (recent['pressure_drop_24h'] > 5).sum()
            if recent_pressure_drops > 0:
                risk_score += 20
                risk_factors.append(f"Barometric pressure drops ({recent_pressure_drops} recent)")
                condition_alerts.append("🌧️ Weather-sensitive flare risk")
        
        # HRV (fibromyalgia/POTS)
        if 'hrv_hr_std' in recent.columns:
            current_hrv = recent['hrv_hr_std'].iloc[-1]
            hrv_baseline = recent['hrv_hr_std'].median()
            if current_hrv < (hrv_baseline * 0.7):
                risk_score += 15
                risk_factors.append("Low HRV (autonomic dysfunction)")
                condition_alerts.append("💗 Fibromyalgia/POTS risk elevated")
        
        # POTS-specific
        if 'hr_spike_count' in recent.columns:
            recent_spikes = recent['hr_spike_count'].iloc[-1]
            if recent_spikes > 10:
                risk_score += 15
                risk_factors.append(f"Frequent HR spikes ({recent_spikes})")
                condition_alerts.append("⚡ POTS symptoms detected")
        
        # Menstrual cycle risk
        if 'premenstrual' in recent.columns and recent['premenstrual'].iloc[-1]:
            risk_score += 15
            risk_factors.append("Pre-menstrual phase (increased flare risk)")
            condition_alerts.append("🌙 Hormonal fluctuation period")
        
        if 'luteal_phase' in recent.columns and recent['luteal_phase'].sum() >= 10:
            risk_score += 10
            risk_factors.append("Extended luteal phase")
        
        # Recent anomalies
        if recent_anomalies >= 4:
            risk_score += 20
            risk_factors.append(f"{recent_anomalies} anomalous days in past 2 weeks")
        
        risk_score = min(100, risk_score)
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = "HIGH"
            warning = "🚨 FLARE WARNING - High Risk"
        elif risk_score >= 40:
            risk_level = "MODERATE"
            warning = "⚡ Elevated Flare Risk"
        else:
            risk_level = "LOW"
            warning = "✓ Risk Level Normal"
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'warning': warning,
            'risk_factors': risk_factors,
            'condition_alerts': condition_alerts,
            'trends': trends,
            'current_metrics': {
                'readiness': current_readiness,
                'temp_deviation': current_temp,
                'pressure': current_pressure
            }
        }
    
    def save_model(self, df, feature_cols, output_dir="oura_model"):
        """Save model and processed data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed data
        df.to_csv(f"{output_dir}/processed_data.csv", index=False)
        
        # Save feature list
        with open(f"{output_dir}/features.json", 'w') as f:
            json.dump(feature_cols, f)
        
        # Save model and scaler
        with open(f"{output_dir}/model.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(f"{output_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"\n✓ Model, scaler, and data saved to {output_dir}/")
    
    def generate_report(self, df, prediction):
        """Generate comprehensive health report"""
        print("\n" + "="*70)
        print("ENHANCED MULTI-CONDITION FLARE RISK ASSESSMENT")
        print("Conditions: UCTD, POTS, Fibromyalgia, HSD, CRMO")
        print("="*70)
        
        print(f"\n{prediction['warning']}")
        print(f"Overall Risk Score: {prediction['risk_score']}/100 ({prediction['risk_level']})")
        
        if prediction['condition_alerts']:
            print("\n🔍 Condition-Specific Alerts:")
            for alert in prediction['condition_alerts']:
                print(f"  {alert}")
        
        if prediction['risk_factors']:
            print("\n⚠️ Current Risk Factors:")
            for factor in prediction['risk_factors']:
                print(f"  • {factor}")
        
        if prediction['trends']:
            print("\n📊 Recent Trends (past 3 days):")
            for metric, trend in prediction['trends'].items():
                direction = "↑" if trend > 0 else "↓"
                print(f"  {direction} {metric}: {trend:+.1f}")
        
        # Current readings
        print("\n📍 Current Readings:")
        metrics = prediction['current_metrics']
        if metrics['readiness']:
            print(f"  Readiness: {metrics['readiness']:.0f}/100")
        if metrics['temp_deviation']:
            print(f"  Temperature Deviation: {metrics['temp_deviation']:+.2f}°C")
        if metrics['pressure']:
            print(f"  Barometric Pressure: {metrics['pressure']:.1f} hPa")
        
        # Cycle information
        if 'inferred_cycle_phase' in df.columns:
            recent_phase = df['inferred_cycle_phase'].iloc[-1]
            estimated_day = df['estimated_cycle_day'].iloc[-1]
            phase_name = {0: 'Unknown', 1: 'Follicular', 2: 'Luteal'}
            print(f"  Estimated Cycle Phase: {phase_name.get(recent_phase, 'Unknown')}")
            print(f"  Estimated Cycle Day: {estimated_day}")
        
        # Historical context
        if 'likely_flare_day' in df.columns:
            recent_flares = df.tail(30)['likely_flare_day'].sum()
            print(f"\n📅 Past 30 Days: {recent_flares} likely flare days")
        
        print("\n" + "="*70)


def get_current_location():
    """Automatically detect current location using IP geolocation"""
    try:
        # Use ipapi.co for free IP geolocation (no key needed)
        response = requests.get('https://ipapi.co/json/', timeout=5)
        response.raise_for_status()
        data = response.json()
        
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        city = data.get('city', 'Unknown')
        region = data.get('region', 'Unknown')
        
        if latitude and longitude:
            print(f"✓ Detected location: {city}, {region} ({latitude:.4f}, {longitude:.4f})")
            return latitude, longitude
        
    except Exception as e:
        print(f"⚠ Could not auto-detect location: {e}")
        print("  Using default: Baltimore, MD")
    
    # Fallback to Baltimore
    return 39.2904, -76.6122


import plotly.graph_objs as go
import plotly.express as px
from plotly.colors import qualitative
from textwrap import dedent
import numpy as np
import pandas as pd
from datetime import datetime

def render_dashboard(df: pd.DataFrame, prediction: dict, output_path: str = "dashboard.html"):
    """
    Create a standalone, self-contained HTML dashboard for the daily report.

    Expects:
      - df: your processed dataframe from extract_features/create_risk_labels/train_model
      - prediction: dict returned by predict_next_days(...)
    Writes:
      - output_path HTML with embedded Plotly JS & CSS (no server required)
    """

    # ---------- helpers ----------
    def num(x, default="—", fmt="{:.1f}"):
        try:
            return fmt.format(float(x))
        except Exception:
            return default

    def last_non_null(s, default=None):
        s = s.dropna()
        return s.iloc[-1] if len(s) else default

    def delta_str(series, periods=3, suffix="", pos="▲", neg="▼"):
        s = series.dropna()
        if len(s) >= periods + 1:
            d = s.iloc[-1] - s.iloc[-(periods+1)]
            arrow = pos if d > 0 else (neg if d < 0 else "→")
            return f"{arrow} {d:+.1f}{suffix}"
        return "—"

    def sparkline(series, title, ytitle="", height=120):
        s = series.copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(width=2)))
        fig.update_layout(
            title=title, height=height, margin=dict(l=30, r=10, t=40, b=30),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(title=ytitle, showgrid=True, gridcolor="rgba(0,0,0,.06)")
        )
        return fig

    # ---------- FIX: Properly handle the dataframe ----------
    df = df.copy()
    
    # Ensure date column exists and is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
        # Set date as index for easier slicing
        if df.index.name != 'date':
            df = df.set_index("date")
    else:
        # If no date column, assume index is already datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a 'date' column or DatetimeIndex")
    
    # Get the most recent 30 and 14 days of data
    # Use tail() instead of last() to get the most recent rows regardless of date gaps
    recent = df.tail(30) if len(df) >= 30 else df
    last14 = df.tail(14) if len(df) >= 14 else df
    
    print(f"  Dashboard using data from {recent.index.min().date()} to {recent.index.max().date()}")

    # ---------- risk gauge ----------
    risk_score = float(prediction.get("risk_score", 0))
    risk_level = prediction.get("risk_level", "LOW")
    warning = prediction.get("warning", "✓ Risk Level Normal")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={"suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.3},
            "steps": [
                {"range": [0, 40], "color": "#cfead6"},
                {"range": [40, 70], "color": "#ffe7bc"},
                {"range": [70, 100], "color": "#ffd1d1"},
            ],
            "threshold": {
                "line": {"color": "#111", "width": 3},
                "thickness": 0.6,
                "value": risk_score,
            },
        },
        title={"text": "Overall Flare Risk"}
    ))
    gauge.update_layout(height=250, margin=dict(l=20, r=20, t=60, b=20))

    # ---------- key metrics tiles - USE MOST RECENT DATA ----------
    readiness = last_non_null(last14.get("readiness_score", pd.Series(dtype=float)))
    temp_dev = last_non_null(last14.get("temp_deviation", pd.Series(dtype=float)))
    pressure = last_non_null(last14.get("pressure", pd.Series(dtype=float)))

    readiness_delta = delta_str(last14.get("readiness_score", pd.Series(dtype=float)), periods=3)
    temp_delta = delta_str(last14.get("temp_deviation", pd.Series(dtype=float)), periods=3, suffix="°C")
    pressure_delta = delta_str(last14.get("pressure", pd.Series(dtype=float)), periods=3, suffix=" hPa")

    metric_cards = [
        {
            "title": "Readiness",
            "value": f"{num(readiness, '—', '{:.0f}')}/100",
            "delta": readiness_delta
        },
        {
            "title": "Temp Dev",
            "value": f"{num(temp_dev, '—', '{:+.2f}')}°C",
            "delta": temp_delta
        },
        {
            "title": "Pressure",
            "value": f"{num(pressure, '—', '{:.1f}')} hPa",
            "delta": pressure_delta
        }
    ]

    # ---------- chips / alerts ----------
    alerts = prediction.get("condition_alerts", []) or []
    factors = prediction.get("risk_factors", []) or []

    # Add to render_dashboard function, after the helpers section:

    # ---------- CORRELATION HEATMAP ----------
    corr_fig = None
    correlation_cols = [c for c in df.columns if c in [
        'readiness_score', 'sleep_score', 'temp_deviation', 
        'pressure', 'stress_high', 'hr_daytime_mean', 'likely_flare_day'
    ]]

    if len(correlation_cols) >= 4:
        corr_matrix = df[correlation_cols].corr()
        
        # Create heatmap
        corr_fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[c.replace('_', ' ').title() for c in corr_matrix.columns],
            y=[c.replace('_', ' ').title() for c in corr_matrix.columns],
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        corr_fig.update_layout(
            title="What Drives Your Flares? (Correlation Matrix)",
            height=400,
            margin=dict(l=120, r=20, t=60, b=100),
            xaxis=dict(tickangle=-45)
        )

    # Add to parts dict:
    parts['correlation'] = corr_fig.to_html(full_html=False, include_plotlyjs=False) if corr_fig else ""
        
    # ---------- trend charts - FIXED ----------
    trend_specs = [
        ("readiness_score", "Readiness (14d)", ""),
        ("sleep_score", "Sleep (14d)", ""),
        ("pressure", "Barometric Pressure (14d)", "hPa"),
        ("temp_deviation", "Temperature Deviation (14d)", "°C"),
    ]
    trend_figs = []
    for col, ttl, ytt in trend_specs:
        if col in last14.columns and last14[col].notna().sum() >= 3:
            # Create series with proper datetime index
            series = last14[col].dropna()
            trend_figs.append(sparkline(series, ttl, ytitle=ytt))
        else:
            # empty placeholder
            fig = go.Figure()
            fig.update_layout(title=ttl, height=120, margin=dict(l=30, r=10, t=40, b=30))
            trend_figs.append(fig)

    # ---------- anomalies bar - FIXED ----------
    if "is_anomaly" in recent.columns:
        anomalies = recent["is_anomaly"].fillna(0)
        anom_fig = px.bar(
            x=anomalies.index,
            y=anomalies.values,
            labels={"x": "Date", "y": "Anomaly"},
            title="Recent Anomalies (30d)"
        )
        anom_fig.update_layout(height=200, margin=dict(l=40, r=10, t=40, b=30), showlegend=False)
    else:
        anom_fig = go.Figure()
        anom_fig.update_layout(title="Recent Anomalies (30d)", height=200)

    # ---------- top features (optional) ----------
    top_feat_fig = None
    # Only create if we have enough numeric columns
    num_cols = df.select_dtypes(include=[np.number])
    if len(num_cols.columns) >= 5:
        # Get columns with highest variance
        variances = num_cols.var().sort_values(ascending=False).head(10)
        if len(variances) > 0:
            top_feat_fig = px.bar(
                y=variances.index,
                x=variances.values,
                orientation="h",
                title="High-Variance Features",
                labels={"x": "Variance", "y": "Feature"}
            )
            top_feat_fig.update_layout(height=280, margin=dict(l=150, r=20, t=50, b=40))

    # ---------- past-30-day flare strip - FIXED ----------
    flare_strip = None
    if "likely_flare_day" in recent.columns:
        strip = recent["likely_flare_day"].fillna(0)
        colors = ["#e8f5e9" if v == 0 else "#ffcccc" for v in strip.values]
        flare_strip = go.Figure(
            data=[go.Bar(
                x=strip.index, 
                y=[1]*len(strip), 
                marker_color=colors, 
                hovertext=[f"{d.date()} — {'Flare' if v==1 else 'OK'}" for d, v in zip(strip.index, strip.values)], 
                hoverinfo="text"
            )]
        )
        flare_strip.update_layout(
            title="Flare Days (Past 30d)",
            height=90,
            margin=dict(l=40, r=20, t=40, b=10),
            xaxis=dict(showticklabels=False),
            yaxis=dict(visible=False)
        )

    # ---------- assemble HTML ----------
    # Convert figures to HTML snippets with embedded JS
    parts = {
        "gauge": gauge.to_html(full_html=False, include_plotlyjs=True),
        "anom": anom_fig.to_html(full_html=False, include_plotlyjs=False),
        "trend_1": trend_figs[0].to_html(full_html=False, include_plotlyjs=False),
        "trend_2": trend_figs[1].to_html(full_html=False, include_plotlyjs=False),
        "trend_3": trend_figs[2].to_html(full_html=False, include_plotlyjs=False),
        "trend_4": trend_figs[3].to_html(full_html=False, include_plotlyjs=False),
        "features": (top_feat_fig.to_html(full_html=False, include_plotlyjs=False) if top_feat_fig else ""),
        "flare_strip": (flare_strip.to_html(full_html=False, include_plotlyjs=False) if flare_strip else ""),
    }

    today_str = datetime.now().strftime("%A, %B %d, %Y")
    badge = {"HIGH": "danger", "MODERATE": "warn", "LOW": "ok"}.get(risk_level, "ok")

    css = dedent("""
    <style>
      :root {
        --bg: #0b0d10;
        --panel: #12151a;
        --panel-2: #171b21;
        --text: #e8eef6;
        --muted: #a9b4c2;
        --ok: #2ea44f;
        --warn: #e2a300;
        --danger: #d83a3a;
        --chip: #20252d;
        --border: #2a2f37;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0; padding: 24px 24px 48px;
        background: radial-gradient(1200px 800px at 10% -10%, #0d1117, #0b0d10);
        color: var(--text); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Apple Color Emoji","Segoe UI Emoji";
      }
      .container { max-width: 1200px; margin: 0 auto; }
      .header {
        display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 16px;
      }
      .title { font-size: 22px; font-weight: 600; letter-spacing: .2px; }
      .subtitle { color: var(--muted); font-size: 14px; }
      .badge {
        padding: 6px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; letter-spacing: .4px;
      }
      .badge.ok { background: color-mix(in oklab, var(--ok) 20%, transparent); color: #b9ffd1; border: 1px solid color-mix(in oklab, var(--ok) 50%, transparent); }
      .badge.warn { background: color-mix(in oklab, var(--warn) 20%, transparent); color: #ffe4a6; border: 1px solid color-mix(in oklab, var(--warn) 50%, transparent); }
      .badge.danger { background: color-mix(in oklab, var(--danger) 20%, transparent); color: #ffd2d2; border: 1px solid color-mix(in oklab, var(--danger) 50%, transparent); }

      .grid { display: grid; gap: 16px; }
      .grid.cols-2 { grid-template-columns: 1.2fr 1fr; }
      .grid.cols-3 { grid-template-columns: repeat(3, 1fr); }
      .panel {
        background: linear-gradient(180deg, var(--panel), var(--panel-2));
        border: 1px solid var(--border); border-radius: 14px; padding: 16px;
        box-shadow: 0 10px 24px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.02);
      }
      .panel h3 { margin: 0 0 10px; font-size: 14px; color: var(--muted); font-weight: 600; letter-spacing: .3px; }
      .metric-cards { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; }
      .metric-card { background: var(--chip); border: 1px solid var(--border); border-radius: 12px; padding: 12px; }
      .metric-title { color: var(--muted); font-size: 12px; margin-bottom: 4px; }
      .metric-value { font-size: 22px; font-weight: 700; }
      .metric-delta { font-size: 12px; color: var(--muted); margin-top: 2px; }
      .chips { display: flex; flex-wrap: wrap; gap: 8px; }
      .chip {
        background: var(--chip); border: 1px solid var(--border); color: var(--text);
        padding: 6px 10px; border-radius: 999px; font-size: 12px;
      }
      .list { display: grid; gap: 6px; font-size: 13px; color: var(--muted); }
      .footer { margin-top: 18px; color: var(--muted); font-size: 12px; text-align: right; }
      @media (max-width: 980px) {
        .grid.cols-2 { grid-template-columns: 1fr; }
        .metric-cards { grid-template-columns: 1fr; }
      }
    /* Add to existing CSS */
    .recommendations { display: grid; gap: 12px; margin-top: 12px; }
    .rec-card {
    display: flex; gap: 12px; align-items: start;
    background: var(--chip); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px;
    }
    .rec-card.critical { border-left: 3px solid var(--danger); }
    .rec-card.high { border-left: 3px solid var(--warn); }
    .rec-card.medium { border-left: 3px solid #5b9bd5; }
    .rec-card.low { border-left: 3px solid var(--ok); }
    .rec-icon { font-size: 24px; line-height: 1; }
    .rec-content { flex: 1; }
    .rec-title { font-weight: 600; font-size: 13px; margin-bottom: 4px; }
    .rec-detail { font-size: 12px; color: var(--muted); line-height: 1.5; }

    /* Correlation heatmap styling */
    .correlation-panel { grid-column: 1 / -1; }

    /* Weekly pattern */
    .pattern-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; text-align: center; }
    .day-card { background: var(--chip); border: 1px solid var(--border); border-radius: 8px; padding: 12px 8px; }
    .day-name { font-size: 11px; color: var(--muted); margin-bottom: 4px; }
    .day-risk { font-size: 20px; font-weight: 700; }
    .day-risk.high { color: var(--danger); }
    .day-risk.medium { color: var(--warn); }
    .day-risk.low { color: var(--ok); }
    </style>
    """)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Daily Flare Dashboard</title>
{css}
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <div class="title">Daily Flare Dashboard</div>
        <div class="subtitle">{today_str}</div>
      </div>
      <div class="badge {badge}">{warning}</div>
    </div>

    <div class="grid cols-2">
      <div class="panel">
        <h3>Risk Overview</h3>
        {parts["gauge"]}
        <div class="metric-cards">
          {''.join([f'''
            <div class="metric-card">
              <div class="metric-title">{m["title"]}</div>
              <div class="metric-value">{m["value"]}</div>
              <div class="metric-delta">{m["delta"]}</div>
            </div>''' for m in metric_cards])}
        </div>
      </div>

      <div class="panel">
        <h3>Condition Alerts</h3>
        <div class="chips">
          {''.join([f'<div class="chip">{a}</div>' for a in alerts]) or '<div class="chip">No active alerts</div>'}
        </div>
        <h3 style="margin-top:14px;">Key Risk Factors</h3>
        <div class="list">
          {''.join([f'<div>• {f}</div>' for f in factors]) or '<div>• None elevated</div>'}
        </div>
      </div>
    </div>

    <div class="grid cols-2" style="margin-top:16px;">
      <div class="panel">{parts["trend_1"]}</div>
      <div class="panel">{parts["trend_2"]}</div>
    </div>
    <div class="grid cols-2" style="margin-top:16px;">
      <div class="panel">{parts["trend_3"]}</div>
      <div class="panel">{parts["trend_4"]}</div>
    </div>

    <div class="grid cols-2" style="margin-top:16px;">
      <div class="panel">
        {parts["anom"]}
      </div>
      <div class="panel">
        {parts["flare_strip"]}
        {'<div style="height:8px;"></div>'+parts["features"] if parts["features"] else '<h3>Features</h3><div style="color:var(--muted);font-size:13px;">No feature view available today.</div>'}
      </div>
    </div>

    <div class="footer">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
  </div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✓ Dashboard written to {output_path}")


def main():
    """Run the enhanced prediction system"""
    print("ENHANCED AUTOMATED FLARE PREDICTION SYSTEM")
    print("UCTD • POTS • Fibromyalgia • HSD • CRMO")
    print("Weather-aware • HRV Analysis • Multi-condition\n")
    
    
    # Get token from environment variable
    OURA_TOKEN = os.environ.get('OURA_TOKEN')
    
    if not OURA_TOKEN:
        raise ValueError("OURA_TOKEN not found. Please set it in your .env file")
    
    DAYS_BACK = 300
    
    # Auto-detect location
    latitude, longitude = get_current_location()
    print()
    
    # Initialize predictor
    predictor = EnhancedFlarePredictor(
        OURA_TOKEN,
        latitude=latitude,
        longitude=longitude
    )
    
    # Fetch all data
    raw_data = predictor.fetch_all_oura_data(days_back=DAYS_BACK)
    
    # Process features
    df = predictor.extract_features(raw_data)
    
    # Create risk labels
    df = predictor.create_risk_labels(df)
    
    # Train models
    df, feature_cols = predictor.train_model(df)
    
    # Predict upcoming risk
    prediction = predictor.predict_next_days(df, feature_cols, days_ahead=3)
    
    # Save everything
    predictor.save_model(df, feature_cols)
    
    # Generate report
    predictor.generate_report(df, prediction)
    
    # Save daily log
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"{today}.txt"
    
    # Save prediction summary to daily log
    with open(log_file, 'w') as f:
        f.write(f"FLARE RISK REPORT - {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{prediction['warning']}\n")
        f.write(f"Risk Score: {prediction['risk_score']}/100 ({prediction['risk_level']})\n\n")
        
        if prediction['condition_alerts']:
            f.write("Condition-Specific Alerts:\n")
            for alert in prediction['condition_alerts']:
                f.write(f"  {alert}\n")
            f.write("\n")
        
        if prediction['risk_factors']:
            f.write("Risk Factors:\n")
            for factor in prediction['risk_factors']:
                f.write(f"  • {factor}\n")
            f.write("\n")
        
        if prediction['trends']:
            f.write("Recent Trends (past 3 days):\n")
            for metric, trend in prediction['trends'].items():
                direction = "↑" if trend > 0 else "↓"
                f.write(f"  {direction} {metric}: {trend:+.1f}\n")
            f.write("\n")
        
        metrics = prediction['current_metrics']
        f.write("Current Readings:\n")
        if metrics['readiness']:
            f.write(f"  Readiness: {metrics['readiness']:.0f}/100\n")
        if metrics['temp_deviation']:
            f.write(f"  Temperature: {metrics['temp_deviation']:+.2f}°C\n")
        if metrics['pressure']:
            f.write(f"  Pressure: {metrics['pressure']:.1f} hPa\n")
        
        if 'inferred_cycle_phase' in df.columns:
            recent_phase = df['inferred_cycle_phase'].iloc[-1]
            estimated_day = df['estimated_cycle_day'].iloc[-1]
            phase_name = {0: 'Unknown', 1: 'Follicular', 2: 'Luteal'}
            f.write(f"  Cycle: Day {estimated_day} ({phase_name.get(recent_phase, 'Unknown')})\n")
        
        if 'likely_flare_day' in df.columns:
            recent_flares = df.tail(30)['likely_flare_day'].sum()
            f.write(f"\nPast 30 Days: {recent_flares} likely flare days\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Model trained on {len(df)} days of data\n")
    
    # after predictor.generate_report(df, prediction)
    render_dashboard(df, prediction, output_path="logs/dashboard.html")

    print(f"\n✓ Daily log saved to {log_file}")
    print("✓ System ready! This will run automatically at 4 PM daily.")


if __name__ == "__main__":
    main()