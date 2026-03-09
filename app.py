
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime, timezone, timedelta
import json
from flask_mail import Mail, Message
import os
import random
import string
from functools import wraps
from flask_bcrypt import Bcrypt
from flask_cors import CORS
import time
import logging
from logging.handlers import RotatingFileHandler

# -----------------------------
# Flask Setup with Enhanced Configuration
# -----------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecret123'
# Asset version to force cache-busting during development
app.config['ASSET_VERSION'] = 'v2'

# Enhanced MongoDB connection
app.config["MONGO_URI"] = "mongodb://localhost:27017/PCOS"
mongo = PyMongo(app)
db = mongo.db
bcrypt = Bcrypt(app)

# Enhanced CORS configuration
CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/predict": {"origins": "*"}
})

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Enhanced Email configuration
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = "tshreesha2004@gmail.com"
app.config["MAIL_PASSWORD"] = "fahijsotlxtytnac"
app.config["MAIL_DEFAULT_SENDER"] = "tshreesha2004@gmail.com"

mail = Mail(app)

# Performance monitoring
app.config['PERFORMANCE_MONITORING'] = True

# Circuit breaker configuration
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None
    
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
    
    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"
    
    def can_execute(self):
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        return True  # HALF_OPEN

# Initialize circuit breakers
prediction_circuit_breaker = CircuitBreaker()
email_circuit_breaker = CircuitBreaker()

# Enhanced logging
if not os.path.exists('logs'):
    os.mkdir('logs')
file_handler = RotatingFileHandler('logs/pcos_care.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('PCOS Care startup')

# In-memory storage for verification codes (use database in production)
verification_codes = {}

# Symptom tracking data structure
symptom_categories = {
    "menstrual": ["irregular_cycles", "heavy_bleeding", "missed_periods", "painful_periods"],
    "physical": ["weight_gain", "acne", "hair_loss", "excess_hair", "skin_darkening", "fatigue"],
    "emotional": ["mood_swings", "anxiety", "depression", "irritability"],
    "metabolic": ["sugar_cravings", "increased_appetite", "thirst", "frequent_urination"]
}

# -----------------------------
# Enhanced User Wrapper for Flask-Login
# -----------------------------
class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data["_id"])
        self.email = user_data["email"]
        self.name = user_data.get("name", "")
        self.age = user_data.get("age", "")
        self.phone = user_data.get("phone", "")
        self.gender = user_data.get("gender", "")
        self.medical_history = user_data.get("medical_history", "")
        self.created_at = user_data.get("created_at", datetime.now(timezone.utc))
        self.last_login = user_data.get("last_login")
        self._is_active = user_data.get("is_active", True)
        self.preferences = user_data.get("preferences", {})
        self.symptom_tracking_enabled = user_data.get("symptom_tracking_enabled", False)

    @property
    def is_active(self):
        return self._is_active

    @is_active.setter
    def is_active(self, value):
        self._is_active = value

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'age': self.age,
            'phone': self.phone,
            'gender': self.gender,
            'preferences': self.preferences,
            'symptom_tracking_enabled': self.symptom_tracking_enabled
        }

@login_manager.user_loader
def load_user(user_id):
    user_data = db.users.find_one({"_id": ObjectId(user_id)})
    return User(user_data) if user_data else None

# -----------------------------
# Enhanced Admin Required Decorator
# -----------------------------
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.email != "admin@pcoscare.com":
            app.logger.warning(f'Unauthorized admin access attempt from {request.remote_addr}')
            return jsonify({'success': False, 'message': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function

# -----------------------------
# Performance Monitoring Decorator
# -----------------------------
def monitor_performance(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not app.config['PERFORMANCE_MONITORING']:
            return f(*args, **kwargs)
        
        start_time = time.time()
        try:
            result = f(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Log slow endpoints
            if execution_time > 1.0:  # Log if execution takes more than 1 second
                app.logger.warning(f'Slow endpoint: {request.path} took {execution_time:.2f}s')
            
            # Add performance header for frontend monitoring
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], str):
                # For template responses
                return result
            elif hasattr(result, 'headers'):
                # For JSON responses
                result.headers['X-Execution-Time'] = f'{execution_time:.3f}s'
            
            return result
        except Exception as e:
            end_time = time.time()
            app.logger.error(f'Error in {request.path}: {str(e)} - took {end_time - start_time:.2f}s')
            raise
    
    return decorated_function

# -----------------------------
# Enhanced Machine Learning Model with Model Comparison
# -----------------------------
class PCOSPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = [
            'age', 'weight', 'height', 'bmi', 'cycle_length',
            'waist_hip_ratio', 'skin_darkening', 'hair_growth',
            'weight_gain', 'pimples', 'fast_food', 'regular_exercise', 
            'mood_swings', 'family_history', 'sleep_hours', 
            'stress_level', 'blood_pressure'
        ]
        self.best_model = None
        self.best_accuracy = 0.0
        self.train_models()

    def generate_synthetic_data(self, n_samples=3000):
        np.random.seed(42)
        data = []
        for i in range(n_samples):
            has_pcos = np.random.choice([0, 1], p=[0.6, 0.4])
            
            if has_pcos:
                # PCOS cases - more realistic parameters
                age = np.random.normal(28, 6)
                weight = np.random.normal(75, 15)
                height = np.random.normal(160, 8)
                bmi = weight / ((height/100) ** 2) + np.random.normal(2, 1)
                # Cycle length from 1 day to very long cycles
                cycle_length = np.random.choice([np.random.normal(15, 5), np.random.normal(45, 15)], p=[0.2, 0.8])
                waist_hip_ratio = np.random.normal(0.85, 0.05)
                skin_darkening = np.random.choice([0, 1], p=[0.2, 0.8])
                hair_growth = np.random.choice([0, 1], p=[0.1, 0.9])
                weight_gain = np.random.choice([0, 1], p=[0.1, 0.9])
                pimples = np.random.choice([0, 1], p=[0.2, 0.8])
                fast_food = np.random.choice([0, 1], p=[0.3, 0.7])
                regular_exercise = np.random.choice([0, 1], p=[0.8, 0.2])
                mood_swings = np.random.choice([0, 1], p=[0.1, 0.9])
                family_history = np.random.choice([0, 1], p=[0.3, 0.7])
                sleep_hours = np.random.normal(6, 1.5)
                stress_level = np.random.normal(7, 2)
                # Blood pressure from 80 mmHg
                blood_pressure = np.random.normal(130, 15)
            else:
                # Non-PCOS cases
                age = np.random.normal(25, 5)
                weight = np.random.normal(60, 10)
                height = np.random.normal(162, 7)
                bmi = weight / ((height/100) ** 2)
                # Normal cycle length
                cycle_length = np.random.normal(28, 3)
                waist_hip_ratio = np.random.normal(0.75, 0.05)
                skin_darkening = np.random.choice([0, 1], p=[0.9, 0.1])
                hair_growth = np.random.choice([0, 1], p=[0.9, 0.1])
                weight_gain = np.random.choice([0, 1], p=[0.8, 0.2])
                pimples = np.random.choice([0, 1], p=[0.7, 0.3])
                fast_food = np.random.choice([0, 1], p=[0.6, 0.4])
                regular_exercise = np.random.choice([0, 1], p=[0.3, 0.7])
                mood_swings = np.random.choice([0, 1], p=[0.7, 0.3])
                family_history = np.random.choice([0, 1], p=[0.8, 0.2])
                sleep_hours = np.random.normal(7.5, 1)
                stress_level = np.random.normal(4, 2)
                # Normal blood pressure
                blood_pressure = np.random.normal(120, 10)
            
            sample = {
                'age': max(18, min(45, age)),
                'weight': max(40, min(120, weight)),
                'height': max(140, min(180, height)),
                'bmi': max(15, min(40, bmi)),
                'cycle_length': max(1, min(90, cycle_length)),  # From 1 day
                'waist_hip_ratio': max(0.6, min(1.0, waist_hip_ratio)),
                'skin_darkening': skin_darkening,
                'hair_growth': hair_growth,
                'weight_gain': weight_gain,
                'pimples': pimples,
                'fast_food': fast_food,
                'regular_exercise': regular_exercise,
                'mood_swings': mood_swings,
                'family_history': family_history,
                'sleep_hours': max(4, min(12, sleep_hours)),
                'stress_level': max(1, min(10, stress_level)),
                'blood_pressure': max(80, min(180, blood_pressure)),  # From 80 mmHg
                'pcos': has_pcos
            }
            data.append(sample)
        return pd.DataFrame(data)

    def train_models(self):
        try:
            df = self.generate_synthetic_data(3000)
            X = df[self.feature_names]
            y = df['pcos']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train multiple models
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.svm import SVC
            from sklearn.linear_model import LogisticRegression
            from sklearn.neighbors import KNeighborsClassifier
            
            models_to_train = {
                'Random Forest': RandomForestClassifier(
                    n_estimators=100, max_depth=10, min_samples_split=5,
                    min_samples_leaf=2, random_state=42, class_weight='balanced'
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
                ),
                'SVM': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced', C=1.0, gamma='scale'),
                'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', C=0.1, max_iter=1000),
                'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7)
            }
            
            print("🧠 Training Machine Learning Models...")
            print("=" * 50)
            
            for model_name, model in models_to_train.items():
                # Create separate scaler for each model
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                self.models[model_name] = model
                self.scalers[model_name] = scaler
                
                print(f"✅ {model_name} trained with accuracy: {accuracy:.3f}")
                
                # Update best model
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_model = model_name
            
            print("=" * 50)
            print(f"🏆 BEST MODEL SELECTED: {self.best_model}")
            print(f"🎯 BEST ACCURACY: {self.best_accuracy:.3f}")
            print("=" * 50)
            
            # Save the best model and scaler
            joblib.dump(self.models[self.best_model], 'pcos_best_model.pkl')
            joblib.dump(self.scalers[self.best_model], 'pcos_best_scaler.pkl')
            
            app.logger.info(f"Models trained. Best: {self.best_model} with accuracy: {self.best_accuracy:.3f}")
            
        except Exception as e:
            app.logger.error(f"Model training failed: {str(e)}")
            raise

    def predict(self, features):
        if not prediction_circuit_breaker.can_execute():
            app.logger.warning("Prediction service temporarily unavailable (circuit breaker open)")
            raise Exception("Prediction service temporarily unavailable. Please try again later.")
        
        try:
            if not self.models:
                try:
                    # Load the best model
                    self.best_model = 'SVM'  # Default to SVM
                    self.models[self.best_model] = joblib.load('pcos_best_model.pkl')
                    self.scalers[self.best_model] = joblib.load('pcos_best_scaler.pkl')
                except:
                    self.train_models()
            
            feature_df = pd.DataFrame([features], columns=self.feature_names)
            features_scaled = self.scalers[self.best_model].transform(feature_df)
            prediction = self.models[self.best_model].predict(features_scaled)[0]
            probability = self.models[self.best_model].predict_proba(features_scaled)[0]
            
            prediction_circuit_breaker.record_success()
            return {
                'prediction': int(prediction),
                'probability': {
                    'no_pcos': float(probability[0]),
                    'pcos': float(probability[1])
                },
                'confidence': float(max(probability)),
                'model_accuracy': self.best_accuracy,
                'model_used': self.best_model
            }
        except Exception as e:
            prediction_circuit_breaker.record_failure()
            app.logger.error(f"Prediction failed: {str(e)}")
            raise

# Initialize ML model
pcos_predictor = PCOSPredictor()

# -----------------------------
# Enhanced Utility Functions
# -----------------------------
def generate_verification_code(length=6):
    return ''.join(random.choices(string.digits, k=length))

def send_verification_email(email, code):
    if not email_circuit_breaker.can_execute():
        app.logger.warning("Email service temporarily unavailable (circuit breaker open)")
        return False
    
    try:
        msg = Message(
            "PCOS Care - Email Verification Code",
            recipients=[email]
        )
        msg.body = f"""
        Thank you for contacting PCOS Care!
        
        Your verification code is: {code}
        
        Please enter this code on our website to verify your email address.
        
        This code will expire in 10 minutes.
        
        If you didn't request this verification, please ignore this email.
        
        Best regards,
        PCOS Care Team
        """
        
        msg.html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #4F46E5;">PCOS Care - Email Verification</h2>
            <p>Thank you for contacting PCOS Care!</p>
            <div style="background: #f8fafc; padding: 20px; border-radius: 8px; text-align: center; margin: 20px 0;">
                <h3 style="color: #1e293b; margin: 0;">Your Verification Code</h3>
                <div style="font-size: 32px; font-weight: bold; color: #4F46E5; letter-spacing: 8px; margin: 15px 0;">
                    {code}
                </div>
            </div>
            <p>Please enter this code on our website to verify your email address.</p>
            <p><small>This code will expire in 10 minutes.</small></p>
            <p>If you didn't request this verification, please ignore this email.</p>
            <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 20px 0;">
            <p style="color: #64748b;">Best regards,<br>PCOS Care Team</p>
        </div>
        """
        
        mail.send(msg)
        email_circuit_breaker.record_success()
        app.logger.info(f"Verification email sent to {email}")
        return True
    except Exception as e:
        email_circuit_breaker.record_failure()
        app.logger.error(f"Email sending failed for {email}: {str(e)}")
        return False

def send_contact_notification(name, email, subject, message):
    try:
        msg = Message(
            f"New Contact Form Message: {subject}",
            recipients=["admin@pcoscare.com"]
        )
        msg.body = f"""
        New contact form submission:
        
        Name: {name}
        Email: {email}
        Subject: {subject}
        Message: {message}
        
        Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
        """
        
        msg.html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #4F46E5;">New Contact Form Submission</h2>
            <div style="background: #f8fafc; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="color: #1e293b; margin: 0 0 10px 0;">Contact Details</h3>
                <p><strong>Name:</strong> {name}</p>
                <p><strong>Email:</strong> {email}</p>
                <p><strong>Subject:</strong> {subject}</p>
                <p><strong>Message:</strong></p>
                <div style="background: white; padding: 15px; border-radius: 5px; border-left: 4px solid #4F46E5;">
                    {message}
                </div>
                <p style="color: #64748b; margin-top: 15px;">
                    <small>Received: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</small>
                </p>
            </div>
        </div>
        """
        
        mail.send(msg)
        app.logger.info(f"Contact notification sent for {email}")
        return True
    except Exception as e:
        app.logger.error(f"Contact notification email failed: {str(e)}")
        return False

# -----------------------------
# Enhanced Risk Analysis Functions
# -----------------------------
def analyze_risk_factors(features):
    risk_factors = []
    
    if features['bmi'] >= 30:
        risk_factors.append("High BMI (Obesity)")
    elif features['bmi'] >= 25:
        risk_factors.append("Overweight")
        
    if features['cycle_length'] > 35 or features['cycle_length'] < 21:
        risk_factors.append("Irregular menstrual cycle")
        
    if features.get('waist_hip_ratio', 0) > 0.85:
        risk_factors.append("High waist-hip ratio (android obesity)")
        
    if features.get('family_history', 0) == 1:
        risk_factors.append("Family history of PCOS")
        
    if features.get('sleep_hours', 0) < 6:
        risk_factors.append("Insufficient sleep")
        
    if features.get('stress_level', 0) >= 7:
        risk_factors.append("High stress levels")
        
    if features.get('blood_pressure', 0) >= 130:
        risk_factors.append("Elevated blood pressure")
        
    symptom_count = features['skin_darkening'] + features['hair_growth'] + features['weight_gain'] + features['pimples']
    if symptom_count >= 3:
        risk_factors.append("Multiple PCOS symptoms")
        
    return risk_factors

# -----------------------------
# Enhanced Routes with Performance Monitoring
# -----------------------------

# ---------------- HEALTH CHECK ----------------
@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'version': '1.2.0',
        'services': {
            'database': 'connected' if db.users.count_documents({}) >= 0 else 'disconnected',
            'model': 'loaded' if pcos_predictor.models else 'unloaded',
            'email': 'available' if email_circuit_breaker.state == 'CLOSED' else 'degraded'
        }
    }
    return jsonify(status)

# ---------------- PERFORMANCE METRICS ----------------
@app.route('/api/performance/metrics')
@admin_required
def performance_metrics():
    """Endpoint to get performance metrics"""
    metrics = {
        'prediction_circuit_breaker': {
            'state': prediction_circuit_breaker.state,
            'failures': prediction_circuit_breaker.failures,
            'last_failure': prediction_circuit_breaker.last_failure_time
        },
        'email_circuit_breaker': {
            'state': email_circuit_breaker.state,
            'failures': email_circuit_breaker.failures,
            'last_failure': email_circuit_breaker.last_failure_time
        },
        'model_accuracy': pcos_predictor.best_accuracy,
        'best_model': pcos_predictor.best_model,
        'total_users': db.users.count_documents({}),
        'total_predictions': db.predictions.count_documents({}),
        'total_contacts': db.contacts.count_documents({}),
        'uptime': time.time() - app.start_time if hasattr(app, 'start_time') else 0
    }
    return jsonify(metrics)

# ---------------- REGISTER ----------------
@app.route('/register', methods=['POST'])
@monitor_performance
def register():
    try:
        data = request.get_json()
        app.logger.info(f"Registration attempt for: {data.get('email')}")
        
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        age = data.get('age')
        phone = data.get('phone')
        gender = data.get('gender')
        medical_history = data.get('medical_history')
        preferences = data.get('preferences', {})

        # Enhanced validation
        if not all([name, email, password]):
            return jsonify({'success': False, 'message': 'Name, email, and password are required!'}), 400

        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters!'}), 400

        # Check if user already exists
        existing_user = db.users.find_one({'email': email})
        if existing_user:
            app.logger.warning(f"Registration failed - email already exists: {email}")
            return jsonify({'success': False, 'message': 'Email already registered!'}), 400

        # Create user document with enhanced fields
        user_data = {
            'name': name,
            'email': email,
            'password': bcrypt.generate_password_hash(password).decode('utf-8'),
            'age': age,
            'phone': phone,
            'gender': gender,
            'medical_history': medical_history,
            'preferences': preferences,
            'symptom_tracking_enabled': False,
            'created_at': datetime.now(timezone.utc),
            'last_login': datetime.now(timezone.utc),
            'is_active': True
        }

        # Insert user into database
        result = db.users.insert_one(user_data)
        app.logger.info(f"User registered successfully: {email}")

        # Create user object for login
        user_obj = User(user_data)
        user_obj.id = str(result.inserted_id)
        
        # Log the user in
        login_user(user_obj)
        
        return jsonify({
            'success': True, 
            'message': f'Welcome {name}! Registration successful!',
            'user': user_obj.to_dict()
        })
        
    except Exception as e:
        app.logger.error(f"Registration error: {str(e)}")
        return jsonify({'success': False, 'message': f'Registration failed: {str(e)}'}), 500

# ---------------- LOGIN ----------------
@app.route('/login', methods=['POST'])
@monitor_performance
def login():
    try:
        data = request.get_json()
        app.logger.info(f"Login attempt for: {data.get('email')}")
        
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password are required!'}), 400

        # Find user
        user_data = db.users.find_one({'email': email})
        if not user_data:
            app.logger.warning(f"Login failed - user not found: {email}")
            return jsonify({'success': False, 'message': 'Invalid email or password!'}), 401

        # Check password
        if not bcrypt.check_password_hash(user_data['password'], password):
            app.logger.warning(f"Login failed - invalid password for: {email}")
            return jsonify({'success': False, 'message': 'Invalid email or password!'}), 401

        # Update last login
        db.users.update_one(
            {'_id': user_data['_id']},
            {'$set': {'last_login': datetime.now(timezone.utc)}}
        )

        # Create user object and log in
        user_obj = User(user_data)
        login_user(user_obj)
        
        app.logger.info(f"Login successful for: {user_data['name']}")
        return jsonify({
            'success': True, 
            'message': f'Welcome back, {user_data["name"]}! Login successful!',
            'user': user_obj.to_dict()
        })
        
    except Exception as e:
        app.logger.error(f"Login error: {str(e)}")
        return jsonify({'success': False, 'message': f'Login failed: {str(e)}'}), 500

# ---------------- LOGOUT ----------------
@app.route('/logout', methods=['POST'])
def logout():
    logout_user()
    app.logger.info("User logged out")
    return jsonify({'success': True, 'message': 'Logged out successfully'})

# ---------------- USER PROFILE ----------------
@app.route('/user/profile')
@login_required
@monitor_performance
def user_profile():
    try:
        user_data = db.users.find_one({"_id": ObjectId(current_user.id)})
        if user_data:
            return jsonify({
                'success': True,
                'user': {
                    'name': user_data['name'],
                    'email': user_data['email'],
                    'age': user_data.get('age'),
                    'phone': user_data.get('phone'),
                    'gender': user_data.get('gender'),
                    'medical_history': user_data.get('medical_history'),
                    'preferences': user_data.get('preferences', {}),
                    'symptom_tracking_enabled': user_data.get('symptom_tracking_enabled', False),
                    'created_at': user_data['created_at'].strftime('%Y-%m-%d %H:%M:%S') if user_data.get('created_at') else None,
                    'last_login': user_data.get('last_login', '').strftime('%Y-%m-%d %H:%M:%S') if user_data.get('last_login') else None
                }
            })
        else:
            return jsonify({'success': False, 'message': 'User not found'}), 404
    except Exception as e:
        app.logger.error(f"Profile fetch error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ---------------- UPDATE USER PREFERENCES ----------------
@app.route('/user/preferences', methods=['PUT'])
@login_required
def update_preferences():
    try:
        data = request.get_json()
        preferences = data.get('preferences', {})
        
        db.users.update_one(
            {'_id': ObjectId(current_user.id)},
            {'$set': {'preferences': preferences}}
        )
        
        app.logger.info(f"Preferences updated for user: {current_user.email}")
        return jsonify({'success': True, 'message': 'Preferences updated successfully'})
    except Exception as e:
        app.logger.error(f"Preferences update error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ---------------- GET USER PREDICTION HISTORY ----------------
@app.route('/user/predictions')
@login_required
@monitor_performance
def user_predictions():
    try:
        predictions = list(db.predictions.find(
            {"user_id": ObjectId(current_user.id)}
        ).sort("created_at", -1).limit(10))
        
        # Convert ObjectId and datetime to strings
        for prediction in predictions:
            prediction['_id'] = str(prediction['_id'])
            prediction['user_id'] = str(prediction['user_id'])
            if prediction.get('created_at'):
                prediction['created_at'] = prediction['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({'success': True, 'predictions': predictions})
    except Exception as e:
        app.logger.error(f"Predictions fetch error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ---------------- GET USER SYMPTOMS ----------------
@app.route('/user/symptoms')
@login_required
def user_symptoms():
    try:
        symptoms = list(db.symptoms.find(
            {"user_id": ObjectId(current_user.id)}
        ).sort("date", -1).limit(30))
        
        # Convert ObjectId and datetime to strings
        for symptom in symptoms:
            symptom['_id'] = str(symptom['_id'])
            symptom['user_id'] = str(symptom['user_id'])
            if symptom.get('date'):
                symptom['date'] = symptom['date'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({'success': True, 'symptoms': symptoms})
    except Exception as e:
        app.logger.error(f"Symptoms fetch error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ---------------- SAVE USER SYMPTOMS ----------------
@app.route('/user/symptoms', methods=['POST'])
@login_required
def save_symptoms():
    try:
        data = request.get_json()
        
        symptom_data = {
            "user_id": ObjectId(current_user.id),
            "symptoms": data.get('symptoms', {}),
            "notes": data.get('notes', ''),
            "date": datetime.now(timezone.utc)
        }
        
        db.symptoms.insert_one(symptom_data)
        app.logger.info(f"Symptoms saved for user: {current_user.email}")
        
        return jsonify({'success': True, 'message': 'Symptoms saved successfully'})
    except Exception as e:
        app.logger.error(f"Symptoms save error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ---------------- PREDICTION - UPDATED FOR 16 PARAMETERS ----------------
@app.route('/predict', methods=['POST'])
@login_required
@monitor_performance
def predict():
    try:
        data = request.get_json()
        
        # Enhanced validation
        required_fields = ['age', 'weight', 'height', 'cycle_length']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'success': False, 'message': f'Missing required field: {field}'}), 400
        
        # Extract all 16 parameters with proper defaults
        features = {
            'age': float(data.get('age', 0)),
            'weight': float(data.get('weight', 0)),
            'height': float(data.get('height', 0)),
            'bmi': float(data.get('bmi', 0)),
            'cycle_length': float(data.get('cycle_length', 0)),
            'waist_hip_ratio': float(data.get('waist_hip_ratio', 0)),
            'skin_darkening': int(data.get('skin_darkening', 0)),
            'hair_growth': int(data.get('hair_growth', 0)),
            'weight_gain': int(data.get('weight_gain', 0)),
            'pimples': int(data.get('pimples', 0)),
            'fast_food': int(data.get('fast_food', 0)),
            'regular_exercise': int(data.get('regular_exercise', 0)),
            'mood_swings': int(data.get('mood_swings', 0)),
            'family_history': int(data.get('family_history', 0)),
            'sleep_hours': float(data.get('sleep_hours', 0)),
            'stress_level': float(data.get('stress_level', 0)),
            'blood_pressure': float(data.get('blood_pressure', 0))
        }
        
        # Validate required fields
        for field in required_fields:
            if features[field] <= 0:
                return jsonify({'success': False, 'message': f'Please provide a valid value for {field}'}), 400
        
        result = pcos_predictor.predict(features)

        # Enhanced prediction result with severity assessment
        severity = "High" if result['probability']['pcos'] >= 0.7 else "Moderate" if result['probability']['pcos'] >= 0.4 else "Low"
        
        # Generate comprehensive recommendations
        recommendations = generate_recommendations(features, result['prediction'], severity)

        # Save prediction to database
        prediction_data = {
            "user_id": ObjectId(current_user.id),
            "email": current_user.email,
            "prediction_result": "PCOS Positive" if result['prediction'] == 1 else "PCOS Negative",
            "confidence_score": result['confidence'],
            "probability": result['probability'],
            "features": features,
            "risk_factors": analyze_risk_factors(features),
            "severity": severity,
            "recommendations": recommendations,
            "model_accuracy": result.get('model_accuracy', 0.0),
            "model_used": result.get('model_used', 'Unknown'),
            "created_at": datetime.now(timezone.utc)
        }
        db.predictions.insert_one(prediction_data)

        app.logger.info(f"Prediction completed for user: {current_user.email} - Result: {prediction_data['prediction_result']} - Model: {result.get('model_used', 'Unknown')}")

        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'probability': result['probability'],
            'risk_factors': analyze_risk_factors(features),
            'severity': severity,
            'recommendations': recommendations,
            'model_accuracy': result.get('model_accuracy', 0.0),
            'model_used': result.get('model_used', 'Unknown'),
            'prediction_id': str(prediction_data.get('_id', ''))
        })
    except Exception as e:
        app.logger.error(f"Prediction error for user {current_user.email}: {str(e)}")
        return jsonify({'success': False, 'message': f'Prediction failed: {str(e)}'}), 500

def generate_recommendations(features, prediction, severity):
    """Generate personalized recommendations based on user data and prediction"""
    recommendations = []
    
    if prediction == 1:  # PCOS Positive
        recommendations.append("Consult with an endocrinologist or gynecologist for proper diagnosis")
        recommendations.append("Consider getting blood tests for hormone levels and insulin resistance")
        
        if features['bmi'] >= 25:
            recommendations.append("Focus on weight management through balanced diet and regular exercise")
            recommendations.append("Aim for 5-10% weight loss to improve symptoms")
        
        if features['cycle_length'] > 35 or features['cycle_length'] < 21:
            recommendations.append("Track menstrual cycles regularly using a calendar or app")
        
        if features['stress_level'] >= 7:
            recommendations.append("Practice stress management techniques like yoga, meditation, or deep breathing")
        
        if features['sleep_hours'] < 7:
            recommendations.append("Aim for 7-9 hours of quality sleep per night")
        
        recommendations.append("Consider a low-glycemic index diet to manage insulin levels")
        recommendations.append("Regular moderate exercise (30-45 minutes, 5 times per week)")
        
    else:  # PCOS Negative
        recommendations.append("Maintain healthy lifestyle habits to prevent future health issues")
        
        if features['bmi'] >= 25:
            recommendations.append("Continue healthy eating and regular physical activity")
        
        if any([features['skin_darkening'], features['hair_growth'], features['pimples']]):
            recommendations.append("Monitor any skin or hair changes and consult dermatologist if needed")
    
    # General recommendations for all users
    recommendations.append("Stay hydrated and maintain a balanced diet rich in fruits and vegetables")
    recommendations.append("Regular health check-ups and maintaining healthy weight")
    
    return recommendations

# ---------------- GENERATE PDF REPORT ----------------
@app.route('/generate_pdf_report', methods=['POST'])
@login_required
def generate_pdf_report():
    try:
        data = request.get_json()
        prediction_id = data.get('prediction_id')
        
        # Fetch prediction data
        prediction = db.predictions.find_one({"_id": ObjectId(prediction_id)})
        if not prediction:
            return jsonify({'success': False, 'message': 'Prediction not found'}), 404
        
        # Fetch user data
        user_data = db.users.find_one({"_id": ObjectId(current_user.id)})
        
        # Generate comprehensive report data
        report_data = {
            'user': {
                'name': user_data['name'],
                'age': user_data.get('age', 'Not provided'),
                'email': user_data['email'],
                'phone': user_data.get('phone', 'Not provided')
            },
            'prediction': {
                'result': prediction['prediction_result'],
                'confidence': f"{prediction['confidence_score'] * 100:.1f}%",
                'pcos_probability': f"{prediction['probability']['pcos'] * 100:.1f}%",
                'severity': prediction.get('severity', 'Not assessed'),

                'date': prediction['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            },
            'health_metrics': prediction['features'],
            'risk_factors': prediction.get('risk_factors', []),
            'recommendations': prediction.get('recommendations', []),
            'additional_info': {
                'report_generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'report_id': str(prediction['_id']),
                'next_steps': [
                    "Consult with healthcare provider within 2 weeks",
                    "Follow up with recommended lifestyle changes",
                    "Schedule blood tests for hormone levels",
                    "Monitor symptoms regularly",
                    "Consider follow-up assessment in 3 months"
                ],
                'emergency_contacts': [
                    "Local Hospital Emergency: Available 24/7",
                    "PCOS Specialist Referral: Contact your healthcare provider",
                    "Mental Health Support: Available through healthcare provider"
                ]
            }
        }
        
        return jsonify({
            'success': True,
            'report_data': report_data,
            'message': 'Report data generated successfully'
        })
        
    except Exception as e:
        app.logger.error(f"PDF report generation error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ---------------- CONTACT FORM ----------------
@app.route('/contact', methods=['POST'])
def contact():
    try:
        data = request.get_json()
        
        name = data.get('name')
        email = data.get('email')
        subject = data.get('subject')
        message = data.get('message')

        if not all([name, email, subject, message]):
            return jsonify({'success': False, 'message': 'All fields are required!'}), 400

        # Save contact message to database
        contact_data = {
            "name": name,
            "email": email,
            "subject": subject,
            "message": message,
            "created_at": datetime.now(timezone.utc),
            "responded": False
        }
        db.contacts.insert_one(contact_data)

        # Send notification email
        send_contact_notification(name, email, subject, message)

        app.logger.info(f"Contact form submitted by: {name} ({email})")
        return jsonify({'success': True, 'message': 'Message sent successfully!'})
        
    except Exception as e:
        app.logger.error(f"Contact form error: {str(e)}")
        return jsonify({'success': False, 'message': 'Failed to send message. Please try again.'}), 500

# ---------------- SUPPORT RESOURCES ----------------
@app.route('/api/support/resources')
def get_support_resources():
    """Enhanced support resources with detailed information"""
    resources = {
        'success': True,
        'resources': {
            'emergency_contacts': [
                {
                    'name': 'National Suicide Prevention Lifeline',
                    'phone': '1-800-273-8255',
                    'description': '24/7 free and confidential support for people in distress'
                },
                {
                    'name': 'Crisis Text Line',
                    'phone': 'Text HOME to 741741',
                    'description': 'Free 24/7 crisis support via text message'
                },
                {
                    'name': 'National Alliance on Mental Illness (NAMI)',
                    'phone': '1-800-950-6264',
                    'description': 'Mental health support and resources'
                }
            ],
            'support_groups': [
                {
                    'name': 'PCOS Support Group - Online Community',
                    'description': 'Largest online PCOS support community with 50,000+ members',
                    'link': 'https://www.pcosupport.org',
                    'meeting_times': '24/7 Online Forum'
                },
                {
                    'name': 'PCOS Challenge: The National PCOS Association',
                    'description': 'Comprehensive support including local meetups and online resources',
                    'link': 'https://pcoschallenge.org',
                    'meeting_times': 'Monthly virtual meetings'
                },
                {
                    'name': 'MyPCOS Team - Mobile App Community',
                    'description': 'Mobile app with symptom tracking and community support',
                    'link': 'https://www.mypcosteam.com',
                    'meeting_times': 'Daily peer support'
                }
            ],
            'mental_health_resources': [
                {
                    'name': 'PCOS and Mental Health Guide',
                    'description': 'Comprehensive guide to managing mental health with PCOS',
                    'link': '/mental-health-guide',
                    'type': 'Educational Resource'
                },
                {
                    'name': 'Mindfulness and PCOS',
                    'description': 'Meditation and mindfulness techniques for PCOS symptom management',
                    'link': '/mindfulness',
                    'type': 'Therapeutic Resource'
                },
                {
                    'name': 'PCOS Nutritionist Directory',
                    'description': 'Find certified nutritionists specializing in PCOS',
                    'link': '/nutritionists',
                    'type': 'Professional Directory'
                }
            ],
            'educational_materials': [
                {
                    'title': 'Understanding PCOS: A Comprehensive Guide',
                    'description': 'Detailed information about PCOS symptoms, diagnosis, and treatment',
                    'type': 'eBook',
                    'pages': 45,
                    'download_url': '/guides/understanding-pcos'
                },
                {
                    'title': 'PCOS Diet and Nutrition Handbook',
                    'description': 'Evidence-based dietary recommendations for PCOS management',
                    'type': 'PDF Guide',
                    'pages': 32,
                    'download_url': '/guides/pcos-nutrition'
                },
                {
                    'title': 'Exercise and PCOS: The Complete Guide',
                    'description': 'Tailored exercise routines and fitness plans for PCOS',
                    'type': 'Video Series',
                    'duration': '2.5 hours',
                    'access_url': '/guides/pcos-exercise'
                }
            ],
            'professional_help': [
                {
                    'specialty': 'Endocrinologists',
                    'description': 'Hormone specialists for PCOS diagnosis and treatment',
                    'finding_tips': 'Look for endocrinologists with reproductive endocrine experience'
                },
                {
                    'specialty': 'Gynecologists',
                    'description': 'Women\'s health specialists for menstrual and reproductive issues',
                    'finding_tips': 'Seek gynecologists experienced in PCOS management'
                },
                {
                    'specialty': 'Registered Dietitians',
                    'description': 'Nutrition experts for PCOS-specific dietary plans',
                    'finding_tips': 'Look for dietitians with PCOS or metabolic disorder experience'
                },
                {
                    'specialty': 'Mental Health Therapists',
                    'description': 'Therapists specializing in chronic illness and women\'s health',
                    'finding_tips': 'Find therapists experienced with PCOS-related anxiety and depression'
                }
            ]
        }
    }
    return jsonify(resources)

# ---------------- OTHER ROUTES ----------------
@app.route('/')
@monitor_performance
def index():
    return render_template('index.html')

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/treatment')
def treatment():
    return render_template('treatment.html')

@app.route('/lifestyle')
def lifestyle():
    return render_template('lifestyle.html')

@app.route('/support')
def support():
    return render_template('support.html')

@app.route('/explore')
def explore():
    return render_template('explore.html')

@app.route('/offline')
def offline():
    return render_template('offline.html')

# -----------------------------
# Error Handlers
# -----------------------------

@app.errorhandler(404)
def not_found_error(error):
    app.logger.warning(f'404 error: {request.url}')
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'message': 'Endpoint not found'}), 404
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'500 error: {str(error)}')
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'message': 'Internal server error'}), 500
    return render_template('500.html'), 500

@app.errorhandler(413)
def too_large(error):
    app.logger.warning(f'413 error: File too large')
    return jsonify({'success': False, 'message': 'File too large'}), 413

# Development: prevent aggressive caching so updated templates/static are always fetched
@app.after_request
def add_dev_headers(response):
    try:
        # Apply no-cache headers for HTML and static assets during development
        if app.debug or os.environ.get('FLASK_ENV', '') in ('development', 'dev'):
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
    except Exception:
        pass
    return response

# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    # Record startup time for performance monitoring
    app.start_time = time.time()
    
    # Create admin user if not exists
    admin_user = db.users.find_one({"email": "admin@pcoscare.com"})
    if not admin_user:
        hashed_pw = bcrypt.generate_password_hash("admin123").decode('utf-8')
        db.users.insert_one({
            "name": "Admin User",
            "email": "admin@pcoscare.com",
            "password": hashed_pw,
            "created_at": datetime.now(timezone.utc),
            "last_login": datetime.now(timezone.utc),
            "is_active": True,
            "role": "admin",
            "preferences": {"theme": "light", "notifications": True}
        })
        app.logger.info("Admin user created: admin@pcoscare.com / admin123")
    
    # Create indexes for better performance
    try:
        db.users.create_index("email", unique=True)
        db.predictions.create_index("user_id")
        db.predictions.create_index("created_at")
        db.symptoms.create_index([("user_id", 1), ("date", -1)])
        db.contacts.create_index("created_at")
        app.logger.info("Database indexes created successfully")
    except Exception as e:
        app.logger.warning(f"Index creation warning: {str(e)}")
    
    app.logger.info("PCOS Care application starting...")
    app.run(debug=True, port=5000)