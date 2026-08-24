import joblib

print("Loading model...")
joblib.load("youtube_revenue_model.pkl")
print("Model loaded")

print("Loading category encoder...")
joblib.load("category_encoder.pkl")
print("Category loaded")

print("Loading device encoder...")
joblib.load("device_encoder.pkl")
print("Device loaded")

print("Loading country encoder...")
joblib.load("country_encoder.pkl")
print("Country loaded")