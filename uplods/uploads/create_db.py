# -*- coding: utf-8 -*-
import sys, os
sys.stdout.reconfigure(encoding="utf-8")

from pymongo import MongoClient, ASCENDING, DESCENDING
from dotenv import load_dotenv
from datetime import datetime, timezone
from werkzeug.security import generate_password_hash

load_dotenv()

MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME   = os.environ.get("DB_NAME", "nextvision")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable is required")

client = MongoClient(MONGO_URI)
db     = client[DB_NAME]

print(f"[NexVision] Connected to MongoDB Atlas - database: '{DB_NAME}'")

# Drop and recreate collections
for col in ["users", "detections", "object_stats", "feedback"]:
    db[col].drop()
    print(f"  [OK] Dropped collection: {col}")

db.create_collection("users")
db.create_collection("detections")
db.create_collection("object_stats")
db.create_collection("feedback")
print("[NexVision] Collections created: users, detections, object_stats, feedback")

# Indexes
db.users.create_index("username", unique=True)
db.users.create_index("email",    unique=True)
db.detections.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
db.detections.create_index("media_type")
db.object_stats.create_index([("user_id", ASCENDING), ("object_name", ASCENDING)], unique=True)
db.object_stats.create_index([("detect_count", DESCENDING)])
db.feedback.create_index("detection_id")
db.feedback.create_index("user_id")
print("[NexVision] All indexes created")

# Schema validators
db.command("collMod", "users", validator={
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["username", "email", "password"],
        "properties": {
            "username":   {"bsonType": "string"},
            "email":      {"bsonType": "string"},
            "password":   {"bsonType": "string"},
            "avatar":     {"bsonType": "string"},
            "created_at": {"bsonType": "date"}
        }
    }
})

db.command("collMod", "detections", validator={
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["user_id", "media_type"],
        "properties": {
            "user_id":           {"bsonType": "string"},
            "original_filename": {"bsonType": "string"},
            "result_filename":   {"bsonType": "string"},
            "objects_json":      {"bsonType": "object"},
            "total_objects":     {"bsonType": "int"},
            "unique_objects":    {"bsonType": "int"},
            "media_type":        {"enum": ["image", "video", "webcam"]},
            "confidence":        {"bsonType": "double"},
            "processing_time":   {"bsonType": "double"},
            "created_at":        {"bsonType": "date"}
        }
    }
})
print("[NexVision] Schema validators applied")

# Sample data
user = db.users.insert_one({
    "username":   "demo",
    "email":      "demo@nextvision.com",
    "password":   generate_password_hash("demo1234"),
    "avatar":     "D",
    "created_at": datetime.now(timezone.utc)
})
uid = str(user.inserted_id)
print(f"[NexVision] Demo user created - id: {uid}")

det = db.detections.insert_one({
    "user_id":           uid,
    "original_filename": "sample.jpg",
    "result_filename":   "result_sample.jpg",
    "objects_json":      {"person": 2, "car": 1, "dog": 1},
    "total_objects":     4,
    "unique_objects":    3,
    "media_type":        "image",
    "confidence":        0.45,
    "processing_time":   1.23,
    "created_at":        datetime.now(timezone.utc)
})
did = str(det.inserted_id)
print(f"[NexVision] Sample detection created - id: {did}")

for obj, cnt in {"person": 2, "car": 1, "dog": 1}.items():
    db.object_stats.update_one(
        {"user_id": uid, "object_name": obj},
        {"$inc": {"detect_count": cnt}, "$set": {"last_seen": datetime.now(timezone.utc)}},
        upsert=True
    )
print("[NexVision] Sample object_stats created")

db.feedback.insert_one({
    "user_id":      uid,
    "detection_id": did,
    "rating":       5,
    "comment":      "Great detection!",
    "created_at":   datetime.now(timezone.utc)
})
print("[NexVision] Sample feedback created")

# Summary
print("\n" + "="*50)
print(f"  Database : {DB_NAME}")
print(f"  Cluster  : cluster0.oeui19a.mongodb.net")
for col in ["users", "detections", "object_stats", "feedback"]:
    count = db[col].count_documents({})
    print(f"  {col:<20} : {count} document(s)")
print("="*50)
print("[NexVision] Database setup complete!")

client.close()
