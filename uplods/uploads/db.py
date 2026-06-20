from pymongo import MongoClient, ASCENDING, DESCENDING
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME   = os.environ.get("DB_NAME", "nextvision")

_client = None

def get_db():
    global _client
    if not MONGO_URI:
        raise RuntimeError("MONGO_URI environment variable is required")
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client[DB_NAME]

def init_db():
    db = get_db()
    db.users.create_index("username", unique=True)
    db.users.create_index("email",    unique=True)
    db.detections.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    db.object_stats.create_index([("user_id", ASCENDING), ("object_name", ASCENDING)], unique=True)
    db.feedback.create_index("detection_id")
    print("[NexVision] MongoDB connected and indexes ready!")
