import os
from pymongo import MongoClient
from datetime import datetime

MONGO_URL = os.getenv("MONGO_URL")

client = MongoClient(MONGO_URL)

db = client["phishing_db"]
emails_collection = db["emails"]
system_collection = db["system_state"]

def store_email(data: dict):
    data["timestamp"] = datetime.utcnow()
    emails_collection.insert_one(data)