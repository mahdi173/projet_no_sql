from pymongo import MongoClient
import os
import csv
from .utils import parse_penguin_row

class MongoDB:
    def connect(self):
        uri = os.getenv("MONGO_URI")
        self.client = MongoClient(uri)
        self.db = self.client.testdb

        # JSON schema definition
        penguin_schema = {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["penguin_id", "island", "species"],
                "properties": {
                    "penguin_id": {"bsonType": "string"},
                    "island": {"bsonType": "string"},
                    "species": {"bsonType": "string"},
                    "sex": {
                        "bsonType": ["string", "null"],
                        "enum": ["MALE", "FEMALE", None],
                    },
                    "features": {
                        "bsonType": "object",
                        "required": [
                            "bill_length_mm",
                            "bill_depth_mm",
                            "flipper_length_mm",
                            "body_mass_g"
                        ],
                        "properties": {
                            "bill_length_mm": {"bsonType": ["double", "null"]},
                            "bill_depth_mm": {"bsonType":  ["double", "null"]},
                            "flipper_length_mm": {"bsonType": ["int", "null"]},
                            "body_mass_g": {"bsonType": ["int", "null"]},
                        }
                    }
                }
            }
        }

        # Create or update collection with schema
        try:
            self.db.create_collection("penguins")
        except Exception:
            pass  # already exists

        # Apply validator
        self.db.command({
            "collMod": "penguins",
            "validator": penguin_schema,
            "validationLevel": "strict",
            "validationAction": "error"
        })

        self.collection = self.db.penguins

    def insert(self, data):
        return self.collection.insert_one(data).inserted_id

    def insert_from_csv(self, csv_path):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)

            docs = []
            for row in reader:
                docs.append(parse_penguin_row(row))

            if docs:
                self.collection.insert_many(docs)
                return len(docs)

    def find(self, query):
        return list(self.collection.find(query))

    def close(self):
        self.client.close()


def get_collection():
    mongo = MongoDB()
    mongo.connect()
    return mongo.collection
