from .get_db import get_database

db = get_database()
db.connect()

inserted = db.insert_from_csv("/app/penguins_size.csv")
print("Inserted:", inserted)

db.close()