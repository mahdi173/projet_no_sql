import os
import csv
from uuid import uuid4
from cassandra.cluster import Cluster
from .utils import parse_penguin_row

class CassandraDB:
    def connect(self):
        host = os.getenv("CASSANDRA_HOST")
        self.cluster = Cluster([host])
        self.session = self.cluster.connect()

        self.session.execute("""
            CREATE KEYSPACE IF NOT EXISTS testks
            WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};
        """)

        self.session.execute("""
            CREATE TABLE IF NOT EXISTS testks.penguins (
                island TEXT,
                species TEXT,
                penguin_id UUID,
                sex TEXT,
                bill_length_mm FLOAT,
                bill_depth_mm FLOAT,
                flipper_length_mm INT,
                body_mass_g INT,
                PRIMARY KEY ((island), species, penguin_id)
            ) WITH CLUSTERING ORDER BY (species ASC, penguin_id ASC);
        """)
        
        self.insert_stmt = self.session.prepare("""
            INSERT INTO testks.penguins (
                island, species, penguin_id, sex,
                bill_length_mm, bill_depth_mm,
                flipper_length_mm, body_mass_g
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """)


    def insert(self, data):
        self.session.execute(
            self.insert_stmt, (
                data["island"],
                data["species"],
                uuid4(),
                data["sex"],
                data["features"]["bill_length_mm"],
                data["features"]["bill_depth_mm"],
                data["features"]["flipper_length_mm"],
                data["features"]["body_mass_g"]
            )
        )

    def insert_from_csv(self, csv_path):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)

            count = 0
            for row in reader:
                peng = parse_penguin_row(row)

                self.insert(peng)
                count += 1

            return count

    def find(self, query):
        rows = self.session.execute("SELECT * FROM testks.penguins")
        return list(rows)

    def close(self):
        self.cluster.shutdown()