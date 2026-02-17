class Database:
    def connect(self):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    def find(self, query):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError