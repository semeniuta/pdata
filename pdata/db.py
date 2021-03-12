import pandas as pd
from sqlalchemy import create_engine
import sqlite3


class MySQLConnection:

    def __init__(self, host, user, password, db):

        conn_str = 'mysql+mysqlconnector://{}:{}@{}/{}?charset=utf8'.format(user, password, host, db)
        engine = create_engine(conn_str, echo=False)

        self.conn = engine.raw_connection()

    def query(self, q):

        return pd.read_sql(q, self.conn)


class SQLiteDatabase:

    def __init__(self, db_path):
        self.db_path = db_path

    def query(self, q):

        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(q, conn)
        conn.close()

        return df
