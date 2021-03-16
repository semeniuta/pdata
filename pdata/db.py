import pandas as pd
from sqlalchemy import create_engine
import sqlite3


def create_mysql_engine(host, user, password, db):

    url_template = 'mysql+mysqlconnector://{}:{}@{}/{}?charset=utf8'
    conn_str = url_template.format(user, password, host, db)
    
    return create_engine(conn_str, echo=False)


class MySQLConnection:

    def __init__(self, host, user, password, db):

        self.engine = create_mysql_engine(host, user, password, db)
        self.conn = self.engine.raw_connection()

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
