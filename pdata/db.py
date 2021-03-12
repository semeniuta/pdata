import pandas as pd
from sqlalchemy import create_engine


class MySQLConnection:

    def __init__(self, host, user, password, db):

        conn_str = 'mysql+mysqlconnector://{}:{}@{}/{}?charset=utf8'.format(user, password, host, db)
        engine = create_engine(conn_str, echo=False)

        self.conn = engine.raw_connection()

    def query(self, q):

        return pd.read_sql(q, self.conn)

    

