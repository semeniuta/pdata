import pandas as pd
from sqlalchemy import create_engine
import sqlite3


def comma_separated_string(values):
    return ', '.join(values)


def create_mysql_engine(host, user, password, db):

    url_template = 'mysql+mysqlconnector://{}:{}@{}/{}?charset=utf8'
    conn_str = url_template.format(user, password, host, db)
    
    return create_engine(conn_str, echo=False)


class SQLiteDatabase:

    def __init__(self, db_path):
        self.db_path = db_path

    def query(self, q):

        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(q, conn)
        conn.close()

        return df


class SQLSelectQuery:

    def __init__(self, what, relations, conditions, name=None):
        
        self.what = what
        self.relations = relations
        self.conditions = conditions
        self.name = name

    def __str__(self):

        what_s = comma_separated_string(self.what)
        relations_s = comma_separated_string(self.relations)
        conditions_s = comma_separated_string(self.conditions)

        q = 'SELECT {} FROM {} WHERE {}'.format(what_s, relations_s, conditions_s)

        if self.name is not None:
            q = '({}) {}'.format(q, self.name)

        return q