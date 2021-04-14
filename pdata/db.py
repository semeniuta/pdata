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

    def __init__(self, what, relations, conditions=None):
        
        self.what = what
        self.relations = relations
        self.conditions = conditions
        
        self.order_cols = None
        self.order_asc = None

        self.limit = None

        self.name = None
        
    def order_by(self, colnames, asc=True):
        
        self.order_cols = colnames
        self.order_asc = asc

        return self

    def set_name(self, name):
        self.name = name    
        return self


    def __str__(self):

        what_s = comma_separated_string(self.what)
        relations_s = comma_separated_string(self.relations)

        q = 'SELECT {} FROM {}'.format(what_s, relations_s)

        if self.conditions is not None:
            
            conditions_s = comma_separated_string(self.conditions)
            q += ' WHERE {}'.format(conditions_s)

        if self.order_cols is not None:

            order_cols_s = comma_separated_string(self.order_cols)
            order = 'ASC' if self.order_asc else 'DESC'
            
            q += ' ORDER BY {} {}'.format(order_cols_s, order)

        if self.name is not None:
            q = '({}) {}'.format(q, self.name)

        return q