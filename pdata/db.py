import pandas as pd
from sqlalchemy import create_engine
import sqlite3


def comma_separated_string(values):
    return ', '.join(values)


def create_mysql_engine(host, user, password, db):

    url_template = 'mysql+mysqlconnector://{}:{}@{}/{}?charset=utf8'
    conn_str = url_template.format(user, password, host, db)
    
    return create_engine(conn_str, echo=False)


def read_sql_with_elapsed_time(query, conn):

    t0 = time.perf_counter()
    df = pd.read_sql(query, conn)
    t1 = time.perf_counter()

    return df, t1 - t0


class SQLiteDatabase:

    def __init__(self, db_path):
        self.db_path = db_path

    def query(self, q):

        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(q, conn)
        conn.close()

        return df


class SQLSelectQuery:

    class NoNameException(Exception):
        def __init__(self):
            super().__init__('Query has not been given a name')

    def __init__(self, what, relations, conditions=None):
        
        self.what = what
        self.relations = relations
        self.conditions = conditions
        
        self.order_cols = None
        self.order_asc = None

        self.limit_n = None

        self.name = None
        
    def order_by(self, colnames, asc=True):
        
        self.order_cols = colnames
        self.order_asc = asc

        return self

    def set_name(self, name):
        self.name = name    
        return self

    def limit(self, n):
        self.limit_n = n
        return self

    def as_relation(self):
        self._check_no_name()
        return '({}) {}'.format(str(self), self.name)

    def as_named_value(self):
        self._check_no_name()
        return '({}) as {}'.format(tr(self), self.name)

    def as_value(self):
        return '({})'.format(str(self))

    def _check_no_name(self):
        if self.name is None:
            raise SQLSelectQuery.NoNameException()

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

        if self.limit_n is not None:
            q += ' LIMIT {:d}'.format(self.limit_n)

        return q