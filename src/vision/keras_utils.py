# coding: utf-8

__author__ = "Aaron Gonzales"
__email__ = "agonzales@tripadvisor.com"


import sqlalchemy
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
import logging

from functools import wraps
from time import time

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s : %(name)s : %(message)s",
                    )
log = logging.getLogger(__name__)

vr_connections = {
    'site_data' : 'mssql+pymssql://VRYieldProcessReader:VQuattr0$R@warehousesql.'+ \
                  'tripadvisor.com:1433/Site_Data?charset=utf8',

    'vr_data' : 'mssql+pymssql://VRYieldProcessReader:VQuattr0$R@warehousesql.'+ \
                'tripadvisor.com:1433/VRData?charset=utf8',

    'hldb' : 'mysql://readonly:9XhrqwPUxSZp@mysql-slave01.hq.internalhl.com/'+ \
             'hldb?charset=utf8',

    'tripmonster':
        'postgresql://tripmonster@tripmonster.tripadvisor.com:5432/tripmonster',

    'vr_sem' : 'mssql+pymssql://VRSEM_Runner:gethGmY9@vrsql01d.d.tripadvisor.com'+ \
               ':1433/vr_sem_new?charset=utf8'

}


def get_na_cols(df):
    """
    Finds the columns with an NA present.
    Args:
        df (pandas.DataFrame) with columns to detect
    Returns:
        list of column names

    """
    nonna = set(df.dropna(axis=1).columns)
    all_cols = set(df.columns)
    return [col for col in all_cols if col not in nonna]


def make_categoricals(df, cat_feats=None):
    """
    Change pandas 'object' type to 'category' type from list in auxilary file.
    Args:
        df (pd.DataFrame): data frame to modify
        cat_feats (string): path to the categorical features file.
    """
    # this could be changed to be a Python list in a utility file.
    if cat_feats is None:
        cat_feats = './categorical_features.txt'
    with open('./categorical_features.txt', 'r') as cf:
        categories = [line.strip() for line in cf.readlines()]

    for c in categories:
        df[c] = df[c].astype('category')


def logistic(x):
    """
    Convenience for  the logistic function.
    """
    return 1 / (1 + np.exp(-x))


def get_db_conn(db):
    """
    Grabs a database connection engine from Sqlalchemy. Allows programmatic
    interfacing with our database tables.
    Args:
        db (str): String of the database you want. Currently supports
                  vr_data, site_data, vr_sem, and hdlp.
    Returns: sqlalchemy.engine
    """
    db = vr_connections[db]
    engine = sqlalchemy.create_engine(db)
    return engine


def query_database(query, db='site_data', debug=False):
    """
    Helper function to pull data from the database to a pandas dataframe.
    Args:
        query (str): the string verison of a valid sql query.
        db (str): database from which you want data.
        debug (boolean): log.infos the query before running if set to True.
    Returns:
        DataFrame of resulting query.
    """
    engine = get_db_conn(db)
    # converts a query to  structured SQL
    query = sqlalchemy.text(query)
    if debug:
        log.debug(query)
    return pd.read_sql_query(sql=query, con=engine)


def timeit(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        t = te - ts
        log.info("function {} took {:02.5f} seconds".format(f.__name__, t))
        print("function {} took {:02.5f} seconds".format(f.__name__, t))
        return result
    return wrap
