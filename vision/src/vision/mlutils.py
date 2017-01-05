import os,sys, signal
import os.path
import pandas as pd
import pandas.indexes.base
import tempfile
import psycopg2
import time
import pickle
import subprocess as sp
import socket
from matplotlib.backends.backend_pdf import PdfPages
import math
import urllib.parse
import multiprocessing
import impala.dbapi
import hashlib
import traceback
import pdb

import numpy as np
from scipy.special import stdtr


################################################################################
## code to launch files on your laptop
################################################################################

## find the host machine on which to launch files
try:
    if os.path.exists("/usr/bin/x2golistsessions"):
        launch_host = os.popen("x2golistsessions").read().split("|")[7]
    else:
        launch_host = os.getenv("SSH_CONNECTION").split()[0]
except:
    launch_host = None


def launch(filename, wait=False):
    """Open a file, possibly copying it to the local machine first
    with scp"""
    if launch_host is None:
        raise Exception("launch_host not defined")
    if os.uname()[0] == 'Darwin':
        if wait:
            os.system("open -W " + filename)
        else:
            os.system("open " + filename)
    elif filename.startswith('https:') or filename.startswith('http:'):
        os.system("ssh " + launch_host + " open '" + filename + "'")
    else:
        # copy the file locally
        os.system("scp '" + filename + "' '" + launch_host + ":/tmp'")
        # open it
        if wait:
            os.system("ssh " + launch_host + " open -W '/tmp/" + os.path.basename(filename) + "'")
        else:
            os.system("ssh " + launch_host + " open '/tmp/" +
                      os.path.basename(filename) + "'")


def excel(object):
    """Open the object in excel"""
    if (type(object) == np.ndarray) and (len(object.shape) == 1):
        object = pd.Series(object)
    if issubclass(type(object), pandas.indexes.base.Index):
        object = pd.Series(object)
    if issubclass(type(object), pd.core.series.Series):
        object = pd.DataFrame(object)
    if issubclass(type(object), pd.core.frame.DataFrame):
        with tempfile.NamedTemporaryFile(suffix=".xlsx") as file:
            object.to_excel(file.name)
            file.flush()
            launch(file.name)
    else:
        raise Exception("Don't know how to handle this type")

###############################################################################
# Tripadvisor data sources
###############################################################################

leo_connection = None

def leo_connect(force=False):
    global leo_connection
    if (leo_connection is None) or force:
        leo_connection = impala.dbapi.connect(
            host='leomaster01ad.hdp.tripadvisor.com',
            port=10000,
            use_ssl=False,
            auth_mechanism='PLAIN',
            user='root',
            password=None)

def query_leo(q, strip_prefix=False, maybe_retry=True):
    """Run an SQL query against Leo"""
    leo_connect()
    try:
        result = pd.read_sql_query(q, leo_connection)
        if (strip_prefix):
            def strip_col_prefix(s):
                if s.find('.'):
                    return s.split('.')[1]
                return s
            result = result.rename(columns=strip_col_prefix)
        return result
    except (socket.error, pd.io.sql.DatabaseError, impala.error.HiveServer2Error):
        if not maybe_retry:
            raise
        leo_connect(force=True)
        return query_leo(q, strip_prefix=strip_prefix, maybe_retry=False)


def strip_prefix(df):
    def strip_col_prefix(s):
        if s.find('.') >= 0:
            return s.split('.')[1]
        return s
    return df.rename(columns=strip_col_prefix)


def query_hive_cli(q, cluster="leo", queue="prodp13n"):
    """Run an sql query using the hive CLI. Shows progress and query_leo
was dying on a large query"""
    with tempfile.NamedTemporaryFile(mode='w', encoding = 'utf-8', delete=False) as f:
        f.write("""
set hive.cli.print.header=true;
set mapred.job.queue.name=""" + queue + """;
""" + q)
        f.flush()
        data = sp.Popen(os.getenv("WHTOP")
                        + '/clusters/' + cluster
                        + '/bin/hive -f ' + f.name,
                        shell=True,
                        stdout=sp.PIPE)
        result = pd.read_table(data.stdout)
        if data.wait() != 0:
            raise Exception("Error executing hive query: " + q)
        return result


def execute_hive(cmd, queue = "prodp13n"):
    with tempfile.NamedTemporaryFile(mode='w', encoding = 'utf-8', delete=False) as f:
        f.write("set mapred.job.queue.name=" + queue + ";\n")
        f.write(cmd)
        f.flush()
        status = os.system(os.getenv("WHTOP")
                           + "/clusters/leo/bin/hive -f '"
                           + f.name + "'")
        if status != 0:
            raise Exception("Error executing hive command: " + cmd)

def upload_to_leo(table_name, data_frame, maybe_retry=True):
    """upload the data_frame to a table on leo, overwriting the table if it exists"""
    leo_connect()
    def col_defn(colname):
        if data_frame[colname].dtype == np.object:
            return colname + " string"
        if data_frame[colname].dtype == np.float64:
            return colname + " double"
        raise Exception("Type not implemented for " + colname)
    cols = str.join(',', [ col_defn(name) for name in data_frame.columns])
    with tempfile.NamedTemporaryFile(mode='w', encoding = 'utf-8') as f:
        data_frame.to_csv(f, sep="\001", index=False,header=False)
        f.flush()
        try:
            cur = leo_connection.cursor()
            cur.execute("drop table if exists " + table_name)
            cur.execute("create table " + table_name + " ( " + cols + ') row format delimited fields terminated by "\001" stored as textfile');
            cur.close()
            ## we need to use the CLI here so that we can read the local tsv,
            ##   instead of trying to read the csv off the hive2 server somewhere
            execute_hive("load data local inpath \"" + f.name + "\" overwrite into table " + table_name + ";")
        except (socket.error, impala.error.HiveServer2Error):
            if not maybe_retry:
                raise
            leo_connect(force=True)
            return upload_to_leo(table_name, data_frame, maybe_retry=False)

rio_connection = None
def query_rio(q):
    """Run an SQL query against rio"""
    global rio_connection
    if rio_connection == None or rio_connection.closed:
        rio_connection = psycopg2.connect(dbname="rio",
                                          user="jpalmucci",
                                          host="rio.cta9rlboj2sy.us-east-1.redshift.amazonaws.com",
                                          password="Helpme124",
                                          port="5439")
    return pd.read_sql_query(q, rio_connection)


tripmonster_connection = None
def query_tripmonster(q):
    """Run an SQL query against tripmonster"""
    global tripmonster_connection
    if tripmonster_connection == None or tripmonster_connection.closed:
        tripmonster_connection = psycopg2.connect(dbname="tripmonster",
                                                  user="tripmonster",
                                                  host="tripmonster")
    return pd.read_sql_query(q, tripmonster_connection)

def show_partitions(table):
    """Return a data frame that shows the partitions that are currently in the table"""
    p = query_leo("show partitions " + table);
    p = p.partition.apply( lambda x: pd.Series(x.split("/")))
    df = pd.DataFrame()
    for col in p.columns:
        name = p.iloc[0,col].split('=')[0]
        df[name] = p[col].str.split("=").apply( lambda x: x[1] )
    return df


################################################################################
## making pdf plots

class pdf_plot:
    """
Make a pdf plot and open it on your laptop. Example:

   with pdf_plot("foo.pdf"):
      plt.plot([1,2,3,4,5])
"""

    def __init__(self, name):
        self.name = name
    def __enter__(self):
        plt.clf()
        self.pp = PdfPages(self.name)
        return self;
    def __exit__(self, type, value, traceback):
        plt.savefig(self.pp, format="pdf")
        self.pp.close()
        plt.clf()
        launch(self.name)


################################################################################
## memoization
################################################################################

def memoize( name, expire_in_days, thunk):
    """evaluate the thunk and return, store the result on disk under the
given name. If the result is already on disk and is less than
'expire_in_days' days old, just read the result.

If 'thunk' is iterable, evaluate each element in parallel subprocesses for speed.
    """
    try:
        os.mkdir("data")
    except:
        pass
    if name is None:
        f = tempfile.mktemp("memoize")
    else:
        f = "data/" + name + ".pickle"
    if hasattr(thunk, "__iter__"):
        ## we have a list of thunks to apply in parallel
        if os.path.exists(f + ".0"):
            days_old = (time.time() - os.path.getmtime(f + ".0")) / 60 / 60 / 24
            if days_old < expire_in_days:
                if name is not None:
                    print ("Loading " + name)
                result = [ pickle.load( open(f + "." + str(i), "rb")) for i,t in enumerate(thunk) ]
                return result

        def eval_subthunk(i):
            frk = os.fork()
            if frk == 0:
                ## make a new process group so we can kill the whole tree on an error
                os.setsid()
                try:
                    result = thunk[i]()
                    pickle.dump( result,
                                 open(f + "." + str(i), "wb"), pickle.HIGHEST_PROTOCOL)
                    os._exit(0)
                except:
                    ## error occurred, save it and return error status
                    pickle.dump( (sys.exc_info()[0], traceback.format_exc()),
                                 open(f + "." + str(i), "wb"), pickle.HIGHEST_PROTOCOL)
                    os._exit(1)
            ## only parent process gets here
            return frk

        def read_result(i,pid):
            (pid, status) = os.waitpid(pid,0)
            fname = f + "." + str(i)
            val = pickle.load(open(fname, "rb"))
            if name is None:
                os.unlink(fname)
            if status != 0:
                print( "Error in subprocess: ", val[1])
                raise val[0]
            return val
        if name is not None:
            print ("Calculating " + name)
        pids = [eval_subthunk(i) for i,t in enumerate(thunk)]
        try:
            result = [read_result(i,pid) for i,pid in enumerate(pids)]
        except (KeyboardInterrupt, Exception) as e:
            ## error waiting (or in) sub process, clean up the garbage before rethrowing
            for pid in pids:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                    os.waitpid(pid,0)
                except OSError:
                    pass
            if name is not None:
                flush_cache(name)
            raise e

    else:  ## single thunk case
        if os.path.exists(f):
            days_old = (time.time() - os.path.getmtime(f)) / 60 / 60 / 24
            if days_old < expire_in_days:
                print ("Loading " + name)
                return pickle.load(open(f, "rb"))

        print ("Calculating " + name)
        result = thunk()
        if os.fork() == 0:
            ## save using a background process so we don't have to
            ## wait for the dump to complete
            pickle.dump( result,
                         open(f, "wb"), pickle.HIGHEST_PROTOCOL)
            os._exit(0)
    return result

def flush_cache(name):
    """flus the cached values from 'memoize'"""
    for file in os.listdir("data"):
        if file.startswith(name + ".pickle"):
            os.remove("data/" + file)

subprocess_id = None

def pmap(fn, l, subprocesses = multiprocessing.cpu_count()):
    """Parallel map using memoize as a base. Split the list 'l' into
'subprocesses' chunks, evaluate each chunk in a subprocess"""
    batch_size = math.ceil(len(l) / subprocesses)
    batch_starts = range(0, len(l), batch_size)
    ## shuffle randomly to avoid hot spots
    permutation = np.random.permutation(len(l))
    shuffled = [l[i] for i in permutation]
    def batch_thunk(id, batch_start):
        def compute_batch():
            global subprocess_id
            subprocess_id = id
            return [fn(x) for x in shuffled[batch_start:batch_start+batch_size]]
        return compute_batch
    results = memoize(None, 100, [batch_thunk(id,batch_start)
                                  for (id, batch_start) in enumerate(batch_starts)])
    calculated =  [item for sublist in results for item in sublist]
    ## undo the random shuffle
    return [ calculated[i] for i in np.argsort(permutation)]


################################################################################
## mathy stuff
################################################################################

def compressed_ttest( ascore, acount, bscore, bcount):
    """A one sided t-test (alternative: b>a)
where each score has a count that tell how many times it occurred"""
    na = float(acount.sum())
    amean = (ascore * acount).sum() / na
    avar = (np.power((ascore - amean),2) * acount).sum() / (na-1)
    nb = float(bcount.sum())
    bmean = (bscore * bcount).sum() / nb
    bvar = (np.power((bscore - bmean),2) * bcount).sum() / (nb-1)
    tf = (amean - bmean) / np.sqrt(avar/na + bvar/nb)
    dof = (avar/na + bvar/nb)**2 / (avar**2/(na**2*(na-1)) + bvar**2/(nb**2*(nb-1)))
    pf = stdtr(dof, tf)
    return (tf,pf)

def aggr_t_test( sum_a, sum_a2, na, sum_b, sum_b2, nb):
    """run a t-test using just the aggragate statistics: sum_a is the sum
of the observations, sum_a2 is the sum of the squared observations, na
is the number of observations"""
    na = float(na)  ## prevent overflow
    nb = float(nb)
    amean = sum_a / na;
    avar = (sum_a2 - 2.0 * sum_a * amean + na * amean * amean) / (na-1)
    bmean = sum_b / nb;
    bvar = (sum_b2 - 2.0 * sum_b * bmean + nb * bmean * bmean) / (nb-1)
    tf = (amean - bmean) / np.sqrt(avar/na + bvar/nb)
    dof = (avar/na + bvar/nb)**2 / (avar**2/(na**2*(na-1)) + bvar**2/(nb**2*(nb-1)))
    pf = stdtr(dof, tf)
    return {"amean": amean, "bmean": bmean, "tstat":tf, "pval": pf}

def compressed_permutation_ttest( ascore, acount, bscore, bcount):
    """A one sided permutation test where the alterantive is b's mean > a's mean"""
    stat = (bscore * bcount).sum() / bcount.sum() - (ascore * acount).sum() / acount.sum()

    combined_score = np.concatenate((ascore, bscore))
    combined_count = np.concatenate((acount, bcount))

    higher = 0
    for i in xrange(1000):
        np.random.shuffle(combined_score)
        asample = (combined_score[0:len(ascore)] * combined_count[0:len(ascore)]).sum() / combined_count[0:len(ascore)].sum()
        bsample = (combined_score[len(ascore):] * combined_count[len(ascore):]).sum() / combined_count[len(ascore):].sum()
        if (bsample - asample) > stat:
            higher += 1
    return higher / 1000.0


def logit(array):
    return np.log(array/(1-array))

def inv_logit(array):
    e = np.exp(array)
    return e / (1 + e)

################################################################################

_published_locations = None

def published_locations():
    """Return a dataframe of QAable locations (as of 7 days ago. Filter out closed locations, vacation rentals, etc"""
    global _published_locations
    if _published_locations is not None:
        return _published_locations
    (published_hotels,published_others)= memoize(
        "published_locations", 7,
        [lambda:
         query_tripmonster("""
         select l.id as locationid, primaryname, placetypeid
         from t_location l
            join t_accomodation a on l.id = a.locationid
            left join t_location_closing_info c on l.id = c.locationid
         where c.locationid is null
            and a.whattype in (1,2,5,6,16,25,26,39)
         and placetypeid in (10022, 10021, 10023) and status = 4
         """).set_index("locationid"),
         lambda:
         query_tripmonster("""
         select l.id as locationid, primaryname, placetypeid
         from t_location l
            left join t_location_closing_info c on l.id = c.locationid
         where c.locationid is null
            and placetypeid in (10022, 10021) and status = 4
         """).set_index("locationid")])
    _published_locations = published_others.append(published_hotels)
    _published_locations['hyperlink'] = (
        '=HYPERLINK("http://tripadvisor.com/0'
        + _published_locations.index.map(str) + '","'
        + _published_locations.primaryname + '")')
    return _published_locations


def confluence(df):
    """Print out a data frame structured for confluence"""
    print('||', str.join('||', df.columns, "||"))
    for row in range(0, df.shape[0]-1):
        sys.stdout.write('|')
        for cell in df.iloc[row, ]:
            sys.stdout.write(str(cell).replace("|", "\\|"))
            sys.stdout.write('|')
        sys.stdout.write('\n')
    sys.stdout.flush()

################################################################################

def grab_url(url):
    """Download a given URL to the local filesystem, if we haven't
downloaded it so far. Return the name of the local file. Throw an
IOError if we could not get the file. The fact that we could not get
the file is cached too, and is represented by a zero length file in
the cache

    """
    target = url_location(url)
    if not os.path.exists(target):
        try:
            urllib.request.urlretrieve(url, target + ".part")
            os.rename(target + ".part", target)
            print("Downloaded " + url)
        except IOError:
            print("Error downloading " + url)
            open(target, "a").close()
            raise IOError("Couldn't grab " + url)
    if os.path.getsize(target) == 0:
        print("Error (previously) downloading " + url)
        raise IOError("Couldn't grab " + url)
    return target

def url_location(url, make_dir=True):
    digest = hashlib.md5( url.encode()).hexdigest()
    dir = "/spindle/webcache/" + digest[0:2] + '/' + digest[2:4] + '/' + digest[4:6]
    if make_dir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir + '/' + urllib.parse.quote_plus(url)

def have_url(url):
    location = url_location(url, make_dir=False)
    return os.path.exists(location)


def port_cache():
    """port from version 1 of the webcache layout to version 2"""
    files = os.listdir("/spindle/webcache.old")
    for (i,file) in enumerate(files):
        if (i % 1000) == 0:
            print( i, "/", len(files))
        url = urllib.parse.unquote_plus(file)
        os.rename('/spindle/webcache.old/' + file, url_location(url))

def url_is_grabbable(url):
    """check to see if we were able to grab the url"""
    try:
        grab_url(url)
        return True
    except IOError:
        return False

def condensed_to_square(k, n):
    if hasattr(k, "__iter__"):
        return [condensed_to_square(ki, n) for ki in k]

    def calc_row_idx(k, n):
        return int(math.ceil((1 / 2.) * (- (-8*k + 4 *n**2 -4*n - 7)**0.5 + 2*n - 1) - 1))

    def elem_in_i_rows(i, n):
        return i * (n - 1 - i) + (i*(i + 1))/2

    def calc_col_idx(k, i, n):
        return int(n - elem_in_i_rows(i + 1, n) + k)
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)
    return i, j

def photo_url_given_path( path ):
    if type(path) == str:
        return "https://media-cdn.tripadvisor.com/media/" + path.replace('photo-o','photo-s')
    return "https://media-cdn.tripadvisor.com/media/" + path.str.replace('^photo-o','photo-s')

def sql_group( iterable ):
    """Make an sql set expression useful for WHERE IN clauses"""
    def maybe_escape(x):
        if type(x) == str:
            return "'" + x.replace("'", "''") + "'"
        else :
            return str(x)
    return '(' + str.join(',', [ maybe_escape(x) for x in iterable]) + ')'
