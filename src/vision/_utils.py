import errno    
import os


def get_fpath_filename(filepath):
    return filepath.split('photo-s/')[-1].replace('{','').replace('}','').replace('/','-')

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise