"""Local, atomic snapshots of application-created fitted runs (never uploaded pickle files)."""
import logging
import os
import re
import tempfile
from pathlib import Path
import joblib


class RunStore(dict):
    def __init__(self, directory):
        super().__init__()
        self.directory=Path(directory)

    def _path(self, key):
        if not isinstance(key,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}',key):
            return None
        return self.directory / f'{key}.joblib'

    def __setitem__(self,key,value):
        path=self._path(key)
        if path is None: raise ValueError('Некорректный идентификатор запуска.')
        self.directory.mkdir(parents=True,exist_ok=True)
        descriptor,temporary=tempfile.mkstemp(prefix='.run-',dir=self.directory)
        os.close(descriptor)
        try:
            joblib.dump(value,temporary,compress=3)
            os.replace(temporary,path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
        super().__setitem__(key,value)

    def get(self,key,default=None):
        if key in self: return super().get(key,default)
        path=self._path(key)
        if path is None or not path.is_file(): return default
        try:
            result=joblib.load(path)
        except Exception:
            logging.getLogger(__name__).exception('Cannot load saved run %s',key)
            return default
        super().__setitem__(key,result)
        return result
