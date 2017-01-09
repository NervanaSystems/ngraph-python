import tempfile
import shutil


class TempDir():
    def __enter__(self):
        self.tmpdir = tempfile.mkdtemp()
        return self.tmpdir

    def __exit__(self, type, value, traceback):
        shutil.rmtree(self.tmpdir)
