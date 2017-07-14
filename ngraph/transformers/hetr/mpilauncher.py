import os
import subprocess
import time

STARTUP_TIME = 0.5


class Launcher(object):
    """
    execute mpirun cmd
    close process for mpirun
    """
    def __init__(self, ports=None):
        self.port_list = ports if ports else\
            ['52051', '52052', '52053', '52054', '52055', '52056', '52057', '52058']
        self.mpirun_proc = None

    def launch(self):
        hetr_server_path = os.path.dirname(os.path.realpath(__file__)) + "/hetr_server.py"
        hetr_server_num = os.getenv('HETR_SERVER_NUM')
        hetr_server_gpu_num = os.getenv('HETR_SERVER_GPU_NUM')
        hetr_server_hostfile = os.getenv('HETR_SERVER_HOSTFILE')

        if (hetr_server_num is not None) & (hetr_server_hostfile is not None):
            # Assumption is that hydra_persist processes are started on remote nodes
            # Otherwise, remove "-bootstrap persist" from the command line (it then uses ssh)
            mpirun_str = "mpirun -n %s -ppn 1 -bootstrap persist -hostfile %s %s"\
                % (hetr_server_num, hetr_server_hostfile, hetr_server_path)
            subprocess.call(mpirun_str, shell=True)
        elif (hetr_server_gpu_num is not None):
            cmd = ['mpirun',
                   '-n', hetr_server_gpu_num,
                   'python', hetr_server_path,
                   '-p'] + self.port_list
            try:
                self.mpirun_proc = subprocess.Popen(cmd)
                time.sleep(STARTUP_TIME)
            except:
                raise RuntimeError("Process launch failed!")

    def close(self):
        if self.mpirun_proc:
            self.mpirun_proc.terminate()
            time.sleep(STARTUP_TIME)
            if self.mpirun_proc:
                self.mpirun_proc.kill()
                self.mpirun_proc.wait()
