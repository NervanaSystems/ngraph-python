import os
import subprocess
import time
import logging
import signal
import tempfile
import fcntl


logger = logging.getLogger(__name__)


class MPILauncher(object):
    """
    execute mpirun cmd
    close process for mpirun
    """
    def __init__(self):
        self.mpirun_proc = None
        self._hostfile = os.getenv('HETR_SERVER_HOSTFILE')
        self._server_count = os.getenv('HETR_SERVER_NUM')
        self._rpc_ports = os.getenv('HETR_SERVER_PORTS')
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self._tmpfile = tempfile.NamedTemporaryFile(mode='r',
                                                    dir=current_dir,
                                                    delete=True)
        if self._hostfile is not None:
            self._hosts = [line.rstrip('\n') for line in open(self._hostfile, 'r')]
        else:
            self._hosts = ['localhost']

    def get_host_by_rank(self, rank):
        return self._hosts[rank % len(self._hosts)]

    def get_rpc_port_by_rank(self, rank):
        if self.mpirun_proc is None:
            raise RuntimeError("Launch mpirun_proc before reading of rpc ports")

        if self._rpc_ports is not None:
            return self._rpc_ports[rank]

        self._tmpfile.seek(0)
        while True:
            fcntl.flock(self._tmpfile, fcntl.LOCK_SH)
            line_count = sum(1 for _ in self._tmpfile)
            self._tmpfile.seek(0)
            fcntl.flock(self._tmpfile, fcntl.LOCK_UN)

            if line_count == self._server_count:
                break
            else:
                time.sleep(0.05)

        server_infos = [tuple(map(int, line.split(':'))) for line in self._tmpfile]
        server_infos = sorted(server_infos, key=lambda x: x[0])
        self._rpc_ports = [row[2] for row in server_infos]
        logger.info("get_rpc_ports: ports (in MPI rank order): %s", self._rpc_ports)
        self._tmpfile.close()
        return self._rpc_ports[rank]

    def get_address_by_rank(self, rank):
        return '{}:{}'.format(self.get_host_by_rank(rank), self.get_rpc_port_by_rank(rank))

    def launch(self, server_count):
        if self.mpirun_proc is not None:
            logger.info("mpirun_proc is already launched")
            return

        if self._server_count is None:
            self._server_count = server_count

        logger.info("mpilauncher: launch: hostfile %s, hosts %s, server_count %s, tmpfile %s",
                    self._hostfile, self._hosts, self._server_count, self._tmpfile)

        server_path = os.path.dirname(os.path.realpath(__file__)) + "/hetr_server.py"
        cmd = ['mpirun',
               '-n', str(self._server_count),
               '-ppn', '1',
               '-l']  # to print MPI rank index for each log line

        if self._hostfile is not None:
            cmd.extend(['-hostfile', self._hostfile])
        elif self._hosts is not None:
            hostlist = ",".join(self._hosts)
            cmd.extend(['-hosts', hostlist])
        else:
            assert False, "Specify hostfile or hosts"

        cmd.extend(['python', server_path, '-tf', self._tmpfile.name])
        if self._rpc_ports is not None:
            cmd.extend(['-p', self._rpc_ports])
            self._rpc_ports = None
        logger.info("mpirun cmd: %s", cmd)

        try:
            mpirun_env = dict(os.environ)
            if 'MLSL_NUM_SERVERS' not in mpirun_env:
                mpirun_env['MLSL_NUM_SERVERS'] = '0'
            if 'MLSL_LOG_LEVEL' not in mpirun_env:
                mpirun_env['MLSL_LOG_LEVEL'] = '0'
            if 'MLSL_ALLOW_REINIT' not in mpirun_env:
                mpirun_env['MLSL_ALLOW_REINIT'] = '1'
            self.mpirun_proc = subprocess.Popen(cmd, preexec_fn=os.setsid, env=mpirun_env)
        except:
            raise RuntimeError("Process launch failed!")

    def close(self):
        if self.mpirun_proc is None:
            return

        if not self._tmpfile.closed:
            self._tmpfile.close()

        os.killpg(os.getpgid(self.mpirun_proc.pid), signal.SIGTERM)
        os.killpg(os.getpgid(self.mpirun_proc.pid), signal.SIGKILL)
        if self.mpirun_proc.poll() is None:
            logger.info("mpirun_proc isn't terminated")
        else:
            logger.info("mpirun_proc is terminated")
        self.mpirun_proc = None
