import os
import subprocess
import time
import logging
import signal
import tempfile
import fcntl
import re


LINE_TOKEN = 'token'
logger = logging.getLogger(__name__)


class MPILauncher(object):
    """
    execute mpirun cmd
    close process for mpirun
    """
    def __init__(self):
        self.mpirun_proc = None
        self._hostfile = os.getenv('HETR_SERVER_HOSTFILE')
        # TODO add handling/error message for improperly formatted var
        self._rpc_ports = os.getenv('HETR_SERVER_PORTS')
        if self._rpc_ports:
            self._rpc_ports = self._rpc_ports.split(',')
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

    def get_rpc_port_by_rank(self, rank, num_servers):
        if self.mpirun_proc is None:
            raise RuntimeError("Launch mpirun_proc before reading of rpc ports")

        if self._rpc_ports is not None:
            return self._rpc_ports[rank]

        server_info_pattern = re.compile("^" + LINE_TOKEN +
                                         ":([\d]+):([\d]+):([\d]+):" +
                                         LINE_TOKEN + "$")
        self._tmpfile.seek(0)
        while True:
            fcntl.lockf(self._tmpfile, fcntl.LOCK_SH)
            line_count = sum(1 for line in self._tmpfile if server_info_pattern.match(line))
            self._tmpfile.seek(0)
            fcntl.lockf(self._tmpfile, fcntl.LOCK_UN)

            if line_count == num_servers:
                break
            else:
                time.sleep(0.1)

        server_infos = [tuple([int(server_info_pattern.match(line).group(1)),
                               int(server_info_pattern.match(line).group(3))])
                        for line in self._tmpfile]
        server_infos = sorted(server_infos, key=lambda x: x[0])
        self._rpc_ports = [row[1] for row in server_infos]
        logger.debug("get_rpc_ports: ports (in MPI rank order): %s", self._rpc_ports)
        self._tmpfile.close()
        return self._rpc_ports[rank]

    def get_address_by_rank(self, rank, num_servers):
        return '{}:{}'.format(self.get_host_by_rank(rank),
                              self.get_rpc_port_by_rank(rank, num_servers))

    def launch(self, num_servers, process_per_node):
        if self.mpirun_proc is not None:
            logger.debug("mpirun_proc is already launched")
            return
        server_path = os.path.dirname(os.path.realpath(__file__)) + "/hetr_server.py"
        mpirun_env = dict(os.environ)

        cmd = ['mpirun',
               '-n', str(num_servers),
               '-ppn', str(process_per_node),
               '-l']  # to print MPI rank index for each log line
        if 'MLSL_NUM_SERVERS' not in mpirun_env:
            mpirun_env['MLSL_NUM_SERVERS'] = '0'
        if 'MLSL_LOG_LEVEL' not in mpirun_env:
            mpirun_env['MLSL_LOG_LEVEL'] = '0'
        if 'MLSL_ALLOW_REINIT' not in mpirun_env:
            mpirun_env['MLSL_ALLOW_REINIT'] = '1'

        logger.debug("mpilauncher: launch: hostfile %s, hosts %s, num_servers %s, tmpfile %s",
                     self._hostfile, self._hosts, num_servers, self._tmpfile)

        if self._hostfile is not None:
            cmd.extend(['-hostfile', self._hostfile])
        elif (self._hosts is not None):
            hostlist = ",".join(self._hosts)
            cmd.extend(['-hosts', hostlist])
        else:
            assert False, "Specify hostfile or hosts"
        cmd.extend(['python', server_path, '-tf', self._tmpfile.name])
        if self._rpc_ports is not None:
            cmd.extend(['-p'] + self._rpc_ports)
            self._rpc_ports = None
        logger.debug("mpirun cmd: %s", cmd)

        try:
            self.mpirun_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                                preexec_fn=os.setsid, env=mpirun_env)
        except:
            raise RuntimeError("Process launch failed!")

    def close(self):
        if self.mpirun_proc is None:
            return

        if not self._tmpfile.closed:
            self._tmpfile.close()

        os.killpg(os.getpgid(self.mpirun_proc.pid), signal.SIGTERM)
        os.killpg(os.getpgid(self.mpirun_proc.pid), signal.SIGKILL)
        self.mpirun_proc.wait()
        self.mpirun_proc = None
