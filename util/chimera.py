from typing import List
import subprocess
import os


class Chimera:

    @staticmethod
    def run(tmp_dir: str, commands: List[str]):
        """
        Runs list of chimera commands

        :param tmp_dir: Output directory where chimera script is temporarily
        saved
        :param commands: List of chimera commands
        """
        print('\n'.join(commands))
        # Create script and save to file system
        with open(os.path.join(tmp_dir, 'chimera_script.cmd'), 'w') as fp:
            fp.write('\n'.join(commands))

        # Run chimera as separate process and pass created script
        p = subprocess.Popen([Chimera.get_binary(), '--nogui', fp.name], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        lines = []

        for line in p.stderr:
            print(line.decode('utf-8').rstrip())
        for line in p.stdout:
            lines.append(line)
        # Remove chimera script from file system
        os.remove(fp.name)

        return lines

    @staticmethod
    def get_binary():
        if 'CHIMERA_PATH' in os.environ and os.path.isfile(os.environ['CHIMERA_PATH']):
            return os.environ['CHIMERA_PATH']

        # If CHIMERA environment variable isn't set check if chimera binary can
        # be found in default locations
        for possible_location in ['/bin/chimera', '/usr/bin/chimera', '/usr/local/bin/chimera', '/usr/local/bin/chimera/bin/chimera']:
            if os.path.isfile(possible_location):
                return possible_location

        raise FileNotFoundError(
            'Chimera binary file not found. Install chimera '
            '(https://www.cgl.ucsf.edu/chimera/download.html) and set the \'CHIMERA_PATH\' '
            'environment variable to point to the chimera binary '
            '(see https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/startup.html)'
        )