from PyInstaller.utils.hooks import collect_data_files
from vispy.app.backends import CORE_BACKENDS

datas = collect_data_files('vispy')

hiddenimports = ["vispy.ext._bundled.six"]

# adding all backends is required for vispy.sys_info() to work
hiddenimports += ["vispy.app.backends." + b[1] for b in CORE_BACKENDS]
hiddenimports += ["vispy.app.backends._test"]
