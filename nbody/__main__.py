#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys

import numpy as np
from PyQt5.QtWidgets import QApplication

from .mainwindow import MainWindow


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('nbody')
    parser.add_argument(
        '-s', '--seed',
        required=False,
        default=np.random.randint(0, 2 << 32 - 1, dtype=np.uint32),
        type=lambda s: int(s, base=0) % (2 << 32))
    args = parser.parse_args()

    print('Using seed {:d}'.format(args.seed))
    np.random.seed(args.seed)


    app = QApplication([])

    # for multisampling
    # from PyQt5.QtGui import QSurfaceFormat
    # fmt = QSurfaceFormat.defaultFormat()
    # fmt.setSamples(4)
    # QSurfaceFormat.setDefaultFormat(fmt)

    mw = MainWindow()
    mw.show()

    sys.exit(app.exec_())
