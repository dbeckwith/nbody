#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys

from PyQt5.QtWidgets import QApplication

from .mainwindow import MainWindow


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('nbody')
    args = parser.parse_args()

    app = QApplication([])

    # for multisampling
    # from PyQt5.QtGui import QSurfaceFormat
    # fmt = QSurfaceFormat.defaultFormat()
    # fmt.setSamples(4)
    # QSurfaceFormat.setDefaultFormat(fmt)

    mw = MainWindow()
    mw.show()

    sys.exit(app.exec_())
