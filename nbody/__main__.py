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

    mw = MainWindow()
    mw.show()

    sys.exit(app.exec_())
