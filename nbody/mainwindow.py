# -*- coding: utf-8 -*-

import numpy as np
from pyrr import Vector3
from PyQt5.QtWidgets import QMainWindow

from .simulationview import SimulationView
from .particle import Particle


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('CS 4732 Final Project — N-Body Simulator — Daniel Beckwith')

        self.sim_view = SimulationView(self)

        np.random.seed(0xdeadbeef)
        for _ in range(20):
            # https://people.cs.kuleuven.be/~philip.dutre/GI/TotalCompendium.pdf
            r1, r2, r3 = np.random.rand(3)
            r1 *= 2 * np.pi
            r2_sqrt = 2 * np.sqrt(r2 * (1 - r2))
            r3 *= 10
            px = r3 * np.cos(r1) * r2_sqrt
            py = r3 * np.sin(r1) * r2_sqrt
            pz = r3 * (1 - 2 * r2)

            p = Particle(
                position=Vector3([px, py, pz]),
                velocity=Vector3(),
                mass=1,
                radius=0.1)
            self.sim_view.sim.particles.add(p)

        self.setCentralWidget(self.sim_view)
