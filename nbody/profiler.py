# -*- coding: utf-8 -*-

import time

import numpy as np


class Profiler(object):
    def __init__(self, print_interval=None, max_samples=1000, stages=[]):
        self.print_interval = print_interval
        if self.print_interval is not None:
            self.print_timer = time.time()
        else:
            self.print_timer = None
        self.max_samples = max_samples
        self.ups_sampler = ValueSampler(self.max_samples)
        self.timer = None
        self.stages = {}
        self.stages_order = []
        self.curr_stage = None
        for stage_name in stages:
            self.stages_order.append(stage_name)
            self.stages[stage_name] = ProfilerStage(self.max_samples)

    def stage(self, stage_name):
        # TODO: support hierarchical stage names
        # i.e. render should show total render time with render.camera and render.copy separate
        t = time.time()
        if self.curr_stage is not None:
            self.curr_stage.end_time = t
        if stage_name not in self.stages:
            self.stages_order.append(stage_name)
            self.stages[stage_name] = ProfilerStage(self.max_samples)
        self.curr_stage = self.stages[stage_name]
        self.curr_stage.start_time = t

    def end_stage(self):
        t = time.time()
        if self.curr_stage is not None:
            self.curr_stage.end_time = t

    def update(self):
        t = time.time()
        if self.timer is None:
            self.timer = t
            return
        dt = t - self.timer
        self.timer = t
        ups = 1 / dt

        self.ups_sampler.add_sample(ups)

        if self.curr_stage is not None:
            self.curr_stage.end_time = t
        for stage in self.stages.values():
            stage.update(dt)

        if self.print_timer is not None and t - self.print_timer >= self.print_interval:
            self.print_timer = t
            ups = self.ups_sampler.capture_value()
            avg_time = 1 / ups
            if self.stages:
                print('[PROFILER]', '='*80)
                for stage_name in self.stages_order:
                    stage = self.stages[stage_name]
                    proportion = stage.time_sampler.capture_value()
                    print('[PROFILER]', '{:s}: {:.1%} ({:.3f}ms)'.format(stage_name, proportion, proportion * avg_time * 1000))
            print('[PROFILER]', 'Total UPS: {:.1f} ({:.3f}ms)'.format(ups, avg_time * 1000))

class ProfilerStage(object):
    def __init__(self, max_samples):
        self.time_sampler = ValueSampler(max_samples)
        self.start_time = None
        self.end_time = None

    def update(self, total_time):
        self.time_sampler.add_sample((self.end_time - self.start_time) / total_time)
        self.start_time = None
        self.end_time = None

class ValueSampler(object):
    def __init__(self, max_samples):
        self.max_samples = max_samples
        self.samples = np.empty((self.max_samples,), dtype=np.float32)
        self.num_samples = 0
        self.sample_pos = 0

    def add_sample(self, sample):
        self.num_samples += 1
        self.num_samples = min(self.num_samples, self.max_samples)

        self.samples[self.sample_pos] = sample

        self.sample_pos += 1
        self.sample_pos %= self.max_samples

    def capture_value(self):
        if self.num_samples == self.max_samples:
            value = np.mean(self.samples)
        else:
            value = np.mean(self.samples[:self.num_samples])
        self.num_samples = 0
        self.sample_pos = 0
        return value
