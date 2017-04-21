# -*- coding: utf-8 -*-

import sys
import time

import numpy as np


class Profiler(object):
    stage_name_separator = '.'
    debug = False

    def __init__(self):
        self._root_stage = Stage('total', None)
        self._curr_stage_name = None

    def begin(self, stage_name=''):
        if self.debug:
            print('-'*80)
            print('begin', stage_name)
        t = time.time()
        stage_name = self._split_name(stage_name)

        self._end(t, stage_name)

        stage = self._root_stage
        add_begin = False
        for i, part in enumerate(stage_name):
            if self.debug: print(part)
            if i > 0:
                stage = stage.get_sub_stage(part)
            if not add_begin and not (self._curr_stage_name and i < len(self._curr_stage_name) and part == self._curr_stage_name[i]):
                add_begin = True
            if add_begin:
                if self.debug: print(stage.name + '.add_begin')
                stage.add_begin(t)

        self._curr_stage_name = stage_name

    def _end(self, t, stage_name):
        ended_count = 0
        if self._curr_stage_name:
            stage = self._root_stage
            add_end = False
            for i, part in enumerate(self._curr_stage_name):
                if self.debug: print(part)
                if i > 0:
                    stage = stage.get_sub_stage(part)
                if not add_end and not (stage_name and i < len(stage_name) and part == stage_name[i]):
                    add_end = True
                if add_end:
                    if self.debug: print(stage.name + '.add_end')
                    stage.add_end(t)
                    ended_count += 1
        return ended_count

    def end(self, stage_name=''):
        if self.debug:
            print('-'*80)
            print('end', stage_name)
        t = time.time()
        stage_name = self._split_name(stage_name)
        ended_count = self._end(t, stage_name[:-1])
        self._curr_stage_name = self._curr_stage_name[:-ended_count]
        if self.debug: print('new curr stage name:', self._curr_stage_name)

        if stage_name == [self._root_stage.name]:
            if all(stage.has_new_times for stage in self._root_stage.iter_preorder()):
                for stage in self._root_stage.iter_preorder():
                    stage.commit()

    def _split_name(self, name):
        if not name:
            return [self._root_stage.name]
        else:
            return [self._root_stage.name] + name.split(self.stage_name_separator)

    def print_stages(self, *args, **kwargs):
        if self._root_stage.used:
            self._root_stage.print_stages(*args, **kwargs)

    def reset(self):
        if self._root_stage.used:
            for stage in self._root_stage.iter_preorder():
                stage.reset()

class Stage(object):
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent

        self._sub_stage_order = []
        self._sub_stage_names = {}

        self.reset()

    def add_begin(self, t):
        self._new_begin_time = t

    def add_end(self, t):
        self._new_end_time = t

    @property
    def has_new_times(self):
        return self._new_begin_time is not None and self._new_end_time is not None

    def commit(self):
        self._begin_times.append(self._new_begin_time)
        self._end_times.append(self._new_end_time)
        self._new_begin_time = None
        self._new_end_time = None
        self._avg_time = None

    def reset(self):
        self._begin_times = []
        self._new_begin_time = None

        self._end_times = []
        self._new_end_time = None

        self._avg_time = None

    def get_sub_stage(self, name):
        if name not in self._sub_stage_names:
            stage = Stage(name, self)
            self._sub_stage_names[name] = stage
            self._sub_stage_order.append(stage)
            return stage
        else:
            return self._sub_stage_names[name]

    @property
    def used(self):
        return self._begin_times and self._end_times

    @property
    def sub_stages(self):
        yield from self._sub_stage_order

    @property
    def avg_time(self):
        if self._avg_time is None:
            self._avg_time = np.mean(np.array(self._end_times) - np.array(self._begin_times))
        return self._avg_time

    def print_stages(self, file=sys.stdout, depth=0):
        for _ in range(depth):
            file.write('\t')
        file.write(self.name)
        file.write(': ')
        file.write(_format_time(self.avg_time))
        if self.parent is not None:
            file.write(' ({:.2%} of {:s}'.format(self.avg_time / self.parent.avg_time, self.parent.name))
            if self.parent.parent is not None:
                root = self
                while root.parent is not None:
                    root = root.parent
                file.write(', {:.2%} of {:s}'.format(self.avg_time / root.avg_time, root.name))
            file.write(')')
        else:
            file.write(' ({:.3g} UPS)'.format(1 / self.avg_time))
        file.write('\n')
        for sub_stage in self.sub_stages:
            sub_stage.print_stages(file, depth + 1)

    def iter_preorder(self):
        yield self
        for sub_stage in self.sub_stages:
            yield from sub_stage.iter_preorder()

def _format_time(t):
    t, micros = divmod(int(t * 1000000), 1000000)
    t, secs = divmod(t, 60)
    t, mins = divmod(t, 60)
    hours = int(t)
    s = ''
    show = False
    if show or hours:
        show = True
        s += '{:d}h '.format(hours)
    if show or mins:
        show = True
        s += '{:d}m '.format(mins)
    if show or secs:
        show = True
        s += '{:d}s '.format(secs)
    if show or micros:
        show = True
        s += '{:.3f}ms'.format(micros / 1000)
    return s

PROFILER = Profiler()


if __name__ == '__main__':
    PROFILER.debug = True

    for _ in range(2):
        print('='*80)
        PROFILER.begin()
        PROFILER.begin('update')
        PROFILER.end('update')
        PROFILER.begin('render')
        PROFILER.begin('render.camera')
        PROFILER.begin('render.particles.data')
        PROFILER.begin('render.particles.copy')
        PROFILER.begin('render.particles.draw')
        PROFILER.end('render.particles.draw')
        PROFILER.end()

    PROFILER.print_stages()
