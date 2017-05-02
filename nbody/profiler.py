# -*- coding: utf-8 -*-

import sys
import time

import numpy as np


class Profiler(object):
    """
    Class for profiling an update loop.
    """

    # separating character between stage name parts
    stage_name_separator = '.'

    def __init__(self):
        # make root stage
        self._root_stage = Stage('total', None)
        self._curr_stage_name = None

    def begin(self, stage_name=''):
        """
        Indicate that the given stage is about to begin.
        If no stage name is given, indicates that the entire loop is about to begin.

        Arguments:
            stage_name: str, the stage name, with hierarchical names separated by Profiler.stage_name_separator
        """
        # capture the current time
        t = time.time()
        # parse the name
        stage_name = self._split_name(stage_name)

        # ensure that the previous stage is ended
        self._end(t, stage_name)

        # add a begin-time to all stage parts of the new stage name that share a common root with the current stage
        stage = self._root_stage
        add_begin = False
        for i, part in enumerate(stage_name):
            if i > 0:
                stage = stage.get_sub_stage(part)
            if not add_begin and not (self._curr_stage_name and i < len(self._curr_stage_name) and part == self._curr_stage_name[i]):
                add_begin = True
            if add_begin:
                stage.add_begin(t)

        # set new current stage
        self._curr_stage_name = stage_name

    def _end(self, t, stage_name):
        """
        Ensures that the previous stage has ended properly by adding an end-time to it.

        Arguments:
            t: float, the time of the event
            stage_name: [str], the new parsed stage name

        Returns:
            int, how many stage parts got new end-times
        """
        ended_count = 0
        if self._curr_stage_name:
            # add an end-time to all stage parts of the current stage name that share a common root with the new stage
            stage = self._root_stage
            add_end = False
            for i, part in enumerate(self._curr_stage_name):
                if i > 0:
                    stage = stage.get_sub_stage(part)
                if not add_end and not (stage_name and i < len(stage_name) and part == stage_name[i]):
                    add_end = True
                if add_end:
                    stage.add_end(t)
                    ended_count += 1
        return ended_count

    def end(self, stage_name=''):
        """
        Indicate that the given stage has just ended.
        If no stage name is given, indicates that the entire loop has just ended.

        Arguments:
            stage_name: str, the stage name, with hierarchical names separated by Profiler.stage_name_separator
        """
        # capture event time
        t = time.time()
        # parse stage name
        stage_name = self._split_name(stage_name)
        # end the current stage
        ended_count = self._end(t, stage_name[:-1])
        # now that this stage is ended, back up stage hierarchy by how many were ended
        self._curr_stage_name = self._curr_stage_name[:-ended_count]

        # if this was the root stage ending, commit all the results from this loop
        if stage_name == [self._root_stage.name]:
            # only commit if every stage got new times
            if all(stage.has_new_times for stage in self._root_stage.iter_preorder()):
                for stage in self._root_stage.iter_preorder():
                    stage.commit()

    def _split_name(self, name):
        """
        Parses the given stage name into stage parts.

        Arguments:
            name: str
        """
        if not name:
            return [self._root_stage.name]
        else:
            return [self._root_stage.name] + name.split(self.stage_name_separator)

    def print_stages(self, file=sys.stdout):
        """
        Prints out the current stage timings to the given file.

        Arguments:
            file: file-like, the file to print to
        """
        # only print if the profiler has been used at all
        if self._root_stage.used:
            self._root_stage.print_stages(file)

    def reset(self):
        """
        Resets the timing values of all stages.
        """
        # only reset if the profiler has been used at all
        if self._root_stage.used:
            for stage in self._root_stage.iter_preorder():
                stage.reset()

class Stage(object):
    """
    Class representing a single profiler stage.
    """
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent

        # child stages, both in creation order and in a lookup dict
        self._sub_stage_order = []
        self._sub_stage_names = {}

        self.reset()

    def add_begin(self, t):
        """
        Record a new beginning time for this stage.
        """
        self._new_begin_time = t

    def add_end(self, t):
        """
        Record a new ending time for this stage.
        """
        self._new_end_time = t

    @property
    def has_new_times(self):
        """
        Indicates if this stage has new times to record or not.
        """
        return self._new_begin_time is not None and self._new_end_time is not None

    def commit(self):
        """
        Commits saved timings.
        """
        self._begin_times.append(self._new_begin_time)
        self._end_times.append(self._new_end_time)
        self._new_begin_time = None
        self._new_end_time = None
        self._avg_time = None

    def reset(self):
        """
        Resets the stage, forgetting any saved timings.
        """
        self._begin_times = []
        self._new_begin_time = None

        self._end_times = []
        self._new_end_time = None

        self._avg_time = None

    def get_sub_stage(self, name):
        """
        Gets a child stage of this stage, creating it if it doesn't exist.
        """
        if name not in self._sub_stage_names:
            stage = Stage(name, self)
            self._sub_stage_names[name] = stage
            self._sub_stage_order.append(stage)
            return stage
        else:
            return self._sub_stage_names[name]

    @property
    def used(self):
        """
        Indicates if this stage has been used since being last reset.
        """
        return self._begin_times and self._end_times

    @property
    def sub_stages(self):
        """
        Returns an iterator of the child stages.
        """
        yield from self._sub_stage_order

    @property
    def avg_time(self):
        """
        Gets the average elapsed time for this stage.
        """
        if self._avg_time is None:
            self._avg_time = np.mean(np.subtract(self._end_times, self._begin_times))
        return self._avg_time

    def print_stages(self, file=sys.stdout, depth=0):
        """
        Prints this stage hierarchy's timings to the given file.
        """
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
        """
        Iterate through this stage tree in pre-order.
        """
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

# global profiler
PROFILER = Profiler()
