from loguru import logger
import prettytable
import time
import torch


TIMER_NAME_KEY = "name"
TIMER_TIME_KEY = "time"
TIMER_CHILDREN_KEY = "children"


class Timer:
    """Time and add results with a recursive structure to a parent list. An example of the format of the resulting dictionary: {'name': 'timer1', 'time': 10.0, 'children': [ {'name': 'timer2', 'time': 5.0, 'children': []} ]}"""

    def __init__(self, name: str, parent: list):
        # name
        self.name = name
        # add results to the parent list at the end of the timer
        self.parent = parent
        # start time
        self.start_time = None
        # children will add results to this list
        self.children = []
        # if the timer has finished
        self.finished = False

    def start(self):
        if self.start_time is not None:
            raise ValueError("Timer has already been started")
        torch.cuda.synchronize()
        self.start_time = time.time()

    def end(self):
        if self.start_time is None:
            raise ValueError("Timer has not been started")
        torch.cuda.synchronize()
        elapsed = time.time() - self.start_time
        res = {
            TIMER_NAME_KEY: self.name,
            TIMER_TIME_KEY: elapsed,
            TIMER_CHILDREN_KEY: self.children,
        }
        self.parent.append(res)
        self.finished = True
        return res

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()


class TimerManager:
    def __init__(self):
        self._stats = []  # will only contain the results of the root timer
        self.timers = []  # stack of timers

    def start(self):
        self.root_timer = Timer(name="root", parent=self._stats)
        self.root_timer.start()
        self.timers.append(self.root_timer)

    def __call__(self, name: str):
        if not self.timers:
            raise ValueError("Need to call timer.start() before using timer()")
        prev = self.timers.pop()
        while prev.finished:  # check if previous timer has finished
            prev = self.timers.pop()
        assert not prev.finished
        self.timers.append(prev)
        timer = Timer(name=name, parent=prev.children)
        self.timers.append(timer)
        return timer

    def end(self):
        self.root_timer.end()

    @property
    def stats(self):
        assert self.root_timer.finished
        assert len(self._stats) == 1
        return self._stats[0]

    def add_rows_recursive(self, table, stats, total_time, level=0):
        for stat in stats:
            if level == 0:
                prefix = ""
            else:
                prefix = " " * (5 * level - 3) + "\\_ "  # manually calculated
            time = stat[TIMER_TIME_KEY]
            percentage = time / total_time * 100
            table.add_row([prefix + stat[TIMER_NAME_KEY], time, percentage])
            self.add_rows_recursive(
                table=table,
                stats=stat[TIMER_CHILDREN_KEY],
                total_time=total_time,
                level=level + 1,
            )
            if level == 0:
                table.add_divider()

    def log(self):
        # create table
        table = prettytable.PrettyTable()
        task_header = "Name"
        time_header = "Time (s)"
        percentage_header = "Time (%)"  # time percentage of parent time
        table.field_names = [task_header, time_header, percentage_header]

        # set style
        table.align[task_header] = "l"
        table.align[time_header] = "r"
        table.align[percentage_header] = "r"
        table.float_format[time_header] = ".4"
        table.float_format[percentage_header] = ".2"

        # add row of total time
        total_time = self.stats[TIMER_TIME_KEY]
        table.add_row(["Total", total_time, 100.0])
        table.add_divider()

        # recursively add rows
        self.add_rows_recursive(
            table=table,
            stats=self.stats[TIMER_CHILDREN_KEY],
            total_time=total_time,
            level=0,
        )

        # log
        logger.info(f"\n{table}")


timer = TimerManager()
