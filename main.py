# This is a sample Python script.

# Press ⌘⏎ to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from collections import OrderedDict
from collections import namedtuple
from itertools import product

def get_runs(params):
    """
    Gets the run parameters using cartesian products of the different parameters.
    :param params: different parameters like batch size, learning rates
    :return: iterable run set
    """
    Run = namedtuple("Run", params.keys())

    runs = []
    for v in product(*params.values()):
        runs.append(Run(*v))

    return runs


def print_hi(name):
    final_parameters = OrderedDict(
        lr=[0.01, 0.001],
        batch_size=[64, 128],
        shuffle=[False]
    )
    run_list = get_runs(final_parameters)
    for run in run_list:
        print("--------------------------------------------")
        print(run)
        print(run.lr)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
