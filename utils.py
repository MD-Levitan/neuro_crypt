import numpy as np
import matplotlib.pyplot as plt

def read_from_file(file, num_bytes: int, transform=lambda x: int.from_bytes(x, byteorder='little')):
    """
    Read binary file and create array with these data.
    :param file: filename
    :param num_bytes: number of bytes in one element of array
    :param transform: function for transform bytes as element
    :return: Integer array
    """
    data = []
    with open(file, 'rb') as f:
        block = bytes(f.read(num_bytes))
        while block:
            data.append(transform(block))
            block = bytes(f.read(num_bytes))
    return data


def get_primitive(model):
    """
    Read primitive data.
    :param model: model = {input_file, output_file, input_transform, output_transform, num_bytes}
    :return: Input and output data as np.array
    """
    __in = read_from_file(model[0], model[4], model[2])

    __out = read_from_file(model[1], model[4], model[3])

    return np.array(__in), np.array(__out)


def bits(num):
    """
    From integer create bit's array
    :param num: integer
    :return: array of bits
    """
    bits_array = []
    for _ in range(0, 8):
        bits_array.append(num % 2)
        num //= 2
    return list(reversed(bits_array))


def split_by_bit(block, bit=32, order='right'):
    """
    From array integers create an array with bits.
    :param block: integer's array
    :param bit: number bits in each integer
    :param order:
    :return: array of bits(0 or 1)
    """
    rv = []
    for i in block:
        rv += bits(i)
    if order == 'left':
        rv = list(reversed(rv))
    # get @bit bites from right to left
    return rv[-bit:]


def split_by_byte(block, bytes=4):
    """
    From array integers create an array with bytes.
    :param block: integer's array
    :param bytes: number bytes in each integer
    :return: array of bytes
    """
    rv = []
    for i in block:
        rv.append(int(i))
    return rv[:bytes]


def xor(x1, x2):
    return list(map(lambda x: x[0] ^ x[1], zip(x1, x2)))


def to_int(bit_array):
    value = 0
    for i in bit_array:
        value *= 2
        value += round(i)
    return int(value)


def hamming_distance(x: int, y: int):
    var = x ^ y
    distance = 0
    while var:
        if var % 2:
            distance += 1
        var >>= 1
    return distance


class Graphic:
    y_percent = False
    y_limits: (float, float) = None
    y_limits_percent: (float, float) = None

    @staticmethod
    def create_graph(x, y, legend: list, filename: str, ticks=None):
        legend = [legend[0], legend[1], legend[2]]
        if Graphic.y_percent is True and Graphic.y_limits is not None:
            y = list(map(lambda iter: (iter / Graphic.y_limits[1]) * 100, y))
            if Graphic.y_limits_percent is None:
                y_limits = (0, 100 if max(y) < 100 else 105)
            else:
                y_limits = Graphic.y_limits_percent
            axes = plt.gca()
            axes.set_ylim(y_limits)
            legend[1] = "{}, %".format(legend[1])

        elif Graphic.y_limits is not None:
            axes = plt.gca()
            axes.set_ylim(Graphic.y_limits)

        plt.plot(x, y)
        if ticks is not None:
            plt.xticks(x, ticks)

        plt.xlabel(legend[0])
        plt.ylabel(legend[1])

        plt.title(legend[2])

        plt.savefig(filename)
        plt.close()

    @staticmethod
    def create_double_graph(x, y1, y2, legend: list, filename: str, ticks=None):
        legend = [legend[0], legend[1], legend[2]]

        if Graphic.y_percent is True and Graphic.y_limits is not None:
            y1 = list(map(lambda iter: (iter / Graphic.y_limits[1]) * 100, y1))
            y2 = list(map(lambda iter: (iter / Graphic.y_limits[1]) * 100, y2))
            if Graphic.y_limits_percent is None:
                y_limits = (0, 100 if max(max(y1), max(y2)) < 100 else 105)
            else:
                y_limits = Graphic.y_limits_percent
            axes = plt.gca()
            axes.set_ylim(y_limits)
            legend[1] = "{}, %".format(legend[1])

        elif Graphic.y_limits is not None:
            axes = plt.gca()
            axes.set_ylim(Graphic.y_limits)

        plt.plot(x, y1, marker='o', label="Test data")
        plt.plot(x, y2, marker='o', linestyle='--', label="Train data")
        plt.gca().legend(('Test data', 'Train data'))

        if ticks is not None:
            plt.xticks(x, ticks)

        plt.xlabel(legend[0])
        plt.ylabel(legend[1])

        plt.title(legend[2])

        plt.savefig(filename)
        plt.close()

    @staticmethod
    def init_graph():
        if Graphic.y_limits_percent is not None:
            axes = plt.gca()
            axes.set_ylim(Graphic.y_limits_percent)

        elif Graphic.y_limits is not None:
            axes = plt.gca()
            axes.set_ylim(Graphic.y_limits)

    @staticmethod
    def add_graph(x, y, model_name: str, ticks=None):
        if Graphic.y_percent is True:
            y = list(map(lambda iter: (iter / Graphic.y_limits[1]) * 100, y))

        plt.plot(x, y, marker='o', label=model_name)
        plt.xticks(x, ticks)

    @staticmethod
    def save_graph(legend: list(), models: list(), filename: str):
        plt.xlabel(legend[0])
        plt.ylabel(legend[1])
        plt.title(legend[2])

        plt.gca().legend(models)

        plt.savefig(filename)
        plt.close()


def calculate_np(n_layers: int, n_params: list(), input_size: int, output_size: int):
    layer = input_size
    np = 0
    for i in range(0, n_layers):
       np += layer * n_params[i]
       layer = n_params[i]
    np += layer * output_size
    return np


class Logger:
    file = "results/log.txt"
    table_file = "results/table.txt"

    @staticmethod
    def clear_files():
        with open(Logger.file, "w") as f:
            f.write("")
        with open(Logger.table_file, "w") as f:
            f.write("")

    @staticmethod
    def save_result(model_name: str, n_layers: int, n_params: list(), input_size: int, output_size: int,
                    accuracy: float, loss: float, number_iteration: int, time: int):
        with open(Logger.file, 'a') as f:
            f.write("Model: {}, NN: NN-{}, Params: {}, NP: {}, Accuracy: {}, Loss: {}, Iterations: {}, Time: {}".
                    format(model_name, n_layers, n_params if n_params is not None else 0,
                           calculate_np(n_layers, n_params, input_size, output_size),
                           accuracy, loss, number_iteration, time))
        with open(Logger.table_file, 'a') as f:
            f.write("	$ {} $ & {} & {} & {} & {} & {} & {:05.4f} & {:05.4f} & {} & {:05.4f} \\\\ \hline\n".
                    format(model_name, "NN" if n_layers == 0 else "MNN", str(n_params) if n_params is not None else 0,
                           calculate_np(n_layers, n_params, input_size, output_size), "-" , "-",
                           accuracy, loss, number_iteration, time))
