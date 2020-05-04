import numpy as np


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

