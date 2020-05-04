import random

def pol_2_1(x):
    bits = [(x >> i) & 0x01 for i in range(0, 4)]
    return (bits[0] & bits[1]) % 2

def pol_3_1(x):
    bits = [(x >> i) & 0x01 for i in range(0, 4)]
    return (bits[2] + (bits[0] & bits[1])) % 2

def pol_4_1(x):
    bits = [(x >> i) & 0x01 for i in range(0, 4)]
    return (bits[0] + bits[1] + bits[2] + (bits[1] & bits[2])) % 2

def pol_4_2(x):
    bits = [(x >> i) & 0x01 for i in range(0, 4)]
    return (bits[0] + bits[1] + bits[2] + (bits[1] & bits[3])) % 2

def pol_5_1(x):
    bits = [(x >> i) & 0x01 for i in range(0, 5)]
    return (bits[0] + bits[1] + bits[2] + (bits[2] & bits[4])) % 2

def pol_5_2(x):
    bits = [(x >> i) & 0x01 for i in range(0, 5)]
    return (bits[0] + bits[2] + (bits[1] & bits[2]) + (bits[3] & bits[4])) % 2

def pol_6_1(x):
    bits = [(x >> i) & 0x01 for i in range(0, 6)]
    return (bits[0] + bits[2] + (bits[3] & bits[2]) + (bits[3] & bits[5])) % 2

def pol_7_1(x):
    bits = [(x >> i) & 0x01 for i in range(0, 7)]
    return (bits[0] + bits[2] + (bits[4] & bits[2]) + (bits[3] & bits[5])) % 2

def pol_8_1(x):
    bits = [(x >> i) & 0x01 for i in range(0, 8)]
    return (bits[0] + bits[2] + (bits[1] & bits[7]) + (bits[3] & bits[2])) % 2

def pol_9_1(x):
    bits = [(x >> i) & 0x01 for i in range(0, 9)]
    return (bits[0] + bits[2] + (bits[2] & bits[7]) + (bits[4] & bits[6])) % 2

def pol_10_1(x):
    bits = [(x >> i) & 0x01 for i in range(0, 10)]
    return (bits[0] + bits[2] + (bits[1] & bits[5]) + (bits[4] & bits[9])) % 2

def pol_11_1(x):
    bits = [(x >> i) & 0x01 for i in range(0, 11)]
    return (bits[0] + bits[4] + (bits[3] & bits[2]) + (bits[2] & bits[10])) % 2

def pol_12_1(x):
    bits = [(x >> i) & 0x01 for i in range(0, 11)]
    return (bits[0] + bits[4] + (bits[3] & bits[2]) + (bits[2] & bits[10])) % 2

function_static = {2: pol_2_1, 3: pol_3_1, 4: pol_4_1, 5: pol_5_1, 6: pol_6_1,
                   7: pol_7_1, 8: pol_8_1, 9: pol_9_1}

class DataGenerator:
    def __init__(self, n_input):
        self.__n_input = n_input
        self.polymon = None

    def generate_polynom(self, params=None):
        if params is None or len(params) != self.__n_input:
            params = [random.random() for _ in range(0, self.__n_input)]
        self.polymon = params

    def generate_data_static(self, input_size, input=None):
        output = []
        input_fmt = []

        for _ in range(0, input_size):
            x = random.randrange(0, pow(2, self.__n_input))
            input_fmt.append([(x >> i) & 0x01 for i in range(0, self.__n_input)])
            data_x = [0, 0]
            data_x[function_static[self.__n_input](x)] = 1
            output.append(data_x)
        return input_fmt, output

    def generate_data(self, input_size, input=None):
        if input != None:
            pass
        output = []
        input_fmt = []

        def polynom_calc(x):
            bits = [(x >> i) & 0x01 for i in range(0, self.__n_input)]
            res = 0
            for i in range(0, len(bits) // 2):
                res += bits[i * 2] & bits[i * 2 + 1]
                res %= 2
            return res

        for _ in range(0, input_size):
            x = random.randrange(0, pow(2, self.__n_input))
            input_fmt.append([(x >> i) & 0x01 for i in range(0, self.__n_input)])
            data_x = [0, 0]
            data_x[polynom_calc(x)] = 1
            output.append(data_x)
        return input_fmt, output

