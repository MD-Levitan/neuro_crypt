import utils


class Model:
    def __init__(self, input_file: str, output_file: str, input_size: int, output_size: int,
                 in_fun=None, out_fun=None):
        """

        :param input_file:
        :param output_file:
        :param input_size:
        :param output_size:
        :param in_fun:
        :param out_fun:
        """
        self.input_file = input_file
        self.output_file = output_file
        self.input_size = input_size
        self.output_size = output_size
        self.input_fun = in_fun
        self.output_fun = out_fun

        if in_fun is None:
            self.input_fun = lambda x: utils.split_by_bit(x, input_size)
        if out_fun is None:
            self.output_fun = lambda x: utils.split_by_bit(x, output_size)

    def get_test_data(self):

        __in = utils.read_from_file(self.input_file, (self.input_size + 7) // 8, self.input_fun)

        __out = utils.read_from_file(self.output_file, (self.output_size + 7) // 8, self.output_fun)

        return utils.np.array(__in), utils.np.array(__out)


class ModelStorage:
    def __init__(self):
        self.storage = dict()

    def add_feistel_models(self, max_iteration=16):
        for iteration in range(1, max_iteration + 1):
            for shift in range(0, 8):
                model_name = "f{}-{}".format(iteration, shift)
                self.storage[model_name] = Model(input_file="generator/bin/{}_x.bin".format(model_name),
                                                 output_file="generator/bin/{}_y.bin".format(model_name),
                                                 input_size=16, output_size=16)

    def add_primitive_models(self):
        # Adding G0, G1, G2 models
        for id in range(0, 3):
            model_name = "g{}".format(id)
            self.storage[model_name] = Model(input_file="generator/bin/{}_x.bin".format(model_name),
                                             output_file="generator/bin/{}_y.bin".format(model_name),
                                             input_size=8, output_size=4)
        model_name = "g3"
        self.storage[model_name] = Model(input_file="generator/bin/{}_x.bin".format(model_name),
                                                 output_file="generator/bin/{}_y.bin".format(model_name),
                                                 input_size=64, output_size=32)

        model_name = "g5"
        self.storage[model_name] = Model(input_file="generator/bin/{}_x.bin".format(model_name),
                                                 output_file="generator/bin/{}_y.bin".format(model_name),
                                                 input_size=4, output_size=4)



        # Adding G4-<S> models
        for size in (4, 8, 16, 32):
            model_name = "g{}".format(size)
            file = "g4-{}".format(size)
            self.storage[model_name] = Model(input_file="generator/bin/{}_x.bin".format(file),
                                             output_file="generator/bin/{}_y.bin".format(file),
                                             input_size=size, output_size=size)

    def add_custom_model(self, model_name: str, input_size: int, output_size: int, in_fun=None, out_fun=None,
                         file_name: str = "generator/bin/"):
        """
        Add model to storage
        :param model_name: model name
        :param file_name: filename prefix
        :param in_fun: function for parsing input
        :param out_fun: function for parsing output
        """
        self.storage[model_name] = Model(input_file=file_name + "{}_x.bin".format(model_name),
                                         output_file=file_name + "{}_x.bin".format(model_name),
                                         input_size=input_size, output_size=output_size,
                                         in_fun=in_fun, out_fun=out_fun)

    def add_standart_models(self):
        self.add_feistel_models()
        self.add_primitive_models()

    def get_model(self, model_name):
        model = self.storage.get(model_name, None)
        return model
