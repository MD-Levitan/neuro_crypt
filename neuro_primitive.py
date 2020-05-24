import matplotlib.pyplot as plt
import numpy as np

import networks
import utils

models = {
    # "g3": ("generator/bin/g3_x.bin", "generator/bin/g3_y.bin",
    #        lambda x: utils.split_by_bit(x, 32), lambda x: utils.split_by_bit(x, 32), 4),
    # "g4": ("generator/bin/g4_x.bin", "generator/bin/g4_y.bin",
    #        lambda x: utils.split_by_bit(x, 4), lambda x: utils.split_by_bit(x, 4), 1),
    # "g4l": ("generator/bin/g4l_x.bin", "generator/bin/g4l_y.bin",
    #        lambda x: utils.split_by_bit(x, 8), lambda x: utils.split_by_bit(x, 8), 1),
}


def add_feistel_model(storage: dict, max_iteration=16):
    for iteration in range(1, max_iteration + 1):
        for shift in range(0, 8):
            model_name = "f{}-{}".format(iteration, shift)
            storage[model_name] = ("generator/bin/{}_x.bin".format(model_name),
                                   "generator/bin/{}_y.bin".format(model_name),
                                   lambda x: utils.split_by_bit(x, 16),
                                   lambda x: utils.split_by_bit(x, 16),
                                   2)


def add_primitive_model(storage: dict):
    for id in range(0, 3):
        model_name = "g{}".format(id)
        storage[model_name] = ("generator/bin/{}_x.bin".format(model_name),
                               "generator/bin/{}_y.bin".format(model_name),
                               lambda x: utils.split_by_bit(x, 8),
                               lambda x: utils.split_by_bit(x, 4), 1)


def add_custom_model(storage: dict, model_name: str, in_fun, out_fun, bytes: int,
                     file_name: str = "generator/bin/"):
    """
    Add model to storage
    :param storage: storage with models
    :param model_name: model name
    :param file_name: filename prefix
    :param in_fun: function for parsing input
    :param out_fun: function for parsing output
    :param bytes: number of bytes
    """
    storage[model_name] = (file_name + "{}_x.bin".format(model_name),
                           file_name + "{}_y.bin".format(model_name),
                           in_fun, out_fun, bytes)


def get_test_data(model_name: str, storage):
    return utils.get_primitive(storage[model_name])


def experiment_std(input_data, output_data, n_input, n_classes, training_epochs=15000, display_step=1000):
    x, y, _, _ = networks.init_multilayer_network(input_data, output_data, n_input, n_classes,
                                                  training_epochs=training_epochs, display_step=display_step)

    x, y, _, _ = networks.init_multilayer_network(input_data, output_data, n_input, [4], n_classes, 1,
                                                  training_epochs=training_epochs, display_step=display_step)

    x, y, _, _ = networks.init_multilayer_network(input_data, output_data, n_input, [8], n_classes, 1,
                                                  training_epochs=training_epochs, display_step=display_step)

    x, y, _, _ = networks.init_multilayer_network(input_data, output_data, n_input, [16], n_classes, 1,
                                                  training_epochs=training_epochs, display_step=display_step)

    x, y, _, _ = networks.init_multilayer_network(input_data, output_data, n_input, [32], n_classes, 1,
                                                  training_epochs=training_epochs, display_step=display_step)

    x, y, _, _ = networks.init_multilayer_network(input_data, output_data, n_input, [32, 32], n_classes, 2,
                                                  training_epochs=training_epochs, display_step=display_step)


def experiment_changeable_0l(input_data, output_data, n_input, n_classes, model_name,
                             training_epochs_min=5000, training_epochs_max=30000, display_step=20000):
    accurancy = []
    loss = []
    train = []
    for i in range(training_epochs_min, training_epochs_max, 5000):
        x, y, _, _ = networks.init_one_layer_network(input_data, output_data, n_input, n_classes,
                                                     training_epochs=i, display_step=display_step)
        accurancy.append(n_classes - x)
        loss.append(y)
        train.append(i)
    create_graph(train, accurancy, ("Number of Training Epochs", "Accurancy", "Model " + model_name),
                 "results/acc_" + model_name + "_0.png")
    create_graph(train, loss, ("Number of Training Epochs", "Accurancy", "Model " + model_name),
                 "results/los" + model_name + "_0.png")


def experiment_changeable_1l_by_epoch(input_data, output_data, n_input, n_classes, n_neurons, model_name,
                                      training_epochs_min=5000, training_epochs_max=20000, display_step=20000):
    accurancy = []
    loss = []
    train = []
    for i in range(training_epochs_min, training_epochs_max, 5000):
        x, y, _, _ = networks.init_multilayer_network(input_data, output_data, n_input, [n_neurons], n_classes, 1,
                                                      training_epochs=i, display_step=display_step)
        print("!! acc " + str(x))
        accurancy.append(n_classes - x)
        loss.append(y)
        train.append(i)
    create_graph(train, accurancy, ("Number of Training Epochs", "Accurancy", "Model " + "g32"),
                 "results/acc_" + model_name + "_1.png")
    create_graph(train, loss, ("Number of Training Epochs", "Accurancy", "Model " + model_name),
                 "results/los" + model_name + "_1.png")


def experiment_changeable_1l(input_data, output_data, n_input, n_classes, n_neurons, model_name,
                             training_epochs=15000, display_step=20000):
    accurancy = []
    loss = []
    n_hidden = []
    for i in n_neurons:
        x, y, _, _ = networks.init_multilayer_network(input_data, output_data, n_input, [i], n_classes, 1,
                                                      training_epochs=training_epochs, display_step=display_step)
        accurancy.append(n_classes - x)
        loss.append(y)
        n_hidden.append(i)
    create_graph(n_hidden, accurancy, ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                 "results/acc_" + model_name + "_1.png", n_neurons)
    create_graph(n_hidden, loss, ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                 "results/los" + model_name + "_1.png", n_neurons)


def experiment_changeable_2l(input_data, output_data, n_input, n_classes, n_nerouns, model_name,
                             training_epochs=15000, display_step=20000):
    accurancy = []
    tikcs = []
    loss = []
    n_hidden = []
    z = 0
    for neurons in n_nerouns:
        x, y, _, _ = networks.init_multilayer_network(input_data, output_data, n_input, neurons, n_classes, 2,
                                                      training_epochs=training_epochs, display_step=display_step)
        accurancy.append(n_classes - x)
        loss.append(y)
        n_hidden.append(z)
        tikcs.append(str(neurons))
        z += 1
    create_graph(n_hidden, accurancy, ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                 "results/acc_" + model_name + "_2.png", tikcs)
    create_graph(n_hidden, loss, ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                 "results/los_" + model_name + "_2.png", tikcs)


def experiment_changeable_nl(input_data, output_data, n_input, n_classes, n_nerouns, model_name, n_layers,
                             training_epochs=15000, display_step=20000):
    accurancy = []
    tikcs = []
    loss = []
    n_hidden = []
    z = 0
    for neurons in n_nerouns:
        x, y, _, _ = networks.init_multilayer_network(input_data, output_data, n_input, neurons, n_classes, n_layers,
                                                      training_epochs=training_epochs, display_step=display_step)
        accurancy.append(n_classes - x)
        loss.append(y)
        n_hidden.append(z)
        tikcs.append(str(neurons))
        z += 1
    create_graph(n_hidden, accurancy, ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                 "results/acc_" + model_name + "_" + str(n_layers) + ".png", tikcs)
    create_graph(n_hidden, loss, ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                 "results/los_" + model_name + "_" + str(n_layers) + ".png", tikcs)


def experiment_changeable_rl(input_data, output_data, n_input, n_classes, n_neurons, model_name, n_layers,
                             training_epochs=15000, display_step=20000):
    accurancy = []
    loss = []
    n_hidden = []
    for i in n_neurons:
        x, y, _, _ = networks.init_recurrent_network(input_data, output_data, n_input, i, n_classes, n_layers,
                                                     training_epochs=training_epochs, display_step=display_step)
        accurancy.append(n_classes - x)
        loss.append(y)
        n_hidden.append(i)
    create_graph(n_hidden, accurancy, ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                 "results/acc_" + model_name + "_r" + str(n_layers) + ".png", n_neurons)
    create_graph(n_hidden, loss, ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                 "results/los" + model_name + "_r" + str(n_layers) + ".png", n_neurons)


def experiment_x(input_data, output_data, n_input, n_layers, n_classes, training_epochs=15000, display_step=20000):
    left_blocks = []
    right_blocks = []
    output_blocks = []
    real_results = []
    predict_results = []
    total_accuracy = 0
    for i in range(0, 8):
        left_blocks.append(input_data[:, 4 * i: 4 * i + 4])
        right_blocks.append(input_data[:, 32 + 4 * i: 32 + 4 * i + 4])
        output_blocks.append(output_data[:, 4 * i: 4 * i + 4])

    for i in range(0, 8):
        z, _, x, y = networks.init_multilayer_network(np.concatenate((left_blocks[i], right_blocks[i]), axis=1),
                                                      output_blocks[i], n_input, [n_layers],
                                                      n_classes, 1, training_epochs=training_epochs,
                                                      display_step=display_step)
        predict_results.append(x)
        real_results.append(y)
        total_accuracy += z
    print(total_accuracy)
    return total_accuracy


def experiment_changeable_x(input_data, output_data, n_input, n_classes, left_b, right_b,
                            training_epochs=15000, display_step=20000):
    accurancy = []
    n_hidden = []
    for i in range(left_b, right_b + 1, 4):
        x = experiment_x(input_data, output_data, n_input, i, n_classes)
        accurancy.append(n_classes * 8 - x)
        n_hidden.append(i)
    create_graph(n_hidden, accurancy, ("Number of neurons on hidden layer", "Accurancy", "GOST"),
                 "results/acc_x_1.png", range(left_b, right_b + 1, 4))


def create_graph(x, y, legend: list, filename: str, ticks=None):
    plt.plot(x, y)
    if ticks is not None:
        plt.xticks(x, ticks)

    plt.xlabel(legend[0])
    plt.ylabel(legend[1])

    plt.title(legend[2])

    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    add_feistel_model(models, max_iteration=8)
    add_custom_model(models, "g4-4", lambda x: utils.split_by_bit(x, 4), lambda x: utils.split_by_bit(x, 4), 1)
    add_custom_model(models, "g4-8", lambda x: utils.split_by_bit(x, 8), lambda x: utils.split_by_bit(x, 8), 1)
    add_custom_model(models, "g4-16", lambda x: utils.split_by_bit(x, 16), lambda x: utils.split_by_bit(x, 16), 2)
    add_custom_model(models, "g4-32", lambda x: utils.split_by_bit(x, 32), lambda x: utils.split_by_bit(x, 32), 4)
    # Network Parameters
    n_input = 32
    n_classes = 32

    model_name = "g4-32"
    input_data, output_data = get_test_data(model_name=model_name, storage=models)
    print(input_data.shape)
    # experiment_changeable_0l(input_data, output_data, n_input, n_classes, model_name)
    experiment_changeable_1l_by_epoch(input_data, output_data, n_input, n_classes, 64, model_name)

    # model_name = "f3-7"
    # for model_name in ["f1-0", "f1-7", "f2-0", "f2-7", "f3-0", "f3-7", "f4-0", "f4-7", "f5-0", "f5-7", "f6-0", "f6-7", "f7-0", "f7-7", "f8-0", "f8-7"]:
    #     print()
    #     print(model_name)
    #     print()
    #     input_data, output_data = get_test_data(model_name=model_name, storage=models)
    #     print("1-Layer")
    #     experiment_changeable_1l(input_data, output_data, n_input, n_classes, [64, 128], model_name, training_epochs=50000)
    #
    #     print("2-Layer")
    #     experiment_changeable_2l(input_data, output_data, n_input, n_classes, [[64, 64]], model_name, training_epochs=50000)
    #
    #     print("3-Layer")
    #     experiment_changeable_nl(input_data, output_data, n_input, n_classes, [[64, 64, 64]], model_name, n_layers=3, training_epochs=50000)
    #
    #     print("4-Layer")
    #     experiment_changeable_nl(input_data, output_data, n_input, n_classes, [[64, 64, 64, 64]], model_name, n_layers=4, training_epochs=50000)

    # experiment_changeable_2l(input_data, output_data, n_input, n_classes, [[32, 32], [64, 64], [128, 128]], model_name, training_epochs=50000)
    # experiment_changeable_rl(input_data, output_data, n_input, n_classes, [[64, 64, 64]], model_name, n_layers=3, training_epochs=50000)


    # experiment_changeable_nl(input_data, output_data, n_input, n_classes, [[64, 64, 64]], model_name, n_layers=3, training_epochs=50000)
    # experiment_changeable_nl(input_data, output_data, n_input, n_classes, [[64, 64, 64, 64, 64, 64, 64, 64]], model_name, n_layers=8)


    #
    # for model_name in models.keys():
    #     print(model_name)
    #     input_data, output_data = get_test_data(model_name=model_name, storage=models)
    #
    #     experiment_changeable_0l(input_data, output_data, n_input, n_classes, model_name)
    #     experiment_changeable_1l(input_data, output_data, n_input, n_classes, [32, 64], model_name)
    #     experiment_changeable_2l(input_data, output_data, n_input, n_classes, [[32, 32], [64, 64]], model_name)
    #     # experiment_changeable_rl(input_data, output_data, n_input, n_classes, [32, 64], model_name)
    #     # experiment_changeable_1l(input_data, output_data, n_input, n_classes,  list(range(32, 65, 8)), model)
    #     # experiment_changeable_2l(input_data, output_data, n_input, n_classes,
    #                              #[[32, 32], [32, 64], [64, 64], [128, 128]], model)
    #     #experiment_changeable_rl(input_data, output_data, n_input, n_classes, list(range(32, 65, 8)), model)
    #     print()

    # model = "g4l"
    # n_input = 8
    # n_classes = 8
    # input_data, output_data = get_test_data(model=model)
    #
    # experiment_changeable_0l(input_data, output_data, n_input, n_classes, model)
    # experiment_changeable_1l(input_data, output_data, n_input, n_classes, list(range(32, 129, 8)), model)
    # experiment_changeable_2l(input_data, output_data, n_input, n_classes, [[64, 64], [128, 128]], model, training_epochs=100000)

    #experiment_changeable_nl(input_data, output_data, n_input, n_classes, [[64, 64, 64]], model, 3,
                             #training_epochs=500000)
    # experiment_changeable_rl(input_data, output_data, n_input, n_classes, list(range(32, 65, 8)), model)
    #experiment_changeable_rl(input_data, output_data, n_input, n_classes, [[64, 64]], model, 2)

    #
    # input_data, output_data = get_test_values(model="g1")
    # print("Experiment g1\n\n")
    # experiment_changeable_0l(input_data, output_data, n_input, n_classes, "g1")
    #
    # input_data, output_data = get_test_values(model="g2")
    # print("Experiment g2\n\n")
    # experiment_changeable_0l(input_data, output_data, n_input, n_classes, "g2")

    # n_input = 8
    # n_classes = 4
    # input_data, output_data = get_test_values1([64, 32])
    # print("Experiment g3\n\n")
    # # experiment_x(input_data, output_data, 8, 4)
    # experiment_changeable_x(input_data, output_data, n_input, n_classes, 8, 32)
    # experiment_changeable_0l(input_data, output_data, n_input, n_classes, "g3")
    # experiment_changeable_1l(input_data, output_data, n_input, n_classes, 8, 32, "g3")
    # experiment_changeable_2l(input_data, output_data, n_input, n_classes, 8, 32, "g3")
