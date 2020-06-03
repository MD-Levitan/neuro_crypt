import numpy as np

import networks
import model
import utils


def experiment_0l_by_epoch(input_data, output_data, n_input: int, n_classes: int, model_name: str,
                           display_step: int = 25000, filenames: (str, str) = None,
                           training_epochs_min: int = 5000, training_epochs_max: int = 26000,
                           training_epoch_step: int = 5000):
    accuracy = []
    accuracy1 = []
    loss = []
    train = []

    var_loss = 0
    var_accuracy = 0
    var_step = 0
    var_time = 0

    if filenames is None:
        accuracy_file = "results/" + model_name + "_0.png"
        loss_file = "results/l_" + model_name + "_0.png"
    else:
        accuracy_file = filenames[0]
        loss_file = filenames[1]

    for i in range(training_epochs_min, training_epochs_max, training_epoch_step):
        x, y, x1, y2, timer, _, _ = networks.init_one_layer_network(input_data, output_data, n_input, n_classes,
                                                     training_epochs=i, display_step=display_step)
        accuracy.append(n_classes - x)
        accuracy1.append(n_classes - x1)
        loss.append(y)
        train.append(i)

        if (n_classes - x) > var_accuracy:
            var_accuracy = (n_classes - x)
            var_loss = y
            var_step = i
            var_time = timer

    utils.Logger.save_result(model_name=model_name, n_layers=0, n_params=None,
                             input_size=n_input, output_size=n_classes,
                             accuracy=var_accuracy, loss=var_loss, number_iteration=var_step, time=var_time)

    utils.Graphic.create_double_graph(train, accuracy, accuracy1,
                               ("Number of Training Epochs", "Accurancy", "Model " + model_name),
                               accuracy_file)
    utils.Graphic.create_graph(train, loss, ("Number of Training Epochs", "Accurancy", "Model " + model_name),
                               loss_file)


def experiment_1l_by_epoch(input_data, output_data, n_input: int, n_classes: int, n_neurons: int,
                           model_name: str, display_step: int = 20000, filenames: (str, str) = None,
                           training_epochs_min: int = 5000, training_epochs_max: int = 26000,
                           training_epoch_step: int = 5000):
    accuracy = []
    accuracy1 = []
    loss = []
    train = []

    var_loss = 0
    var_accuracy = 0
    var_step = 0
    var_time = 0

    if filenames is None:
        accuracy_file = "results/" + model_name + "_1_x.png"
        loss_file = "results/l" + model_name + "_1_xlayer.png"
    else:
        accuracy_file = filenames[0]
        loss_file = filenames[1]

    for i in range(training_epochs_min, training_epochs_max, training_epoch_step):
        x, y, x1, y1, timer, _, _ = networks.init_multilayer_network(input_data, output_data, n_input, [n_neurons],
                                                                     n_classes, 1, training_epochs=i,
                                                                     display_step=display_step)
        accuracy.append(n_classes - x)
        accuracy1.append(n_classes - x1)
        loss.append(y)
        train.append(i)

        if (n_classes - x) > var_accuracy:
            var_accuracy = (n_classes - x)
            var_loss = y
            var_step = i
            var_time = timer

    utils.Logger.save_result(model_name=model_name, n_layers=1, n_params=[n_neurons],
                             input_size=n_input, output_size=n_classes,
                             accuracy=var_accuracy, loss=var_loss, number_iteration=var_step, time=var_time)

    utils.Graphic.create_double_graph(train, accuracy, accuracy1, ("Number of Training Epochs", "Accurancy", "Model " + model_name),
                               accuracy_file)
    utils.Graphic.create_graph(train, loss, ("Number of Training Epochs", "Accurancy", "Model " + model_name),
                               loss_file)


def experiment_1l_by_layers(input_data, output_data, n_input: int, n_classes: int, n_neurons: list(),
                            model_name: str, display_step: int = 20000, filenames: (str, str) = None,
                            training_epochs: int = 15000):
    accuracy = []
    accuracy1 = []
    loss = []

    if filenames is None:
        accuracy_file = "results/" + model_name + "_1.png"
        loss_file = "results/l" + model_name + "_1.png"
    else:
        accuracy_file = filenames[0]
        loss_file = filenames[1]

    for i in n_neurons:
        x, y, x1, y1, _, _, _ = networks.init_multilayer_network(input_data, output_data, n_input, [i], n_classes, 1,
                                                      training_epochs=training_epochs, display_step=display_step)
        accuracy.append(n_classes - x)
        accuracy1.append(n_classes - x1)
        loss.append(y)

    utils.Graphic.create_double_graph(n_neurons, accuracy, accuracy1,
                               ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                               accuracy_file, n_neurons)
    utils.Graphic.create_graph(n_neurons, loss,
                               ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                               loss_file, n_neurons)


def experiment_nl_by_layers(input_data, output_data, n_input: int, n_classes: int, n_neurons: list(), n_layers: int,
                            model_name: str, display_step: int = 20000, filenames: (str, str) = None,
                            training_epochs: int = 15000):
    accuracy = []
    accuracy1 = []
    ticks = []
    loss = []
    n_hidden = []
    z = 0

    if filenames is None:
        accuracy_file = "results/{}_{}.png".format(model_name, n_layers)
        loss_file = "results/l_{}_{}.png".format(model_name, n_layers)
    else:
        accuracy_file = filenames[0]
        loss_file = filenames[1]

    for neurons in n_neurons:
        x, y, x1, y1, timer, _, _ = networks.init_multilayer_network(input_data, output_data, n_input, neurons, n_classes, n_layers,
                                                      training_epochs=training_epochs, display_step=display_step)
        accuracy.append(n_classes - x)
        accuracy1.append(n_classes - x1)
        loss.append(y)
        n_hidden.append(z)
        ticks.append(str(neurons))
        z += 1
        utils.Logger.save_result(model_name=model_name, n_layers=n_layers, n_params=neurons,
                                 input_size=n_input, output_size=n_classes,
                                 accuracy=(n_classes - x), loss=y, number_iteration=training_epochs, time=timer)


    utils.Graphic.create_double_graph(n_hidden, accuracy, accuracy1,
                               ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                               accuracy_file, ticks)
    utils.Graphic.create_graph(n_hidden, loss,
                               ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                               loss_file, ticks)


def experiment_nl_by_layers_multi(input_data, output_data, n_input: int, n_classes: int, n_neurons: list(), n_layers: int,
                                  model_name: str, display_step: int = 20000, filenames: (str, str) = None,
                                  training_epochs:int = 15000):
    accuracy = []
    ticks = []
    loss = []
    n_hidden = []
    z = 0

    for neurons in n_neurons:
        x, y, x1, y1, timer, _, _ = networks.init_multilayer_network(input_data, output_data, n_input, neurons,
                                                                     n_classes, n_layers,
                                                                     training_epochs=training_epochs,
                                                                     display_step=display_step)
        accuracy.append(n_classes - x)
        loss.append(y)
        n_hidden.append(z)
        ticks.append(str(neurons))
        z += 1

    utils.Graphic.add_graph(n_hidden, accuracy, model_name, ticks)


def experiment_rl_by_layers(input_data, output_data, n_input: int, n_classes: int, n_neurons: list(), n_layers: int,
                            model_name: str, display_step: int = 20000, filenames: (str, str) = None,
                            training_epochs: int = 15000):
    accuracy = []
    loss = []
    n_hidden = []

    if filenames is None:
        accuracy_file = "results/acc_{}_{}R-layer.png".format(model_name, n_layers)
        loss_file = "results/loss_{}_{}R-layer.png".format(model_name, n_layers)
    else:
        accuracy_file = filenames[0]
        loss_file = filenames[1]

    for i in n_neurons:
        x, y, _, _ = networks.init_recurrent_network(input_data, output_data, n_input, i, n_classes, n_layers,
                                                     training_epochs=training_epochs, display_step=display_step)
        accuracy.append(n_classes - x)
        loss.append(y)
        n_hidden.append(i)

    utils.Graphic.create_graph(n_hidden, accuracy,
                               ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                               accuracy_file, n_neurons)
    utils.Graphic.create_graph(n_hidden, loss,
                               ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                               loss_file, n_neurons)
    
# def experiment_x(input_data, output_data, n_input, n_layers, n_classes, training_epochs=15000, display_step=20000):
#     left_blocks = []
#     right_blocks = []
#     output_blocks = []
#     real_results = []
#     predict_results = []
#     total_accuracy = 0
#     for i in range(0, 8):
#         left_blocks.append(input_data[:, 4 * i: 4 * i + 4])
#         right_blocks.append(input_data[:, 32 + 4 * i: 32 + 4 * i + 4])
#         output_blocks.append(output_data[:, 4 * i: 4 * i + 4])
#
#     for i in range(0, 8):
#         z, _, x, y = networks.init_multilayer_network(np.concatenate((left_blocks[i], right_blocks[i]), axis=1),
#                                                       output_blocks[i], n_input, [n_layers],
#                                                       n_classes, 1, training_epochs=training_epochs,
#                                                       display_step=display_step)
#         predict_results.append(x)
#         real_results.append(y)
#         total_accuracy += z
#     print(total_accuracy)
#     return total_accuracy
#
#
# def experiment_changeable_x(input_data, output_data, n_input, n_classes, left_b, right_b,
#                             training_epochs=15000, display_step=20000):
#     accurancy = []
#     n_hidden = []
#     for i in range(left_b, right_b + 1, 4):
#         x = experiment_x(input_data, output_data, n_input, i, n_classes)
#         accurancy.append(n_classes * 8 - x)
#         n_hidden.append(i)
#     create_graph(n_hidden, accurancy, ("Number of neurons on hidden layer", "Accurancy", "GOST"),
#                  "results/acc_x_1.png", range(left_b, right_b + 1, 4))


if __name__ == "__main__":
    models = model.ModelStorage()
    models.add_standart_models()
    # utils.Logger.clear_files()

    for i in range(2, 3):
        for j in (0, 7):
            model_name = "f{}-{}".format(i, j)
            print(model_name)
            model = models.get_model(model_name=model_name)
            input_data, output_data = model.get_test_data()
            utils.Graphic.y_limits = (0, model.output_size)
            utils.Graphic.y_percent = True

            experiment_0l_by_epoch(input_data, output_data, model.input_size, model.output_size, model_name)

            experiment_1l_by_layers(input_data, output_data, model.input_size, model.output_size, [8, 16, 32, 64],
                                     model_name)
            experiment_1l_by_epoch(input_data, output_data, model.input_size, model.output_size,
                                    32, model_name)
            # experiment_nl_by_layers(input_data, output_data, model.input_size, model.output_size,
            #                         [[32, 32], [32, 64], [64, 32], [64, 64]], 2, model_name)
            # experiment_nl_by_layers(input_data, output_data, model.input_size, model.output_size,
            #                         [[32, 32, 32], [32, 64, 32], [64, 64, 64]], 3, model_name)

    # utils.Graphic.y_limits_percent = (0, 102)
    # utils.Graphic.y_limits = (0, 16)
    # utils.Graphic.y_percent = True
    # utils.Graphic.init_graph()
    # for j in range(0, 8):
    #     model_name = "f1-{}".format(j)
    #     print(model_name)
    #     model = models.get_model(model_name=model_name)
    #     input_data, output_data = model.get_test_data()
    #
    #
    #     experiment_nl_by_layers_multi(input_data, output_data, model.input_size, model.output_size,
    #                                   [[8], [16], [32], [48], [64]], 1, model_name)
    # utils.Graphic.save_graph(("Number of neurons on hidden layer", "Accurancy, %", "Models f1"),
    #                          list(map(lambda x: "f1-{}".format(x), range(0, 8))), "results/f0_diff.png")


    # for i in (32, ):
    #     model_name = "g{}".format(i)
    #     print(model_name)
    #     model = models.get_model(model_name=model_name)
    #     input_data, output_data = model.get_test_data()
    #     utils.Graphic.y_limits = (0, model.output_size)
    #     utils.Graphic.y_percent = True
    #     experiment_0l_by_epoch(input_data, output_data, model.input_size, model.output_size, model_name)
    #
    #     experiment_1l_by_layers(input_data, output_data, model.input_size, model.output_size, [4, 8, 12, 16, 20, 24, 28, 32, 64], model_name)
    #     experiment_1l_by_epoch(input_data, output_data, model.input_size, model.output_size,
    #                            64, model_name)
    #     #experiment_nl_by_layers(input_data, output_data, model.input_size, model.output_size,
    #     #                        [[64, 64]], 2, model_name)


    # total_input = []
    # total_output = []
    # for i in range(0, 8):
    #     model_name = "f1-{}".format(i)
    #     print(model_name)
    #     model = models.get_model(model_name=model_name)
    #     input_data, output_data = model.get_test_data()
    #     utils.Graphic.y_limits = (0, model.output_size)
    #     utils.Graphic.y_percent = True
    #     total_input.append(input_data)
    #     total_output.append(output_data)
    # experiment_nl_by_layers_var(total_input, total_output, model.input_size, model.output_size, [[4], [8], [12], [16], [20], [24], [28], [32]], 1, model_name)

        # experiment_0l_by_epoch(input_data, output_data, model.input_size, model.output_size, model_name)
        #
        # experiment_1l_by_layers(input_data, output_data, model.input_size, model.output_size, [4, 8, 12, 16, 20, 24, 28, 32, 64], model_name)
        # experiment_1l_by_epoch(input_data, output_data, model.input_size, model.output_size,
        #                        64, model_name)
        #experiment_nl_by_layers(input_data, output_data, model.input_size, model.output_size,
        #                        [[64, 64]], 2, model_name)



    # model_name = "f3-7"
    # model = models.get_model(model_name=model_name)
    # input_data, output_data = model.get_test_data()
    # # experiment_changeable_0l(input_data, output_data, n_input, n_classes, model_name)
    # #utils.Graphic.y_limits = (0, 32)
    # #utils.Graphic.y_percent = True
    # experiment_nl_by_layers(input_data, output_data, model.input_size, model.output_size, [[64, 64, 64]], 3, model_name, training_epochs=30000)

    # model_name = "g4-32"
    # model = models.get_model(model_name=model_name)
    # input_data, output_data = model.get_test_data()
    # # experiment_changeable_0l(input_data, output_data, n_input, n_classes, model_name)
    # utils.Graphic.y_limits = (0, 32)
    # utils.Graphic.y_percent = True
    # experiment_1l_by_epoch(input_data, output_data, model.input_size, model.output_size, 64, model_name,
    #                        training_epochs_min=1000, training_epochs_max=10000, training_epoch_step=3000)

    # for i in (4, 8, 16, 32):
    #     model_name = "f4-{}".format(i)
    #     model = models.get_model(model_name=model_name)
    #     input_data, output_data = model.get_test_data()
    #     utils.Graphic.y_limits = (0, i)
    #     #utils.Graphic.y_percent = True
    #     experiment_0l_by_epoch(input_data, output_data, model.input_size, model.output_size, model_name)
    #
    #     # experiment_1l_by_layers(input_data, output_data, model.input_size, model.output_size, [32, 64, 128], model_name)


    # model_name = "f3-7"
    # model = models.get_model(model_name=model_name)
    # input_data, output_data = model.get_test_data()
    # utils.Graphic.y_limits = (0, 16)
    # utils.Graphic.y_percent = True
    # experiment_1l_by_layers(input_data, output_data, model.input_size, model.output_size, [32, 64, 128], model_name)

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
