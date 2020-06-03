import numpy as np

import networks
import model
import utils


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


def experiment_0l_by_epoch(input_data, output_data, n_input: int, n_classes: int, model_name: str,
                           display_step: int = 20000, filenames: (str, str) = None,
                           training_epochs_min: int = 5000, training_epochs_max: int = 30000,
                           training_epoch_step: int = 5000):
    accuracy = []
    loss = []
    train = []

    if filenames is None:
        accuracy_file = "results/acc_" + model_name + "_0layer.png"
        loss_file = "results/loss_" + model_name + "_0layer.png"
    else:
        accuracy_file = filenames[0]
        loss_file = filenames[1]

    for i in range(training_epochs_min, training_epochs_max, training_epoch_step):
        x, y, _, _ = networks.init_one_layer_network(input_data, output_data, n_input, n_classes,
                                                     training_epochs=i, display_step=display_step)
        accuracy.append(n_classes - x)
        loss.append(y)
        train.append(i)

    utils.Graphic.create_graph(train, accuracy,
                               ("Number of Training Epochs", "Accurancy", "Model " + model_name),
                               accuracy_file)
    utils.Graphic.create_graph(train, loss, ("Number of Training Epochs", "Accurancy", "Model " + model_name),
                               loss_file)


def experiment_1l_by_epoch(input_data, output_data, n_input: int, n_classes: int, n_neurons: int,
                           model_name: str, display_step: int = 20000, filenames: (str, str) = None,
                           training_epochs_min: int = 5000, training_epochs_max: int = 30000,
                           training_epoch_step: int = 5000):
    accuracy = []
    loss = []
    train = []

    if filenames is None:
        accuracy_file = "results/acc_" + model_name + "_1layer.png"
        loss_file = "results/loss_" + model_name + "_1layer.png"
    else:
        accuracy_file = filenames[0]
        loss_file = filenames[1]

    for i in range(training_epochs_min, training_epochs_max, training_epoch_step):
        x, y, _, _ = networks.init_multilayer_network(input_data, output_data, n_input, [n_neurons], n_classes, 1,
                                                      training_epochs=i, display_step=display_step)
        print("!! acc " + str(x))
        accuracy.append(n_classes - x)
        loss.append(y)
        train.append(i)

    print(accuracy)
    utils.Graphic.create_graph(train, accuracy, ("Number of Training Epochs", "Accurancy", "Model " + model_name),
                               accuracy_file)
    utils.Graphic.create_graph(train, loss, ("Number of Training Epochs", "Accurancy", "Model " + model_name),
                               loss_file)


def experiment_1l_by_layers(input_data, output_data, n_input: int, n_classes: int, n_neurons: list(),
                            model_name: str, display_step: int = 20000, filenames: (str, str) = None,
                            training_epochs:int = 15000):
    accuracy = []
    loss = []
    n_hidden = []

    if filenames is None:
        accuracy_file = "results/acc_" + model_name + "_1layer.png"
        loss_file = "results/loss_" + model_name + "_1layer.png"
    else:
        accuracy_file = filenames[0]
        loss_file = filenames[1]

    for i in n_neurons:
        x, y, _, _ = networks.init_multilayer_network(input_data, output_data, n_input, [i], n_classes, 1,
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


def experiment_nl_by_layers(input_data, output_data, n_input: int, n_classes: int, n_neurons: list(), n_layers: int,
                            model_name: str, display_step: int = 20000, filenames: (str, str) = None,
                            training_epochs:int = 15000):
    accuracy = []
    ticks = []
    loss = []
    n_hidden = []
    z = 0

    if filenames is None:
        accuracy_file = "results/acc_{}_{}layer.png".format(model_name, n_layers)
        loss_file = "results/loss_{}_{}layer.png".format(model_name, n_layers)
    else:
        accuracy_file = filenames[0]
        loss_file = filenames[1]

    for neurons in n_neurons:
        x, y, _, _ = networks.init_multilayer_network(input_data, output_data, n_input, neurons, n_classes, n_layers,
                                                      training_epochs=training_epochs, display_step=display_step)
        accuracy.append(n_classes - x)
        loss.append(y)
        n_hidden.append(z)
        ticks.append(str(neurons))
        z += 1

    utils.Graphic.create_graph(n_hidden, accuracy,
                               ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                               accuracy_file, ticks)
    utils.Graphic.create_graph(n_hidden, loss,
                               ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                               loss_file, ticks)


def experiment_rl_by_layers(input_data, output_data, n_input: int, n_classes: int, n_neurons: list(), n_layers: int,
                            model_name: str, display_step: int = 20000, filenames: (str, str) = None,
                            training_epochs:int = 15000):
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

if __name__ == "__main__":
    models = model.ModelStorage()
    models.add_standart_models()

	# Caclulate model f3-7 with 1 hidden layer NN. 
    model_name = "f3-7"
    model = models.get_model(model_name=model_name)
    input_data, output_data = model.get_test_data()
    utils.Graphic.y_limits = (0, 16)
    utils.Graphic.y_percent = True
    experiment_1l_by_layers(input_data, output_data, model.input_size, model.output_size, [32, 64, 128], model_name)

