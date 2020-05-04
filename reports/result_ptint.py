from pylab import *

input_x = list(range(2, 12))
#result_accuracy = [0.7815533995628356, 0.7689320355653763, 0.7398058295249939, 0.6087378561496735, 0.5378640800714493, 0.4864077627658844, 0.49805825650691987, 0.49126213490962983, 0.5038834899663925, 0.5018834899663925]
#result_loss = [0.46932488977909087, 0.4350429594516754, 0.5035425156354905, 0.7072348535060883, 1.6553226709365845, 48.21521291732788,  777.3107452392578, 14258.725219726562, 287503.7603515625, 580503.7603515625]
result_accuracy = [0.8135922372341156, 0.76601941883564, 0.8145631074905395, 0.6873786389827728, 0.724271833896637, 0.7378640711307526, 0.6757281601428986, 0.6009708762168884, 0.58, 0.572]
result_loss = [0.4554207630455494, 0.5027005076408386, 0.4080037593841553, 0.5359159976243972, 0.4667157411575317, 0.49344966411590574, 0.5992215126752853, 0.7462544322013855]

ax = plt.axes()
ax.set_ylim([0.4, min(max(result_accuracy) + 0.1, 1)])
plot(input_x, result_accuracy)
xlabel('Number of variables boolean function')
ylabel('Accuracy')
grid(True)
show()
#savefig("Accuracy_params.png")


##
result_accuracy = [0.815533971786499, 0.8805825233459472, 0.9126213550567627, 0.8048543632030487, 0.7922330021858215, 0.7524271845817566, 0.6766990184783935, 0.6174757272005081, 0.6074757272005081, 0.6004757272005081]
result_loss = [0.43770390301942824, 0.3317414462566376, 0.3028559772996232, 0.40666568353772165, 0.39652785658836365, 0.4975146263837814, 0.8015267506241799, 1.4045054852962493]

ax = plt.axes()
ax.set_ylim([0.4, min(max(result_accuracy) + 0.1, 1)])
plot(input_x, result_accuracy)
xlabel('Number of variables boolean function')
ylabel('Accuracy')
grid(True)
show()
#savefig("Loss_params.png")


input_x = list(range(2, 12))
#result_accuracy = [0.650485435128212, 0.9582524240016937, 0.9679611682891845, 0.9689320385456085, 0.9640776693820954, 0.9621359169483185, 0.9737864017486573, 0.914563101530075, 0.7912621319293975, 0.6961165010929108]
#result_loss = [0.5381422519683838, 0.10528383061755449, 0.10116118341684341, 0.08858435596339405, 1.3367720436086814, 65.12044744491577, 9018.000984573364, 4840326.575, 4391369676.8, 2152903960166.4]


result_accuracy = [0.8330097079277039, 0.9126213610172271, 0.937864077091217, 0.9592233061790466, 0.9582524299621582, 0.9786407768726348, 0.9902912557125092, 0.9029126107692719, 0.891212, 0.89000]
result_loss = [0.43544345200061796, 0.30147217866033316, 0.1457182300509885, 0.10246582765248605, 0.1645127294077781, 37.59508514404297, 13788.395408248902, 7860310.2125]


ax = plt.axes()
ax.set_ylim([0.4, min(max(result_accuracy) + 0.1, 1)])
plot(input_x, result_accuracy)
xlabel('Number of variables boolean function')
ylabel('Accuracy')
grid(True)
show()
#savefig("Accuracy_params_teta.png")






input_x = list(range(1, 6))
result_accuracy = [0.493232, 0.67999998807907104, 0.7200000166893005,  0.8299999833106995, 0.6800000047683716]

ax = plt.axes()
ax.set_ylim([0.4, min(max(result_accuracy) + 0.1, 1)])
plot(input_x, result_accuracy)
xlabel('Number of hidden layers')
ylabel('Accuracy')
grid(True)
show()


input_x = list(range(1, 6))
result_accuracy = [0.513232, 0.550000011920929, 0.6299999952316284,  0.5699999928474426, 0.5099999904632568]

ax = plt.axes()
ax.set_ylim([0.4, min(max(result_accuracy) + 0.1, 1)])
plot(input_x, result_accuracy)
xlabel('Number of hidden layers')
ylabel('Accuracy')
grid(True)
show()

input_x = list(range(1, 6))
result_accuracy = [0.4913232, 0.5600000023841858, 0.4399999976158142,  0.49000000953674316, 0.5099999904632568]

ax = plt.axes()
ax.set_ylim([0.4, min(max(result_accuracy) + 0.1, 1)])
plot(input_x, result_accuracy)
xlabel('Number of hidden layers')
ylabel('Accuracy')
grid(True)
show()



input_x = list(range(1, 8))
result_accuracy = [0.513232, 0.722000002861023, 0.7169999957084656, 0.7160000026226043, 0.6479999899864197, 0.6549999922513962, 0.617999991774559]

ax = plt.axes()
ax.set_ylim([0.4, min(max(result_accuracy) + 0.1, 1)])
plot(input_x, result_accuracy)
xlabel('Number of hidden layers')
ylabel('Accuracy')
grid(True)
show()


input_x = list(range(1, 8))
result_accuracy = [0.513232, 0.5269999951124191, 0.5299999922513962, 0.5219999879598618, 0.5180000007152558, 0.49699999392032623, 0.5329999834299087]

ax = plt.axes()
ax.set_ylim([0.4, min(max(result_accuracy) + 0.1, 1)])
plot(input_x, result_accuracy)
xlabel('Number of hidden layers')
ylabel('Accuracy')
title('Number of heidden layers to accuracy')
grid(True)
show()

input_x = list(range(1, 8))
result_accuracy = [0.4913232, 0.4909999966621399, 0.5209999978542328, 0.4919999986886978, 0.4739999920129776, 0.5019999980926514, 0.4959999918937683]

ax = plt.axes()
ax.set_ylim([0.4, min(max(result_accuracy) + 0.1, 1)])
plot(input_x, result_accuracy)
xlabel('Number of hidden layers')
ylabel('Accuracy')
grid(True)
show()