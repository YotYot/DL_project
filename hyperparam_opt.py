from hyperopt import hp
from hyperopt import fmin, tpe, space_eval, Trials
from train import agent
import hyperopt.pyll.stochastic
from models import Net
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# num_of_neurons, num_of_hidden_layers,learning_rate, train_data_loader, test_data_loader, epochs, device
best_list = list()
for inputs in range (5,10):
    num_of_inputs = inputs*5
    depth = 10

    ag = agent(num_of_inputs=num_of_inputs, depth=depth)

    # learning_rate_choices = [0.1,0.01,0.001]
    learning_rate_choices = [0.01]

    space = hp.choice('a',
        [
            {
                'num_of_neurons': hp.quniform('num_of_neurons', 10,500,1),
                'num_of_hidden_layers': hp.quniform('num_of_hidden_layers', 1,5,1),
                'learning_rate': hp.choice('learning_rate', learning_rate_choices),
                'epochs': hp.quniform('epochs', 20,20,1)
            }
        ])

    # print (hyperopt.pyll.stochastic.sample(space))

    # minimize the objective over the space
    trials = Trials()
    best = fmin(ag.objective, space, algo=tpe.suggest, max_evals=10, trials=trials)

    print(best)
    print (trials.results)
    print (trials.losses())
    print(space_eval(space, best))
    num_of_neurons_list = [a['misc']['vals']['num_of_neurons'][0] for a in trials.trials]
    num_of_hidden_layers_list = [a['misc']['vals']['num_of_hidden_layers'][0] for a in trials.trials]
    num_of_epochs = [a['misc']['vals']['epochs'][0] for a in trials.trials]
    learning_rate_list = [a['misc']['vals']['learning_rate'][0] for a in trials.trials]
    for i, trial in enumerate(trials.losses()):
        print ("Experiment {:2}: Hidden Layers: {:3}, Neurons: {:3}, Learing Rate: {:3}, Epochs: {:3}, Loss: {:1.5}".
               format(i, num_of_hidden_layers_list[i], num_of_neurons_list[i],learning_rate_choices[learning_rate_list[i]],num_of_epochs[i], trial))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(num_of_neurons_list, num_of_hidden_layers_list, trials.losses())
    ax.set_xlabel('#Neurons')
    ax.set_ylabel('#Layers')
    ax.set_zlabel('Loss')
    plt.show()
    best_list.append(best)
print (best_list)
net = Net(num_of_neurones=int(best['num_of_neurons']), num_of_hidden_layers=int(best['num_of_hidden_layers']), num_of_inputs=ag.hw_model.inputs, num_of_outputs=ag.hw_model.outputs)
net = net.to(ag.device)
print (ag.test(net))
