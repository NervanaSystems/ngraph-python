import pickle
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import csv
import pdb
import numpy as np

def mov_avg(loss=None, half_window_size = 5):
    steps = len(loss['train_loss'])
    new_loss = {'loss': [], 'iteration': loss['iteration'][half_window_size:steps-half_window_size]} 
    new_loss['loss'] = [loss['train_loss'][i-half_window_size:i+half_window_size+1] for i in range(half_window_size, steps-half_window_size)] 
    new_loss['loss'] = np.mean(np.array(new_loss['loss']), axis=1)
    return new_loss

def read_csv(filename):
    loss = {'train_loss': [], 'iteration': []}
    with open(filename, 'rb') as csvfile:
        loss_reader = csv.DictReader(csvfile)
        for row in loss_reader:
            loss['iteration'].append(float(row['Step']))
            loss['train_loss'].append(float(row['Value']))

    loss['iteration'] = np.array(loss['iteration'])
    loss['train_loss'] = np.array(loss['train_loss'])
    return loss

loss_names = ['sgd_gpu', 'rmsprop_gpu']
base_path = '/nfs/site/home/gkeskin/work/ngraph/private-ngraph/examples/inceptionv3/'
losses = {}
half_window_size = 200 
for loss in loss_names:
    tmp_loss = pickle.load( open( base_path + ("losses_%s.pkl" % loss), "rb" ))
    losses[loss] = {}
    losses[loss] = mov_avg(loss=tmp_loss, half_window_size = half_window_size)

losses['tf'] = read_csv(base_path + 'loss_tflow_bs4.csv')
losses['tf'] = mov_avg(losses['tf'], half_window_size = 10) 

fig, ax = plt.subplots(figsize=(12,6))
for loss in losses.keys():
    line = ax.plot(losses[loss]['iteration'], losses[loss]['loss'], '--', linewidth=2,
                 label=loss)

plt.xlabel('Iteration', fontsize=18)
plt.ylabel('Cross Entropy Loss', fontsize=18)
fig.suptitle('Training Loss for Mini Inceptionv3 Network for Tensorflow and Ngraph', fontsize=20)
ax.legend()
ax.grid(True)
fig.savefig('losses.png', dpi=128)
