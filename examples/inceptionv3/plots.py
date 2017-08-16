import pickle
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import csv
import pdb
import numpy as np

def mov_avg(loss, half_window_size = 5):
    steps = len(loss['loss'])
    new_loss = {'loss': [], 'iteration': loss['iteration'][half_window_size:steps-half_window_size]} 
    new_loss['loss'] = [loss['loss'][i-half_window_size:i+half_window_size+1] for i in range(half_window_size, steps-half_window_size)] 
    new_loss['loss'] = np.mean(np.array(new_loss['loss']), axis=1)
    return new_loss

def read_csv(filename):
    loss = {'loss': [], 'iteration': []}
    with open(filename, 'rb') as csvfile:
        loss_reader = csv.DictReader(csvfile)
        for row in loss_reader:
            loss['iteration'].append(float(row['Step']))
            loss['loss'].append(float(row['Value']))

    loss['iteration'] = np.array(loss['iteration'])
    loss['loss'] = np.array(loss['loss'])
    return loss

base_path = '/home/gkeskin//work/ngraph/private-ngraph/examples/inceptionv3/'
losses = pickle.load( open( base_path + "losses.pkl", "rb" ) )
loss_ngraph = {'loss': [], 'iteration': []}
loss_ngraph['loss'] = np.array(losses['train_loss'])
loss_ngraph['iteration'] = np.array(losses['iteration'])
#loss_ngraph['iteration'] = np.array([(i+1)*2000 for i in range(len(loss_ngraph['loss']))])

loss_tf_momentum = read_csv(base_path + 'losses_batch_8_momentum.csv')
loss_tf_no_momentum = read_csv(base_path + 'losses_batch_8_no_momentum.csv')
 
half_window_size = 3 
loss_tf_no_momentum = mov_avg(loss_tf_no_momentum, half_window_size = 6*half_window_size) 
loss_tf_momentum = mov_avg(loss_tf_momentum, half_window_size = 6*half_window_size) 
loss_ngraph = mov_avg(loss_ngraph, half_window_size = half_window_size)

fig, ax = plt.subplots(figsize=(12,6))
line1 = ax.plot(loss_tf_no_momentum['iteration'], loss_tf_no_momentum['loss'], '--', linewidth=2,
                 label='TF No Momentum')

line2 = ax.plot(loss_ngraph['iteration'], loss_ngraph['loss'], '-', linewidth=2,
                 label='Ngraph No Momentum')

line3 = ax.plot(loss_tf_momentum['iteration'], loss_tf_momentum['loss'], '--', linewidth=2,
                 label='TF Momentum')

plt.xlabel('Iteration', fontsize=18)
plt.ylabel('Cross Entropy Loss', fontsize=18)
fig.suptitle('Training Loss for Mini Inceptionv3 Network for Tensorflow and Ngraph\nBatchSize = 8, RMSProp', fontsize=20)
ax.legend()
ax.grid(True)
fig.savefig('losses.png', dpi=128)
