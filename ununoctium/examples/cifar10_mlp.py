from __future__ import print_function
from geon.backends.graph.graphneon import *

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.set_defaults(backend='dataloader')
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()

# setup data provider
imgset_options = dict(inner_size=32, scale_range=40, aspect_ratio=110,
                      repo_dir=args.data_dir, subset_pct=args.subset_pct)
train = ImageLoader(set_name='train', shuffle=True, do_transforms=True, **imgset_options)
test = ImageLoader(set_name='validation', shuffle=False, do_transforms=False, **imgset_options)

init_uni0 = Uniform(low=-0.002 , high=0.002)
init_uni1 = Uniform(low=-0.1, high=0.1)

opt_gdm = GradientDescentMomentum(learning_rate=0.0001, momentum_coef=0.9)

# set up the model layers
layers = [Affine(nout=200, init=init_uni0, activation=Rectlin()),
          Affine(nout=train.nclasses, axes=(ax.Y,), init=init_uni1, activation=Softmax())]

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

mlp = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(mlp, eval_set=test, **args.callback_args)

np.seterr(divide='raise', over='raise', invalid='raise')
mlp.fit(train, input_axes=(ax.C, ax.H, ax.W), target_axes=(ax.Y,), optimizer=opt_gdm, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

print('Misclassification error = %.1f%%' % (mlp.eval(test, metric=Misclassification())*100))
