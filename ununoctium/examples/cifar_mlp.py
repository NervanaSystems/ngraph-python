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

scale = 1.0 / 32.0
init_uni0 = Uniform(low=-0.1 * scale, high=0.1 * scale)
init_uni1 = Uniform(low=-0.1, high=0.1)

#opt_gdm = GradientDescentMomentum(learning_rate=0.01 * scale, momentum_coef=0.9)
opt_gdm = GradientDescent(learning_rate=0.01 * scale)

# set up the model layers
# TODO switch back to neon cifar_mlp example activations
# TODO use some sort of axis aliasing mechanism so we don't need axes in last layer
layers = [Affine(nout=200, init=init_uni0, activation=be.tanh),
          Affine(nout=train.nclasses, axes=(ax.Y,), init=init_uni1, activation=be.softmax)]

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

mlp = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(mlp, eval_set=test, **args.callback_args)


mlp.fit(train, input_axes=(ax.C, ax.H, ax.W), target_axes=(ax.Y,), optimizer=opt_gdm, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

#TODO
print('Misclassification error = %.1f%%' % (mlp.eval(test, metric=Misclassification())*100))
