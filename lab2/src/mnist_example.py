import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np

train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=lambda data, label: (data.astype(np.float32)/255, label)),
                                      batch_size=32, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=lambda data, label: (data.astype(np.float32)/255, label)),
                                     batch_size=32, shuffle=False)
net = gluon.nn.Sequential()

with net.name_scope():
    net.add(gluon.nn.Dense(128, activation="relu"))
    net.add(gluon.nn.Dense(64, activation="relu"))
    net.add(gluon.nn.Dense(10))

net.collect_params().initialize(mx.init.Normal(sigma=0.05))
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})

epochs = 10
for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(mx.cpu()).reshape((-1, 784))
        label = label.as_in_context(mx.cpu())
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
            loss.backward()
        trainer.step(data.shape[0])
        curr_loss = ndarray.mean(loss).asscalar()
    print("Epoch {}. Current Loss: {}.".format(e, curr_loss))
