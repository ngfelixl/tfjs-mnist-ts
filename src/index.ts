import { MnistDataset } from './utils/data';
import { Sequential, Model, train as optimizers } from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';

import { createFFN } from './networks/dense-network';
import { createCNN } from './networks/convolutional-network';
import { createSimpleCNN } from './networks/simple-convolutional-network';
import { parseArguments } from './utils/parse-arguments';
import { train } from './train';
import { evaluate } from './evaluate';
import { Config } from './models/config';

const mnist = new MnistDataset();
const learningRate = 0.015;

function compile(model: Sequential) {
  // const optimizer = 'rmsprop';
  const optimizer = optimizers.rmsprop(learningRate);
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: [ 'accuracy' ]
  });
}

async function run(config: Config) {
  let model: Sequential;
  await mnist.loadData();

  switch (config.model) {
    case 'dense': model = createFFN(28, 28); break;
    case 'conv': model = createCNN(28, 28); break;
    case 'simpleConv': model = createSimpleCNN(28, 28); break;
  }

  compile(model);
  await train(model, mnist, config);
  await evaluate(model, mnist);

  if (config.modelSavePath != null) {
    await model.save(`file://${config.modelSavePath}`);
    console.log(`Saved model to path: ${config.modelSavePath}`);
  }
}

const config = parseArguments();
run(config);
