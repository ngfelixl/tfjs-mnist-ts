import { MnistDataset } from './data';
import { ArgumentParser } from 'argparse';
import { Sequential } from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';

import { DenseNetwork } from './models/dense-network';
import { ConvolutionalNetwork } from './models/convolutional-network';
import { SimpleConvolutionalNetwork } from './models/simple-convolutional-network';

const mnist = new MnistDataset();

async function run(epochs: number, batchSize: number, model: string, modelSavePath: string) {
  let nn: { model: Sequential };
  switch (model) {
    case 'dense': nn = new DenseNetwork(28, 28); break;
    case 'conv': nn = new ConvolutionalNetwork(28, 28); break;
    case 'simpleConv': nn = new SimpleConvolutionalNetwork(28, 28); break;
  }
  console.log('Loading data...');
  await mnist.loadData();

  const { images: trainImages, labels: trainLabels } = mnist.getTrainData();
  nn.model.summary();

  // let epochBeginTime;
  // let millisPerStep;

  const validationSplit = 0.15;
  // const numTrainExamplesPerEpoch = trainImages.shape[0] * (1 - validationSplit);
  // const numTrainBatchesPerEpoch = Math.ceil(numTrainExamplesPerEpoch / batchSize);

  console.log('Train model...');

  await nn.model.fit(trainImages, trainLabels, {
    epochs,
    batchSize,
    validationSplit,
    stepsPerEpoch: 1
  });

  const { images: testImages, labels: testLabels } = mnist.getTestData();
  const evalOutput = nn.model.evaluate(testImages, testLabels);

  console.log(
    `\nEvaluation result:\n` +
    `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
    `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

  console.log(
    `\nEvaluation result:\n` +
    `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
    `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

  if (modelSavePath != null) {
    await nn.model.save(`file://${modelSavePath}`);
    console.log(`Saved model to path: ${modelSavePath}`);
  }
}


const parser = new ArgumentParser({
  description: 'TensorFlow.js-Node MNIST Example.',
  addHelp: true
});
parser.addArgument('--epochs', {
  type: 'int',
  defaultValue: 20,
  help: 'Number of epochs to train the model for.'
});
parser.addArgument('--batch_size', {
  type: 'int',
  defaultValue: 128,
  help: 'Batch size to be used during model training.'
});
parser.addArgument('--model', {
  type: 'string',
  defaultValue: 'conv',
  help: 'Model to be used. Choose between `conv` and `dense`'
});
parser.addArgument('--model_save_path', {
  type: 'string',
  help: 'Path to which the model will be saved after training.'
});
const args = parser.parseArgs();

run(args.epochs, args.batch_size, args.model, args.model_save_path);
