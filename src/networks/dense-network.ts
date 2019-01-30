import { Sequential, sequential, layers } from '@tensorflow/tfjs';

export function createFFN(inputHeight: number, inputWidth: number): Sequential {
  const model = sequential();

  model.add(layers.flatten({
    inputShape: [ inputHeight, inputWidth, 1 ]
  }));

  model.add(layers.dense({
    units: 42,
    activation: 'relu'
  }));

  model.add(layers.dense({
    units: 10,
    activation: 'softmax'
  }));

  return model;
}
