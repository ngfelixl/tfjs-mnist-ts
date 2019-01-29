import { Sequential, sequential, layers } from '@tensorflow/tfjs';

export class DenseNetwork {
  model: Sequential;

  constructor(inputHeight: number, inputWidth: number) {
    this.model = sequential();

    this.model.add(layers.flatten({
      inputShape: [ inputHeight, inputWidth, 1 ]
    }));

    this.model.add(layers.dense({
      units: 42,
      activation: 'relu'
    }));

    this.model.add(layers.dense({
      units: 10,
      activation: 'softmax'
    }));

    const optimizer = 'rmsprop';
    this.model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: [ 'accuracy' ]
    });
  }
}
