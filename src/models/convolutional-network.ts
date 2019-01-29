import { Sequential, sequential, layers } from "@tensorflow/tfjs";

export class ConvolutionalNetwork {
  model: Sequential;

  constructor(inputHeight: number, inputWidth: number) {
    this.model = sequential();

    this.model.add(layers.conv2d({
      inputShape: [ inputHeight, inputWidth, 1 ],
      kernelSize: 3,
      filters: 16,
      activation: 'relu'
    }));

    this.model.add(layers.maxPooling2d({
      poolSize: 2,
      strides: 2
    }));

    this.model.add(layers.conv2d({
      kernelSize: 3,
      filters: 32,
      activation: 'relu'
    }));

    this.model.add(layers.maxPooling2d({
      poolSize: 2,
      strides: 2
    }));

    this.model.add(layers.conv2d({
      kernelSize: 3,
      filters: 32,
      activation: 'relu'
    }));

    this.model.add(layers.flatten({}));

    this.model.add(layers.dense({
      units: 64,
      activation: 'relu'
    }));

    this.model.add(layers.dense({
      units: 10,
      activation: 'softmax'
    }));

    // const optimizer = 'rmsprop';
    const optimizer = 'sgd';
    this.model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: [ 'accuracy' ]
    });
  }
}
