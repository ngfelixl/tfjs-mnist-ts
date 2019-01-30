import { Sequential, sequential, layers } from "@tensorflow/tfjs";


export function createSimpleCNN(inputHeight: number, inputWidth: number): Sequential {
  const model = sequential();

  model.add(layers.conv2d({
    inputShape: [ inputHeight, inputWidth, 1 ],
    kernelSize: 3,
    filters: 16,
    activation: 'relu'
  }));

  model.add(layers.maxPooling2d({
    poolSize: 3,
    strides: 3
  }));

  model.add(layers.flatten({}));
  
  model.add(layers.dense({
    units: 10,
    activation: 'softmax'
  }));

  return model;
}
