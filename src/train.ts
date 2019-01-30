import { Sequential, Model } from "@tensorflow/tfjs";
import { MnistDataset } from "./utils/data";
import { Config } from "./models/config";


export async function train(model: Sequential | Model, mnist: MnistDataset, config: Config) {

  const { images: trainImages, labels: trainLabels } = mnist.getTrainData();
  model.summary();

  const validationSplit = 0.15;

  await model.fit(trainImages, trainLabels, {
    epochs: config.epochs,
    batchSize: config.batchSize,
    validationSplit,
    stepsPerEpoch: 1
  });
}
