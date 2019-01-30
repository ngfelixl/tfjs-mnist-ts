import { Sequential, Model } from "@tensorflow/tfjs";
import { MnistDataset } from "./utils/data";

export async function evaluate(model: Sequential | Model, mnist: MnistDataset) {
  const { images: testImages, labels: testLabels } = mnist.getTestData();
  const evalOutput = model.evaluate(testImages, testLabels);

  console.log(
    `\nEvaluation result:\n` +
    `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
    `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);
}
