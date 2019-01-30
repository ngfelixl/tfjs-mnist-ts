import { ArgumentParser } from 'argparse';
import { Config } from 'src/models/config';

export function parseArguments(): Config {
  const argList = [
    { arg: '--epochs', type: 'int', defaultValue: 20, help: 'Number of epochs to train the model for.' },
    { arg: '--batchSize', type: 'int', defaultValue: 128, help: 'Batch size to be used during model training.' },
    { arg: '--model', type: 'string', defaultValue: 'conv', help: 'Model to be used. Choose between `conv`, `simpleConv` and `dense`' },
    { arg: '--modelSavePath', type: 'string', defaultValue: 'result', help: 'Path to which the model will be saved after training.' },
    { arg: '--retrain', type: 'string', defaultValue: false, help: 'If model exists overwrite training.' }
  ];
  
  const parser = new ArgumentParser({
    description: 'TensorFlow.js-Node MNIST',
    addHelp: true
  });
  
  for (let arg of argList) {
    parser.addArgument(arg.arg, {...arg});
  }
  return parser.parseArgs();
}
