export interface Config {
  epochs: number;
  batchSize?: number;
  model?: string;
  modelSavePath?: string;
  retrain?: boolean;
}
