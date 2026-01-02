using System;
using AForge.Neuro;
using AForge.Neuro.Learning;

namespace NeuralNetwork1
{
    public class AForgeNetwork : BaseNetwork
    {
        private ActivationNetwork _network;
        private BackPropagationLearning _teacher;

        public AForgeNetwork(int[] structure)
        {
            // Создаем слои. Первый элемент structure - входы, остальные - нейроны в слоях
            // AForge.Neuro ActivationNetwork конструктор: (функция, кол-во входов, кол-во нейронов в слоях...)
            int inputs = structure[0];
            int[] layers = new int[structure.Length - 1];
            Array.Copy(structure, 1, layers, 0, layers.Length);

            _network = new ActivationNetwork(new SigmoidFunction(2.0), inputs, layers);
            _teacher = new BackPropagationLearning(_network);
            _teacher.LearningRate = 0.5;
            _teacher.Momentum = 0.1;
        }

        protected override double[] Compute(double[] input)
        {
            return _network.Compute(input);
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            // AForge возвращает ошибку, а не количество итераций, но для совместимости интерфейса...
            return _teacher.Run(sample.input, sample.Output) < acceptableError ? 1 : 0; 
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            // Подготовка массивов для AForge
            int samples = samplesSet.Count;
            double[][] inputs = new double[samples][];
            double[][] outputs = new double[samples][];

            for(int i = 0; i < samples; i++)
            {
                inputs[i] = samplesSet[i].input;
                outputs[i] = samplesSet[i].Output;
            }

            double error = double.MaxValue;
            var sw = System.Diagnostics.Stopwatch.StartNew();

            for (int i = 0; i < epochsCount; i++)
            {
                error = _teacher.RunEpoch(inputs, outputs) / samples;
                OnTrainProgress((double)i / epochsCount, error, sw.Elapsed);
                if (error < acceptableError) break;
            }
            
            sw.Stop();
            OnTrainProgress(1.0, error, sw.Elapsed);
            return error;
        }
    }
}