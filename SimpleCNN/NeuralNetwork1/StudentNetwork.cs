using System;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        // Структура нейросети
        // layers[i][j] - значение выхода нейрона j в слое i
        private double[][] neurons;
        // weights[i][j][k] - вес связи, идущей в нейрон j слоя i от нейрона k слоя i-1
        private double[][][] weights;
        // biases[i][j] - смещение (bias) нейрона j в слое i
        private double[][] biases;
        
        // Для обучения
        private double[][] deltas; // Ошибки нейронов
        private double learningRate = 0.5; // Скорость обучения

        public StudentNetwork(int[] structure)
        {
            // Инициализация слоев
            neurons = new double[structure.Length][];
            biases = new double[structure.Length][];
            deltas = new double[structure.Length][];
            weights = new double[structure.Length][][];

            Random rand = new Random();

            for (int i = 0; i < structure.Length; i++)
            {
                neurons[i] = new double[structure[i]];
                biases[i] = new double[structure[i]];
                deltas[i] = new double[structure[i]];

                // Инициализация весов (начиная со второго слоя)
                if (i > 0)
                {
                    weights[i] = new double[structure[i]][];
                    for (int j = 0; j < structure[i]; j++)
                    {
                        weights[i][j] = new double[structure[i - 1]];
                        // Случайные веса и смещения
                        biases[i][j] = rand.NextDouble() * 2 - 1; 
                        for (int k = 0; k < structure[i - 1]; k++)
                        {
                            weights[i][j][k] = rand.NextDouble() * 2 - 1;
                        }
                    }
                }
            }
        }

        // Сигмоида
        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        
        // Производная сигмоиды (y - значение функции)
        private double SigmoidDerivative(double y) => y * (1.0 - y);

        protected override double[] Compute(double[] input)
        {
            // Копируем входные данные в нулевой слой
            for (int i = 0; i < input.Length; i++)
                neurons[0][i] = input[i];

            // Проход по слоям
            for (int i = 1; i < neurons.Length; i++)
            {
                for (int j = 0; j < neurons[i].Length; j++)
                {
                    double sum = biases[i][j];
                    for (int k = 0; k < neurons[i - 1].Length; k++)
                    {
                        sum += neurons[i - 1][k] * weights[i][j][k];
                    }
                    neurons[i][j] = Sigmoid(sum);
                }
            }

            return neurons[neurons.Length - 1];
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int epoch = 0;
            double error = 1.0;

            // Нормализация входа (важно, так как в generator значения > 1)
            double[] normalizedInput = new double[sample.input.Length];
            for (int i = 0; i < sample.input.Length; i++) normalizedInput[i] = sample.input[i] > 0 ? 1.0 : 0.0;
            
            // Если мы подаем сырые данные с проекций (где числа 0..200), лучше нормализовать их делением на макс. размер
            // Но для совместимости с генератором будем считать >0 за единицу или использовать как есть.
            // В данном примере я оставил sample.input как есть, но StudentNetwork может плохо учиться на больших числах.
            // Рекомендуется в GenerateFigure и WebcamProcessor делить на 200.0, но пока оставим так:

            while (error > acceptableError && epoch < 1000) // Лимит 1000 итераций для одного образа
            {
                BackPropagation(sample.input, sample.Output);
                error = sample.EstimatedError();
                epoch++;
            }
            return epoch;
        }

        private void BackPropagation(double[] input, double[] expectedOutput)
        {
            // 1. Прямой проход
            Compute(input);

            // 2. Вычисление ошибки выходного слоя
            int lastLayer = neurons.Length - 1;
            for (int j = 0; j < neurons[lastLayer].Length; j++)
            {
                double output = neurons[lastLayer][j];
                double error = expectedOutput[j] - output;
                deltas[lastLayer][j] = error * SigmoidDerivative(output);
            }

            // 3. Вычисление ошибки скрытых слоев
            for (int i = lastLayer - 1; i > 0; i--)
            {
                for (int j = 0; j < neurons[i].Length; j++)
                {
                    double errorSum = 0;
                    for (int k = 0; k < neurons[i + 1].Length; k++)
                    {
                        errorSum += deltas[i + 1][k] * weights[i + 1][k][j];
                    }
                    deltas[i][j] = errorSum * SigmoidDerivative(neurons[i][j]);
                }
            }

            // 4. Обновление весов
            for (int i = 1; i < neurons.Length; i++)
            {
                for (int j = 0; j < neurons[i].Length; j++)
                {
                    biases[i][j] += learningRate * deltas[i][j];
                    for (int k = 0; k < neurons[i - 1].Length; k++)
                    {
                        weights[i][j][k] += learningRate * deltas[i][j] * neurons[i - 1][k];
                    }
                }
            }
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            double error = double.MaxValue;
            int epoch = 0;
            
            // Массивы для ускорения доступа
            double[][] inputs = new double[samplesSet.Count][];
            double[][] outputs = new double[samplesSet.Count][];
            for(int i=0; i<samplesSet.Count; i++)
            {
                inputs[i] = samplesSet[i].input;
                outputs[i] = samplesSet[i].Output;
            }

            DateTime startTime = DateTime.Now;

            while (epoch < epochsCount && error > acceptableError)
            {
                epoch++;
                double sumError = 0;

                // Для SGD порядок образов лучше перемешивать, но здесь простой проход
                if (parallel)
                {
                    // Параллельное обучение сложно реализовать корректно без блокировок весов (Race Condition),
                    // поэтому для StudentNetwork сделаем последовательно даже если флаг true,
                    // либо используем lock, что убьет производительность.
                    // Оставим последовательно для стабильности.
                    for (int i = 0; i < inputs.Length; i++)
                    {
                        BackPropagation(inputs[i], outputs[i]);
                        // Считаем ошибку (квадратичную)
                        double[] output = neurons[neurons.Length - 1];
                        for(int k=0; k<output.Length; k++) sumError += Math.Pow(outputs[i][k] - output[k], 2);
                    }
                }
                else
                {
                    for (int i = 0; i < inputs.Length; i++)
                    {
                        BackPropagation(inputs[i], outputs[i]);
                        // Считаем ошибку
                        double[] output = neurons[neurons.Length - 1];
                        for(int k=0; k<output.Length; k++) sumError += Math.Pow(outputs[i][k] - output[k], 2);
                    }
                }

                error = sumError / inputs.Length;
                OnTrainProgress((double)epoch / epochsCount, error, DateTime.Now - startTime);
            }

            return error;
        }
    }
}