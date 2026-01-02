using System;
using System.Diagnostics;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        // Структура сети: веса [слой][нейрон][вход]
        // Входы нейрона включают +1 для нейрона смещения (Bias)
        private double[][][] _weights;
        
        // Значения выходов нейронов на текущей итерации [слой][нейрон] для backpropagation
        private double[][] _outputs;

        // Ошибки для каждого нейрона [слой][нейрон]
        private double[][] _deltas;

        private Random _random;

        public StudentNetwork(int[] structure)
        {
            _random = new Random();

            // Инициализация массивов слоев
            int layersCount = structure.Length - 1; // -1, т.к. первый элемент - это размер входного вектора, а не слой
            
            _weights = new double[layersCount][][];
            _outputs = new double[layersCount][];
            _deltas = new double[layersCount][];

            for (int l = 0; l < layersCount; l++)
            {
                int inputSize = structure[l];
                int neuronsCount = structure[l + 1];

                _weights[l] = new double[neuronsCount][];
                _outputs[l] = new double[neuronsCount];
                _deltas[l] = new double[neuronsCount];

                for (int n = 0; n < neuronsCount; n++)
                {
                    // +1 для веса смещения (Bias)
                    _weights[l][n] = new double[inputSize + 1];

                    // Инициализация весов случайными значениями [-0.5; 0.5]
                    for (int w = 0; w < inputSize + 1; w++)
                    {
                        _weights[l][n][w] = _random.NextDouble() - 0.5;
                    }
                }
            }
        }

        // Сигмоида
        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        // Производная сигмоиды
        private double SigmoidDerivative(double y)
        {
            return y * (1.0 - y);
        }

        // Прямой проход. Сохраняет выходы всех слоев
        private double[] RunForward(double[] input)
        {
            double[] currentInput = input;

            for (int l = 0; l < _weights.Length; l++)
            {
                int neuronsCount = _weights[l].Length;
                
                for (int n = 0; n < neuronsCount; n++)
                {
                    double sum = 0;
                    // Сумма взвешенных входов
                    for (int w = 0; w < currentInput.Length; w++)
                    {
                        sum += currentInput[w] * _weights[l][n][w];
                    }
                    // Добавляем смещение (Bias). Последний вход всегда 1.0
                    sum += 1.0 * _weights[l][n][currentInput.Length];

                    _outputs[l][n] = Sigmoid(sum);
                }
                
                // Передаём на следующий слой
                currentInput = _outputs[l];
            }

            return currentInput;
        }

        // Реализация метода предсказания
        protected override double[] Compute(double[] input)
        {
            return RunForward(input);
        }

        // Обучение на одном образце с Backpropagation
        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            // Прямой проход
            RunForward(sample.input);
            
            // Текущий learning rate (можно сделать адаптивным, но лень)
            double learningRate = 0.2; 
            int outputLayerIndex = _weights.Length - 1;

            // Вычисление ошибки выходного слоя
            // delta = (y_t - y_o) * f'(y_o)
            for (int n = 0; n < _deltas[outputLayerIndex].Length; n++)
            {
                double output = _outputs[outputLayerIndex][n];
                double error = sample.Output[n] - output; // Target - Actual
                _deltas[outputLayerIndex][n] = error * SigmoidDerivative(output);
            }

            // Обратное распространение ошибки на скрытые слои
            for (int l = outputLayerIndex - 1; l >= 0; l--)
            {
                for (int n = 0; n < _deltas[l].Length; n++)
                {
                    double sumError = 0;
                    
                    // Собираем ошибки со следующего слоя
                    for (int nextN = 0; nextN < _deltas[l+1].Length; nextN++)
                    {
                        sumError += _deltas[l+1][nextN] * _weights[l+1][nextN][n];
                    }

                    double output = _outputs[l][n];
                    _deltas[l][n] = sumError * SigmoidDerivative(output);
                }
            }

            // Обновление весов (Градиентный спуск)
            // W_new = W_old + LR * delta * input
            
            // Вход первого слоя - вход сети
            double[] inputs = sample.input;

            for (int l = 0; l < _weights.Length; l++)
            {
                for (int n = 0; n < _weights[l].Length; n++)
                {
                    // Обновляем веса связей с предыдущим слоем
                    for (int w = 0; w < inputs.Length; w++)
                    {
                        _weights[l][n][w] += learningRate * _deltas[l][n] * inputs[w];
                    }
                    // Обновляем вес Bias
                    _weights[l][n][inputs.Length] += learningRate * _deltas[l][n] * 1.0;
                }
                
                // Вход следующего слоя - это выход текущего
                inputs = _outputs[l];
            }

            return 1; // Сделали одну итерацию
        }

        // Обучение на датасете
        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            double currentError = double.MaxValue;
            int epoch = 0;

            // Перемешивания выборки
            int[] indexes = new int[samplesSet.Count];
            for (int i = 0; i < samplesSet.Count; i++) indexes[i] = i;

            while (epoch < epochsCount && currentError > acceptableError)
            {
                epoch++;
                
                // Fisher-Yates shuffle
                for (int i = indexes.Length - 1; i > 0; i--)
                {
                    int j = _random.Next(i + 1);
                    // Может я совсем нуб C# 🤔
                    int temp = indexes[i]; 
                    indexes[i] = indexes[j];
                    indexes[j] = temp; 
                }

                double errorSum = 0;

                // Проход по всем образцам
                for (int i = 0; i < samplesSet.Count; i++)
                {
                    Sample sample = samplesSet[indexes[i]];
                    Train(sample, acceptableError, parallel);
                    
                    double[] networkOutput = _outputs[_outputs.Length - 1];

                    sample.ProcessPrediction(networkOutput);

                    // Квадратичная ошибка
                    errorSum += sample.EstimatedError();
                }

                currentError = errorSum / samplesSet.Count; // SE -> MSE

                // Оповещаем форму о прогрессе
                OnTrainProgress((double)epoch / epochsCount, currentError, stopwatch.Elapsed);
            }

            stopwatch.Stop();
            OnTrainProgress(1.0, currentError, stopwatch.Elapsed);

            return currentError;
        }
    }
}