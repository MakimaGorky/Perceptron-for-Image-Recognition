using System;
using System.Diagnostics;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        // Структура сети: веса [слой][нейрон][вход]
        // Входы нейрона включают +1 для нейрона смещения (Bias)
        private double[][][] _weights;
        
        // Значения выходов нейронов на текущей итерации [слой][нейрон]
        // Нужны для Backpropagation
        private double[][] _outputs;

        // "Дельты" (ошибки) для каждого нейрона [слой][нейрон]
        private double[][] _deltas;

        private Random _random;

        public StudentNetwork(int[] structure)
        {
            _random = new Random();

            // Инициализация массивов слоев
            // structure.Length - 1, т.к. первый элемент - это размер входного вектора, а не слой нейронов
            int layersCount = structure.Length - 1;
            
            _weights = new double[layersCount][][];
            _outputs = new double[layersCount][];
            _deltas = new double[layersCount][];

            for (int l = 0; l < layersCount; l++)
            {
                int inputSize = structure[l];      // Количество входов для текущего слоя
                int neuronsCount = structure[l + 1]; // Количество нейронов в текущем слое

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

        // Сигмоидальная функция активации
        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        // Производная сигмоиды (через само значение функции y = Sigmoid(x))
        private double SigmoidDerivative(double y)
        {
            return y * (1.0 - y);
        }

        // Прямой проход (Feed Forward)
        // Сохраняет выходы всех слоев для дальнейшего обучения
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
                    // Добавляем смещение (Bias). Представим, что последний вход всегда 1.0
                    sum += 1.0 * _weights[l][n][currentInput.Length]; // Последний вес - для Bias

                    _outputs[l][n] = Sigmoid(sum);
                }
                
                // Выход текущего слоя становится входом для следующего
                currentInput = _outputs[l];
            }

            return currentInput;
        }

        // Реализация метода базового класса для предсказания
        protected override double[] Compute(double[] input)
        {
            return RunForward(input);
        }

        // Обучение на одном образце (Backpropagation)
        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            // 1. Прямой проход
            RunForward(sample.input);
            
            // Текущий learning rate (можно сделать адаптивным, но для простоты константа)
            // Чем меньше alpha, тем стабильнее, но медленнее обучение
            double learningRate = 0.2; 
            int outputLayerIndex = _weights.Length - 1;

            // 2. Вычисление ошибки выходного слоя
            // delta_out = (Target - Output) * f'(Output)
            for (int n = 0; n < _deltas[outputLayerIndex].Length; n++)
            {
                double output = _outputs[outputLayerIndex][n];
                double error = sample.Output[n] - output; // Target - Actual
                _deltas[outputLayerIndex][n] = error * SigmoidDerivative(output);
            }

            // 3. Обратное распространение ошибки на скрытые слои
            for (int l = outputLayerIndex - 1; l >= 0; l--)
            {
                for (int n = 0; n < _deltas[l].Length; n++)
                {
                    double sumError = 0;
                    // Собираем ошибки со следующего слоя
                    // Веса следующего слоя [l+1][next_neuron][current_neuron_index]
                    for (int nextN = 0; nextN < _deltas[l+1].Length; nextN++)
                    {
                        sumError += _deltas[l+1][nextN] * _weights[l+1][nextN][n];
                    }

                    double output = _outputs[l][n];
                    _deltas[l][n] = sumError * SigmoidDerivative(output);
                }
            }

            // 4. Обновление весов (Градиентный спуск)
            // Weight_new = Weight_old + LR * delta * input
            
            // Входом для первого слоя является sample.input
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
                    // Обновляем вес Bias (вход для него всегда 1.0)
                    _weights[l][n][inputs.Length] += learningRate * _deltas[l][n] * 1.0;
                }
                
                // Вход для следующего слоя - это выход текущего (перед следующей итерацией цикла l)
                inputs = _outputs[l];
            }

            return 1; // Возвращаем 1, так как сделали одну итерацию
        }

        // Обучение на датасете
        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            double currentError = double.MaxValue;
            int epoch = 0;

            // Массив индексов для перемешивания выборки (SGD работает лучше, если данные идут случайно)
            int[] indexes = new int[samplesSet.Count];
            for (int i = 0; i < samplesSet.Count; i++) indexes[i] = i;

            while (epoch < epochsCount && currentError > acceptableError)
            {
                epoch++;
                
                // Перемешивание индексов (Fisher-Yates shuffle)
                for (int i = indexes.Length - 1; i > 0; i--)
                {
                    int j = _random.Next(i + 1);
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

                    // Считаем квадратичную ошибку
                    // Sample.EstimatedError() считает сумму квадратов ошибок выходов
                    errorSum += sample.EstimatedError();
                }

                currentError = errorSum / samplesSet.Count; // Средняя квадратичная ошибка (MSE)

                // Оповещаем форму о прогрессе (например, каждые 5% эпох или каждую эпоху, если их мало)
                // Чтобы не тормозить GUI, можно обновлять реже
                OnTrainProgress((double)epoch / epochsCount, currentError, stopwatch.Elapsed);
            }

            stopwatch.Stop();
            OnTrainProgress(1.0, currentError, stopwatch.Elapsed);

            return currentError;
        }
    }
}