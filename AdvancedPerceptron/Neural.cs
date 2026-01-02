using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace NeuralNetwork1
{
    // Типы смайликов согласно заданию (10 классов)
    public enum FigureType : byte
    {
        Grin = 0,       // 😁
        Cloud = 1,      // ☁️
        Flushed = 2,    // 😳
        Heart = 3,      // ❤️
        Joy = 4,        // 😂
        ThumbsUp = 5,   // 👍
        Pout = 6,       // 😡
        Exploding = 7,  // 🤯
        Sunglasses = 8, // 😎
        Nerd = 9,       // 🤓
        Undef = 255
    }

    public abstract class BaseNetwork
    {
        // Обучение на одном образце
        public abstract int Train(Sample sample, double acceptableError, bool parallel);
        
        // Обучение на наборе данных
        public abstract double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel);
        
        // Предсказание (возвращает выходной вектор)
        protected abstract double[] Compute(double[] input);

        public event Action<double, double, TimeSpan> TrainProgress;

        protected void OnTrainProgress(double progress, double error, TimeSpan duration)
        {
            TrainProgress?.Invoke(progress, error, duration);
        }

        // Высокоуровневый метод классификации
        public FigureType Predict(Sample sample)
        {
            double[] output = Compute(sample.input);
            return sample.ProcessPrediction(output);
        }
    }

    public class Sample
    {
        public double[] input;
        public double[] Output; // Целевой вектор (для обучения)
        public double[] error;
        public FigureType actualClass;
        public FigureType recognizedClass;

        public Sample(double[] inputValues, int classesCount, FigureType sampleClass = FigureType.Undef)
        {
            input = (double[])inputValues.Clone();
            Output = new double[classesCount];
            if (sampleClass != FigureType.Undef && (int)sampleClass < classesCount)
                Output[(int)sampleClass] = 1.0;

            actualClass = sampleClass;
            recognizedClass = FigureType.Undef;
        }

        public FigureType ProcessPrediction(double[] neuralOutput)
        {
            if (error == null) error = new double[neuralOutput.Length];
            
            int maxIndex = 0;
            for (int i = 0; i < neuralOutput.Length; ++i)
            {
                if (Output != null) error[i] = neuralOutput[i] - Output[i];
                if (neuralOutput[i] > neuralOutput[maxIndex]) maxIndex = i;
            }
            recognizedClass = (FigureType)maxIndex;
            return recognizedClass;
        }

        public double EstimatedError()
        {
            double res = 0;
            if (error == null) return 0;
            foreach (var e in error) res += e * e;
            return res;
        }
    }

    public class SamplesSet
    {
        public List<Sample> samples = new List<Sample>();
        public int Count => samples.Count;
        public Sample this[int i] => samples[i];
        
        public void AddSample(Sample sample) => samples.Add(sample);
    }
}