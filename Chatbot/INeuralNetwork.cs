using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace AIMLTGBot
{
    public interface INeuralNetwork
    {
        Task<string> Predict(Stream imageStream);
    }

    // Временная реализация
    public class MockNeuralNetwork : INeuralNetwork
    {
        private readonly Random _rnd = new Random();
        private readonly string[] _classes = new[] { "HEART", "SMILE", "MIND BLOWN" };

        public Task<string> Predict(Stream imageStream)
        {
            // Имитация бурной деятельности
            Thread.Sleep(500); 
            
            var result = _classes[_rnd.Next(_classes.Length)];
            return Task.FromResult(result);
        }
    }
}