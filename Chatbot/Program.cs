using System;

namespace AIMLTGBot
{
    class Program
    {
        static void Main(string[] args)
        {
            if (!System.IO.File.Exists("TGToken.txt"))
            {
                System.IO.File.WriteAllText("TGToken.txt", "FILL_ME");
            }

            var token = System.IO.File.ReadAllText("TGToken.txt");
            if (token == "FILL_ME" || string.IsNullOrWhiteSpace(token))
            {
                Console.WriteLine("Пожалуйста, заполните файл TGToken.txt корректным токеном.");
                Console.ReadLine();
                return;
            }

            INeuralNetwork neuralNet = new MockNeuralNetwork();
            
            using (var tg = new TelegramService(token, new AIMLService(), neuralNet))
            {
                Console.WriteLine($"Подключились к Telegram как {tg.Username}.");
                Console.WriteLine("Нажмите Enter для выхода...");
                Console.ReadLine();
            }
        }
    }
}