using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Telegram.Bot;
using Telegram.Bot.Exceptions;
using Telegram.Bot.Extensions.Polling;
using Telegram.Bot.Types;
using Telegram.Bot.Types.Enums;

namespace AIMLTGBot
{
    public class TelegramService : IDisposable
    {
        private readonly TelegramBotClient client;
        private readonly AIMLService aiml;
        private readonly INeuralNetwork neuralNetwork; 
        private readonly CancellationTokenSource cts = new CancellationTokenSource();
        public string Username { get; }

        public TelegramService(string token, AIMLService aimlService, INeuralNetwork nn)
        {
            aiml = aimlService;
            neuralNetwork = nn;
            client = new TelegramBotClient(token);
            client.StartReceiving(HandleUpdateMessageAsync, HandleErrorAsync, new ReceiverOptions
            {
                AllowedUpdates = new[] { UpdateType.Message }
            },
            cancellationToken: cts.Token);
            Username = client.GetMeAsync().Result.Username;
        }

        async Task HandleUpdateMessageAsync(ITelegramBotClient botClient, Update update, CancellationToken cancellationToken)
        {
            var message = update.Message;
            if (message == null) return;

            var chatId = message.Chat.Id;
            var username = message.Chat.FirstName ?? "Незнакомец";

            // Обработка текста
            if (message.Type == MessageType.Text)
            {
                var messageText = message.Text;
                Console.WriteLine($"Text message from {chatId} ({username}): {messageText}");

                await botClient.SendTextMessageAsync(
                    chatId: chatId,
                    text: aiml.Talk(chatId, username, messageText),
                    cancellationToken: cancellationToken);
                return;
            }

            // Обработка фото
            if (message.Type == MessageType.Photo)
            {
                Console.WriteLine($"Photo received from {chatId}");
                
                // Скачиваем фото
                var photoId = message.Photo.Last().FileId;
                var fileInfo = await client.GetFileAsync(photoId, cancellationToken: cancellationToken);
                
                using (var imageStream = new MemoryStream())
                {
                    await client.DownloadFileAsync(fileInfo.FilePath, imageStream, cancellationToken: cancellationToken);
                    imageStream.Seek(0, 0);

                    // Распознаем образ (сейчас это заглушка)
                    string recognizedClass = await neuralNetwork.Predict(imageStream);
                    Console.WriteLine($"Neural Network predicted: {recognizedClass}");
                    
                    // Запрос к AIML. Формат: "EVENT IMAGE RECOGNIZED [CLASS]"
                    string aimlRequest = $"EVENT IMAGE RECOGNIZED {recognizedClass}";
                    
                    string botResponse = aiml.Talk(chatId, username, aimlRequest);

                    await client.SendTextMessageAsync(
                        chatId: chatId,
                        text: botResponse, 
                        cancellationToken: cancellationToken);
                }
                return;
            }
        }

        Task HandleErrorAsync(ITelegramBotClient botClient, Exception exception, CancellationToken cancellationToken)
        {
            var apiRequestException = exception as ApiRequestException;
            if (apiRequestException != null)
                Console.WriteLine($"Telegram API Error:\n[{apiRequestException.ErrorCode}]\n{apiRequestException.Message}");
            else
                Console.WriteLine(exception.ToString());
            return Task.CompletedTask;
        }

        public void Dispose()
        {
            cts.Cancel();
        }
    }
}