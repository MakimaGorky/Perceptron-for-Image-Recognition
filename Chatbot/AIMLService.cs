using AIMLbot;
using System.Collections.Generic;
using System.IO;

namespace AIMLTGBot
{
    public class AIMLService
    {
        readonly Bot bot;
        readonly Dictionary<long, User> users = new Dictionary<long, User>();

        public AIMLService()
        {
            bot = new Bot();
            
            // Предполагается, что папки aiml и config лежат рядом с exe
            bot.loadSettings(); 
            
            bot.isAcceptingUserInput = false;
            // Загружаем загрузчик
            bot.loadAIMLFromFiles(); 
            bot.isAcceptingUserInput = true;
        }

        public string Talk(long userId, string userName, string phrase)
        {
            var result = "";
            User user;
            if (!users.ContainsKey(userId))
            {
                user = new User(userId.ToString(), bot);
                users.Add(userId, user);
                // Отправляем команду для установки имени
                Request r = new Request($"Меня зовут {userName}", user, bot);
                result += bot.Chat(r).Output + System.Environment.NewLine;
            }
            else
            {
                user = users[userId];
            }

            Request request = new Request(phrase, user, bot);
            Result res = bot.Chat(request);
            result += res.Output;
            return result;
        }
    }
}