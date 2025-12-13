using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace NeuralNetwork1
{
    public static class ImageProcessor
    {
        // Метод для конвертации изображения с веб-камеры в Sample для нейросети
        public static double[] ProcessImage(Bitmap original, int networkInputSize = 400)
        {
            // 1. Бинаризация и кроп (отсечение лишнего фона)
            Bitmap processed = TransformImage(original, 200, 200);
            
            // 2. Создание вектора признаков (как в GenerateFigure: проекции X и Y)
            // Важно: GenerateFigure создает массив 400 элементов (200 по X + 200 по Y)
            double[] input = new double[400];
            
            for (int y = 0; y < 200; y++)
            {
                for (int x = 0; x < 200; x++)
                {
                    Color pixel = processed.GetPixel(x, y);
                    // Если пиксель черный (рисунок)
                    if (pixel.R == 0) 
                    {
                        input[x] += 1;          // Проекция на X
                        input[200 + y] += 1;    // Проекция на Y
                    }
                }
            }

            // Опционально: нормализация данных, чтобы нейросети было легче учиться (0..1)
            // Но AccordNet уже настроен на входные данные генератора, где значения могут быть большими.
            // Для StudentNetwork лучше раскомментировать строку ниже:
            // for(int i=0; i<400; i++) input[i] /= 200.0;

            return input;
        }

        private static Bitmap TransformImage(Bitmap source, int width, int height)
        {
            // Приводим к Grayscale и ищем границы объекта
            int minX = source.Width, maxX = 0, minY = source.Height, maxY = 0;
            bool found = false;

            // Блокировка битов для скорости (или GetPixel для простоты, если камера не HD)
            // Используем простой подход для читаемости
            Bitmap bw = new Bitmap(source.Width, source.Height);
            
            for (int y = 0; y < source.Height; y++)
            {
                for (int x = 0; x < source.Width; x++)
                {
                    Color c = source.GetPixel(x, y);
                    // Простой порог яркости
                    if (c.GetBrightness() < 0.5f) 
                    {
                        bw.SetPixel(x, y, Color.Black);
                        if (x < minX) minX = x;
                        if (x > maxX) maxX = x;
                        if (y < minY) minY = y;
                        if (y > maxY) maxY = y;
                        found = true;
                    }
                    else
                    {
                        bw.SetPixel(x, y, Color.White);
                    }
                }
            }

            if (!found) return new Bitmap(width, height); // Пустой лист

            // Вырезаем контент (Crop)
            int w = maxX - minX + 1;
            int h = maxY - minY + 1;
            Rectangle cropRect = new Rectangle(minX, minY, w, h);
            Bitmap cropped = bw.Clone(cropRect, bw.PixelFormat);

            // Масштабируем в 200x200
            Bitmap result = new Bitmap(width, height);
            using (Graphics g = Graphics.FromImage(result))
            {
                g.Clear(Color.White);
                // Сохраняем пропорции или растягиваем?
                // Для простоты растянем, как делает генератор фигур
                g.DrawImage(cropped, 0, 0, width, height);
            }
            
            return result;
        }
    }
}