from CNN.network import *
from CNN.utils import *

from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser(description='Обучение нейронной сети.')
parser.add_argument('save_path', metavar = 'Save Path', help='Укажите имя файла, в который хотите сохранить параметры.')

if __name__ == '__main__':
    
    args = parser.parse_args()
    save_path = args.save_path
    
    cost = train(save_path = save_path)

    params, cost = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params
    
    # График потерь
    plt.plot(cost, 'r')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend('Loss', loc='upper right')
    plt.show()

    # Получаем тестовые данные
    m = 10000
    X = extract_data('test_images.gz', m, 28)
    y_dash = extract_labels('test_labels.gz', m).reshape(m,1)
    # Нормализация данных
    X-= int(np.mean(X)) # Вычитаем среднее значение
    X/= int(np.std(X)) # Делим на стандартное отклонение
    test_data = np.hstack((X,y_dash))
    
    X = test_data[:,0:-1]
    X = X.reshape(len(test_data), 1, 28, 28)
    y = test_data[:,-1]

    corr = 0
    digit_count = [0 for i in range(10)]
    digit_correct = [0 for i in range(10)]
   
    print()
    print("Расчет accuracy на тестовых данных:")

    t = tqdm(range(len(X)), leave=True)

    for i in t:
        x = X[i]
        pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)

        digit_count[int(y[i])]+=1
        if pred==y[i]:
            corr+=1
            digit_correct[pred]+=1

        t.set_description("Acc:%0.2f%%" % (float(corr/(i+1))*100))
        
    print("Общая Accuracy: %.2f" % (float(corr/len(test_data)*100)))
    x = np.arange(10)
    digit_recall = [x/y for x,y in zip(digit_correct, digit_count)]
    plt.xlabel('Классы')
    plt.ylabel('Правильно распознанные')
    plt.title("Тестирование")
    plt.bar(x,digit_recall)
    plt.show()

    random_indices = random.sample(range(len(X)), 10)

    # Создаем фигуру для отображения выбранных случайных изображений
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # Отображаем рандомные изображения и рамки
    for i, ax in enumerate(axes.flat):
        index = random_indices[i]
        x = X[index]
        pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
        
        # Выводим изображение с bbox
        draw_bbox(ax, x.reshape(28, 28), pred, int(y[index]))

    plt.tight_layout()
    plt.show()