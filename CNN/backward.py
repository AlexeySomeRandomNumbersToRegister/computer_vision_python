import numpy as np

from CNN.utils import *
        
def convolutionBackward(dconv_prev, conv_in, filt, s):

    (n_f, n_c, f, _) = filt.shape
    (_, orig_dim, _) = conv_in.shape
    # Инициализация переменных
    dout = np.zeros(conv_in.shape) # Пустой массив для хранения градиента ошибки по входу свёрточного слоя
    dfilt = np.zeros(filt.shape) # Пустой массив для хранения градиента ошибки по весам фильтра
    dbias = np.zeros((n_f,1)) # Пустой массив для хранения градиента ошибки по смещениям
    for curr_f in range(n_f):
        # Проход по всем фильтрам
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                #  Вычисляем градиент ошибки по весам фильтра путём перемножения ошибки на выходе сверточного слоя и входных данных, через которые прошла свёртка
                dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y+f, curr_x:curr_x+f]
                # Вычисляем градиент ошибки по входу свёрточного слоя, перемножая ошибку на выходе слоя свертки с соответствующими весами фильтра
                dout[:, curr_y:curr_y+f, curr_x:curr_x+f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f] 
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        # Вычисляем градиент ошибки по смещению путём суммирования ошибки по всем элементам выхода сверточного слоя для каждого фильтра
        dbias[curr_f] = np.sum(dconv_prev[curr_f])
    
    return dout, dfilt, dbias



def maxpoolBackward(dpool, orig, f, s):

    (n_c, orig_dim, _) = orig.shape
    
    dout = np.zeros(orig.shape) # Инициализируем массив для градиентов
    
    for curr_c in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # Получаем индекс наибольшего значения во входных данных для текущего окна
                (a, b) = nanargmax(orig[curr_c, curr_y:curr_y+f, curr_x:curr_x+f])
                # Присваиваем градиентам пулинга значение из соответствующего градиента пулинга
                dout[curr_c, curr_y+a, curr_x+b] = dpool[curr_c, out_y, out_x]
                
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        
    return dout