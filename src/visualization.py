import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import numpy as np


class Visualization:
    '''
    Класс для визуализации результатов.
    '''

    def __init__(self, feature_names, target_names):
        '''
        Инициализация Visualization.

        Args:
            feature_names (list): Названия признаков.
            target_names (list): Названия классов.
        '''
        self.feature_names = feature_names
        self.target_names = target_names
        sns.set_style('whitegrid')


    def plot_pairplot(self, X, y, title='Pairplot of Iris Dataset', filename=r'report\graphics\iris_dataset_pairplot.png'):
        '''
        Строит pairplot (матрицу диаграмм рассеяния) для визуализации взаимосвязей между признаками.

        Args:
            X (array-like): Признаки.
            y (array-like): Метки классов.
            title (str): Заголовок графика.
        '''
        df = pd.DataFrame(X, columns=self.feature_names)
        df['target'] = [self.target_names[i] for i in y]
        fig = px.scatter_matrix(df, dimensions=self.feature_names, color='target', title=title)
        fig.update_layout(title_x=0.5)
        fig.write_image(filename)
        fig.show()
        
        print(f'\n\033[33mМатрица диаграмм рассеяния взаимосвязей между признаками\033[0m сохранёна в файл \033[33m{filename}\033[0m')


    def plot_predictions(self, X_test, y_test, y_pred, title='Predictions vs Actual', filename=r'report\graphics\predictions_vs_actual.png'):
        '''
        Визуализирует предсказанные и истинные значения. Использует первые два признака для отображения.

        Args:
            X_test (array-like): Тестовые признаки.
            y_test (array-like): Истинные метки.
            y_pred (array-like): Предсказанные метки.
            title (str): Заголовок графика.
        '''
        plt.figure(num=title, figsize=(10, 8))
        plt.title(title, fontsize=16, pad=20)

        #  Визуализация для первых двух признаков
        for i, target_name in enumerate(self.target_names):
            plt.scatter(
                X_test[y_test == i, 0],
                X_test[y_test == i, 1],
                label=f'Actual {target_name}',
                marker='o',
                s=50
            )
            plt.scatter(
                X_test[y_pred == i, 0],
                X_test[y_pred == i, 1],
                label=f'Predicted {target_name}',
                marker='x',
                s=50
            )

        plt.xlabel(self.feature_names[0], fontsize=12)
        plt.ylabel(self.feature_names[1], fontsize=12)
        plt.legend()
        plt.savefig(filename)
        plt.show()
        
        print(f'\033[33mГрафик предсказанных и истинных значений\033[0m сохранён в файл \033[33m{filename}\033[0m')


    def plot_decision_boundaries(self, X, y, model, X_train, title='Decision Boundaries', filename=r'report\graphics\decision_boundaries.png'):
        '''
        Визуализирует границы решений классификатора KNN.
        Args:
            X (array-like): Признаки (только 2).
            y (array-like): Метки классов.
            model: Обученная модель KNN.
            X_train (array-like): Обучающие данные (полный набор признаков).
            title (str): Заголовок графика.
        '''
        # Определяем границы графика
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))

        # Создаем фиктивные данные для остальных признаков, используя средние значения из обучающего набора
        dummy_data = np.zeros((xx.size, X_train.shape[1] - 2))
        for i in range(X_train.shape[1] - 2):
            dummy_data[:, i] = np.mean(X_train[:, i+2])  # Используем среднее значение соответствующего признака

        # Объединяем координаты сетки с фиктивными данными
        grid_points = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1), dummy_data], axis=1)

        # Предсказываем класс для каждой точки сетки
        Z = model.predict(grid_points)
        Z = Z.reshape(xx.shape)

        # Строим контурный график
        plt.figure(num=title, figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.viridis)

        # Отображаем точки данных
        for i, target_name in enumerate(self.target_names):
            plt.scatter(X[y == i, 0], X[y == i, 1], label=target_name, s=50, edgecolors='k')

        plt.xlabel(self.feature_names[0], fontsize=12)
        plt.ylabel(self.feature_names[1], fontsize=12)
        plt.title(title, fontsize=16, pad=20)
        plt.legend()
        plt.savefig(filename)
        plt.show()
        
        print(f'\033[33mГрафик границ решений классификатора KNN\033[0m сохранён в файл \033[33m{filename}\033[0m')