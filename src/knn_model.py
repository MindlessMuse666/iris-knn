from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNNModel:
    '''
    Класс для обучения и предсказания с использованием KNN.
    '''

    def __init__(self, n_neighbors=5, algorithm='auto'):
        '''
        Инициализация KNNModel.

        Args:
            n_neighbors (int): Количество соседей.
            algorithm (str): Алгоритм поиска ближайших соседей.
        '''
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, algorithm=self.algorithm)

    def fit(self, X_train, y_train):
        '''
        Обучение модели KNN.

        Args:
            X_train (array-like): Обучающие признаки.
            y_train (array-like): Обучающие метки.
        '''
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        '''
        Предсказание классов для тестовых данных.

        Args:
            X_test (array-like): Тестовые признаки.

        Returns:
            array-like: Предсказанные классы.
        '''
        return self.model.predict(X_test)

    def kneighbors(self, X, n_neighbors=None):
        '''
        Находит K ближайших соседей для каждой точки в X.

        Args:
            X (array-like): Данные для поиска соседей.
            n_neighbors (int, optional): Количество соседей для возврата. По умолчанию используется self.n_neighbors.

        Returns:
            tuple: (distances, indices) - расстояния и индексы K ближайших соседей.
        '''
        return self.model.kneighbors(X, n_neighbors=n_neighbors)
