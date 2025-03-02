import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class DataLoader:
    '''
    Класс для загрузки и подготовки данных Iris.
    '''

    def __init__(self, test_size=0.2, random_state=42):
        '''
        Инициализация DataLoader.

        Args:
            test_size (float): Размер тестовой выборки.
            random_state (int): random_state для воспроизводимости.
        '''
        self.test_size = test_size
        self.random_state = random_state
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

    def get_data(self):
        '''
        Возвращает данные для обучения и тестирования.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        '''
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_feature_names(self):
        '''
        Возвращает названия признаков.

        Returns:
            list: Названия признаков.
        '''
        return self.iris.feature_names

    def get_target_names(self):
        '''
        Возвращает названия классов.

        Returns:
            list: Названия классов.
        '''
        return self.iris.target_names
