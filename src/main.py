from data_loader import DataLoader
from knn_model import KNNModel
from metrics import Metrics
from visualization import Visualization
from sklearn.model_selection import cross_val_score


def main():
    '''
    Основная функция для запуска проекта.
    '''

    # 1. Загрузка данных
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.get_data()
    X = data_loader.iris.data  # Все данные для кросс-валидации
    y = data_loader.iris.target
    feature_names = data_loader.get_feature_names()
    target_names = data_loader.get_target_names()

    # 2. Обучение модели
    knn_model = KNNModel(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # Кросс-валидация
    cv_scores = cross_val_score(knn_model.model, X, y, cv=10, scoring='accuracy')  # 10-кратная кросс-валидация
    print(f'Кросс-валидация (accuracy): \033[32m{cv_scores.mean()}\033[0m (+/- \033[31m{cv_scores.std()}\033[0m)\n')

    # 3. Предсказание классов
    y_pred = knn_model.predict(X_test)

    # 4. Оценка производительности
    metrics = Metrics()
    evaluation = metrics.calculate_metrics(y_test, y_pred)

    print('Метрики оценки:')
    for metric, value in evaluation.items():
        print(f'{metric}: {value}')

    # 5. Визуализация результатов
    visualization = Visualization(feature_names, target_names)
    visualization.plot_pairplot(X_train, y_train, title='Взаимосвязи между признаками Iris')
    visualization.plot_predictions(X_test, y_test, y_pred, title='Предсказанные и истинные значения')
    visualization.plot_decision_boundaries(X_train[:, :2], y_train, knn_model.model, X_train, title='Границы решений KNN (2 признака)')


if __name__ == '__main__':
    main()