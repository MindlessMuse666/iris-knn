from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Metrics:
    '''
    Класс для расчета метрик оценки модели.
    '''

    def calculate_metrics(self, y_true, y_pred, average='weighted'):
        '''
        Рассчитывает метрики accuracy, precision, recall, F1-score.

        Args:
            y_true (array-like): Истинные метки.
            y_pred (array-like): Предсказанные метки.
            average (str): Тип усреднения для precision, recall и F1-score.

        Returns:
            dict: Словарь с метриками.
        '''
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)
        f1 = f1_score(y_true, y_pred, average=average)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }