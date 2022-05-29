# Permutation Importance class

Класс алгоритма для поиска информативных признаков.
Идея проста - обучаем модель, фиксируем полученное качество по скользящему контролю.
Затем перемешиваем первый признак, обучаем модель, оцениваем качество по скользящему контролю и сравниваем с оригинальным значением. Если качество снизилось значит первый признак важен. 
И так делаем для всех признков.

Данная реализация использует самописный KNN со встроенным методом скользящего контроля.

Starting evaluate feature importance...

Feature importance by Permutation:
feature 2 its importance 0.18
feature 3 its importance 0.09
feature 4 its importance 0.02
feature 6 its importance 0.03
feature 7 its importance 0.00
feature 8 its importance 0.02
feature 9 its importance 0.02