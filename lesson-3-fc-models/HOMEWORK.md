# Домашнее задание к уроку 3: Полносвязные сети

## Цель задания
Изучить влияние архитектуры полносвязных сетей на качество классификации, провести эксперименты с различными конфигурациями моделей.

## Задание 1: Эксперименты с глубиной сети (30 баллов)

Создайте файл `homework_depth_experiments.py`:

### 1.1 Сравнение моделей разной глубины (15 баллов)
```python
# Создайте и обучите модели с различным количеством слоев:
# - 1 слой (линейный классификатор)
# - 2 слоя (1 скрытый)
# - 3 слоя (2 скрытых)
# - 5 слоев (4 скрытых)
# - 7 слоев (6 скрытых)
# 
# Для каждого варианта:
# - Сравните точность на train и test
# - Визуализируйте кривые обучения
# - Проанализируйте время обучения
```

### 1.2 Анализ переобучения (15 баллов)
```python
# Исследуйте влияние глубины на переобучение:
# - Постройте графики train/test accuracy по эпохам
# - Определите оптимальную глубину для каждого датасета
# - Добавьте Dropout и BatchNorm, сравните результаты
# - Проанализируйте, когда начинается переобучение
```

## Задание 2: Эксперименты с шириной сети (25 баллов)

Создайте файл `homework_width_experiments.py`:

### 2.1 Сравнение моделей разной ширины (15 баллов)
```python
# Создайте модели с различной шириной слоев:
# - Узкие слои: [64, 32, 16]
# - Средние слои: [256, 128, 64]
# - Широкие слои: [1024, 512, 256]
# - Очень широкие слои: [2048, 1024, 512]
# 
# Для каждого варианта:
# - Поддерживайте одинаковую глубину (3 слоя)
# - Сравните точность и время обучения
# - Проанализируйте количество параметров
```

### 2.2 Оптимизация архитектуры (10 баллов)
```python
# Найдите оптимальную архитектуру:
# - Используйте grid search для поиска лучшей комбинации
# - Попробуйте различные схемы изменения ширины (расширение, сужение, постоянная)
# - Визуализируйте результаты в виде heatmap
```

## Задание 3: Эксперименты с регуляризацией (25 баллов)

Создайте файл `homework_regularization_experiments.py`:

### 3.1 Сравнение техник регуляризации (15 баллов)
```python
# Исследуйте различные техники регуляризации:
# - Без регуляризации
# - Только Dropout (разные коэффициенты: 0.1, 0.3, 0.5)
# - Только BatchNorm
# - Dropout + BatchNorm
# - L2 регуляризация (weight decay)
# 
# Для каждого варианта:
# - Используйте одинаковую архитектуру
# - Сравните финальную точность
# - Проанализируйте стабильность обучения
# - Визуализируйте распределение весов
```

### 3.2 Адаптивная регуляризация (10 баллов)
```python
# Реализуйте адаптивные техники:
# - Dropout с изменяющимся коэффициентом
# - BatchNorm с различными momentum
# - Комбинирование нескольких техник
# - Анализ влияния на разные слои сети
```

## Дополнительные требования

1. **Код должен быть модульным** - создайте отдельные функции для каждого эксперимента
2. **Визуализация** - создайте информативные графики и диаграммы
3. **Документация** - добавьте подробные комментарии и анализ результатов
4. **Тестирование** - проверьте корректность экспериментов на простых примерах
5. **Логирование** - используйте logging для отслеживания экспериментов

## Структура проекта

```
homework/
├── homework_depth_experiments.py
├── homework_width_experiments.py
├── homework_regularization_experiments.py
├── utils/
│   ├── experiment_utils.py
│   ├── visualization_utils.py
│   └── model_utils.py
├── results/
│   ├── depth_experiments/
│   ├── width_experiments/
│   └── regularization_experiments/
├── plots/                   # Графики и визуализации
└── README.md               # Описание результатов
```

## Срок сдачи
Домашнее задание должно быть выполнено до начала занятия 5.

## Полезные ссылки
- [PyTorch nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
- [PyTorch BatchNorm](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
- [Understanding Dropout](https://jmlr.org/papers/v15/srivastava14a.html)

Удачи в выполнении задания! 🚀 