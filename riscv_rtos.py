import heapq  # Модуль для реализации очереди с приоритетом
import time  # Модуль для работы со временем
import random  # Модуль для генерации случайных чисел
import logging  # Модуль для логирования событий
import threading  # Модуль для работы с потоками
import pandas as pd # Модуль для работы с данными в виде таблиц
import matplotlib   # Модуль для построения графиков
matplotlib.use('Agg')    # Использование бэкэнда 'Agg' для работы без отображения окон
import matplotlib.pyplot as plt
import seaborn as sns    # Модуль для улучшенного визуального представления данных
from dataclasses import dataclass, field  # Модули для создания классов данных
from typing import Optional  # Модуль для аннотаций типов, добавляет возможность присвоить NULL объекту или полю
from enum import Enum, auto  # Модули для создания перечислений
from tabulate import tabulate  # импорт для форматированного вывода таблиц

# Определение типов задач
class TaskType(Enum):
    COMPUTE = auto()      # Вычислительная задача
    NETWORK = auto()      # Сетевое задание
    STORAGE = auto()      # Задача хранения данных
    ENCRYPTION = auto()   # Задача шифрования
    DECRYPTION = auto()   # Задача дешифрования

    def cycles_per_operation(self):
        # Определяем количество тактов, необходимых для выполнения задачи конкретного типа
        cycles = {
            TaskType.COMPUTE: 4000,        # 4000 тактов на операцию вычислений
            TaskType.NETWORK: 2000,        # 2000 тактов на сетевую операцию
            TaskType.STORAGE: 3000,        # 3000 тактов на операцию хранения
            TaskType.ENCRYPTION: 5000,     # 5000 тактов на шифрование
            TaskType.DECRYPTION: 5000      # 5000 тактов на дешифрование
        }
        return cycles[self]

# Определение состояний задач
class TaskState(Enum):
    PENDING = auto()      # Задача ожидает выполнения
    RUNNING = auto()      # Задача выполняется
    COMPLETED = auto()    # Задача завершена
    DROPPED = auto()      # Задача дропнута (истек ttl)

# Определение класса для задач
@dataclass
class Task:
    priority: int  # Приоритет задачи
    id: int  # Уникальный идентификатор задачи
    data_size: int  # Размер данных в битах
    type: TaskType = TaskType.COMPUTE  # Тип задачи, по умолчанию COMPUTE
    operations: int = field(default=0)  # Количество операций, необходимых для выполнения задачи
    processor: Optional[int] = field(default=None, compare=False)  # Идентификатор процессора, на котором выполняется задача
    cluster: Optional[int] = field(default=None, compare=False)  # Идентификатор кластера внутри процессора
    core: Optional[int] = field(default=None, compare=False)  # Идентификатор ядра внутри кластера
    state: TaskState = field(default=TaskState.PENDING, compare=False)  # Текущее состояние задачи, по умолчанию PENDING
    execution_time: float = field(default=0.0, compare=False)  # Время выполнения задачи в секундах
    time_added: float = field(default=0.0, compare=False)  # Время добавления задачи в очередь
    time_started: float = field(default=0.0, compare=False)  # Время начала выполнения задачи
    waiting_time: float = field(default=0.0, compare=False)  # Время ожидания перед выполнением
    transfer_time: float = field(default=0.0, compare=False)  # Время передачи данных
    time_completed: Optional[float] = field(default=None, compare=False)  # Время завершения задачи
    lifetime: float = field(default=0.0, compare=False)  # Время жизни задачи (waiting_time)
    max_lifetime: float = field(default=0.0, compare=False)  # Максимальное время жизни задачи в секундах

    def __lt__(self, other):
        # Метод для сравнения задач по приоритету (нужен для очереди с приоритетом)
        return self.priority < other.priority

    def __eq__(self, other):
        # Метод для проверки равенства задач по приоритету
        return self.priority == other.priority

# Определение класса для процессора
class RISCVProcessor:
    def __init__(self, processor_id):
        self.id = processor_id # Уникальный идентификатор процессора
        self.cores = 64  # Общее количество ядер
        self.frequency = 2.0  # Частота процессора в ГГц
        self.clusters = 16  # Количество кластеров внутри процессора
        self.cores_per_cluster = 4  # Количество ядер в каждом кластере
        # Двумерный список для отслеживания статуса ядер (True - свободно, False - занято)
        self.core_status = [[True for _ in range(self.cores_per_cluster)] for _ in range(self.clusters)]
        self.core_lock = threading.Lock()  # Блокировка для потокобезопасного доступа к ядрам
        self.task_history = []  # История выполненных задач
        self.logger = logging.getLogger(f"Processor-{processor_id}")  # Логгер для процессора
        # Определяем множители времени жизни для каждого приоритета
        self.priority_multiplier_map = {
            1: 1.5,  # Приоритет 1: 1.5 × execution_time
            2: 2.0,  # Приоритет 2: 2.0 × execution_time
            3: 2.5,  # Приоритет 3: 2.5 × execution_time
            4: 2.75,  # Приоритет 4: 2.75 × execution_time
            5: 10.0  # Приоритет 5: 10.0 × execution_time
        }

    def allocate_core(self, task: Task) -> Optional[tuple]:
        """
        Попытка выделить свободное ядро для выполнения задачи.
        Если успешна, обновляет статус ядра и возвращает кортеж (cluster, core).
        Если нет свободных ядер, возвращает None.
        """
        # Блокируем доступ к ядрам для предотвращения одновременной попытки использования ядра разными потоками
        with self.core_lock:
            for cluster in range(self.clusters):  # Проходим по всем кластерам
                for core in range(self.cores_per_cluster):  # Проходим по всем ядрам в кластере
                    if self.core_status[cluster][core]:  # Если ядро свободно
                        self.core_status[cluster][core] = False  # Помечаем ядро как занятое
                        task.core = core + 1  # Присваиваем задаче идентификатор ядра (начиная с 1)
                        task.cluster = cluster + 1  # Присваиваем задаче идентификатор кластера (начиная с 1)
                        task.processor = self.id  # Присваиваем задаче идентификатор процессора
                        task.state = TaskState.RUNNING  # Обновляем состояние задачи на RUNNING
                        task.execution_time = self.calculate_execution_time(task)  # Вычисляем время выполнения задачи
                        multiplier = self.priority_multiplier_map.get(task.priority, 2.0)  # Множитель для ttl. Значение по умолчанию 2.0
                        task.max_lifetime = multiplier * task.execution_time    # Рассчитываем время жизни задачи
                        if task.max_lifetime < 0.5 : task.max_lifetime = 0.5
                        # Логируем информацию о выделении ядра
                        self.logger.debug(
                            f"Allocated Task {task.id} to Cluster {task.cluster}, Core {task.core}"
                            f" with execution time {task.execution_time:.6f} сек and max_lifetime {task.max_lifetime:.6f} сек")
                        return (task.cluster, task.core)  # Возвращаем кортеж с идентификаторами кластера и ядра
        return None  # Возвращаем None, если свободных ядер нет

    def calculate_execution_time(self, task: Task) -> float:
        """
        Рассчитывает время выполнения задачи на основе количества тактов, объему данных и частоты процессора.
        """
        cycles_per_op = task.type.cycles_per_operation()  # Получаем такты на операцию для типа задачи
        total_cycles = task.operations * cycles_per_op  # Общее количество тактов
        execution_time = total_cycles / (self.frequency * 1e9)  # Время выполнения в секундах
        return execution_time

    def release_core(self, cluster: int, core: int):
        """
        Освобождает ранее занятое ядро после завершения задачи.
        """
        with self.core_lock:  # Блокируем доступ к ядрам для других потоков
            self.core_status[cluster - 1][core - 1] = True  # Помечаем ядро как свободное (учитываем смещение на 1)
            self.logger.debug(f"Released Cluster {cluster}, Core {core}")  # Логируем освобождение ядра

class MultiprocessorRTOS:
    def __init__(self, amount_of_processors):
        """
        Инициализирует RTOS с заданной конфигурацией процессоров.
        """
        # Создаём процессор, выполняющий роль управляющего устройства
        self.central_processor = RISCVProcessor('central')

        # Создаём словарь процессоров, индексированных по их номеру (начиная с 1)
        self.processors = {
            idx + 1: RISCVProcessor(idx + 1)  # Идентификаторы процессоров начинаются с 1
            for idx in range(amount_of_processors)
        }
        self.task_queue = []  # Очередь задач с приоритетом (используем heapq)
        self.completed_tasks = []  # Список завершённых задач
        self.task_transfer_log = []  # Лог передачи задач
        self.queue_lock = threading.Lock()  # Блокировка для доступа к очереди задач
        self.completed_lock = threading.Lock()  # Блокировка для доступа к списку завершённых задач
        self.logger = logging.getLogger("MultiprocessorRTOS")  # Логгер для RTOS
        self.next_processor_id = 1  # Индекс следующего процессора для циклического распределения задач (начиная с 1)
        self.threads = []  # Список активных потоков
        self.transferring_tasks = []  # Список задач в процессе передачи
        self.executing_tasks = []  # Список задач в процессе выполнения
        self.processing_lock = threading.Lock()  # Блокировка для доступа к спискам задач в обработке

    def add_task(self, task: Task):
        """
        Добавляет задачу в очередь задач с приоритетом.
        """
        with self.queue_lock:  # Блокируем доступ к очереди задач
            task.time_added = time.time()  # Фиксируем время добавления задачи
            heapq.heappush(self.task_queue, task)  # Добавляем задачу в очередь с приоритетом
            self.logger.debug(f"Added Task {task.id} to queue with priority {task.priority}")  # Логируем добавление задачи

    def _select_processor(self, task: Task) -> Optional[RISCVProcessor]:
        """
        Выбирает свободный процессор для выполнения задачи.
        Использует циклический метод распределения задач по процессорам.
        """
        num_processors = len(self.processors)  # Общее количество процессоров
        for _ in range(num_processors):  # Проходимся по всем процессорам
            processor = self.processors[self.next_processor_id]  # Выбираем процессор по текущему индексу
            self.next_processor_id = (self.next_processor_id + 1) % (num_processors + 1)  # Обновляем индекс для следующего вызова
            if self.next_processor_id == 0:
                self.next_processor_id = 1  # Убедимся, что индекс не равен 0

            # Проверяем, есть ли свободные ядра у выбранного процессора
            with processor.core_lock:  # Блокируем доступ потокам к ядрам процессора
                for cluster in range(1, processor.clusters + 1):  # Проходим по всем кластерам (начиная с 1)
                    for core in range(1, processor.cores_per_cluster + 1):  # Проходим по всем ядрам в кластере (начиная с 1)
                        if processor.core_status[cluster - 1][core - 1]:  # Если ядро свободно (учитываем смещение на 1)
                            self.logger.debug(f"Selected Processor {processor.id} for Task {task.id}")  # Логируем выбор процессора
                            return processor  # Возвращаем выбранный процессор
        # Если нет свободных ядер на всех процессорах
        self.logger.debug(f"No available cores for Task {task.id} on any processor")  # Логируем отсутствие доступных ядер
        return None  # Возвращаем None

    def send_task(self, task: Task, processor: RISCVProcessor):
        """
        Моделирует отправку задачи на выбранный процессор через центральный процессор.
        """
        def transfer():
            """
            Внутренняя функция для передачи задачи на выбранный процессор.
            """
            # Пытаемся выделить ядро на целевом процессоре
            core_allocation = processor.allocate_core(task)
            if core_allocation:  # Если удалось выделить ядро
                with self.processing_lock:
                    self.transferring_tasks.append(task)  # Добавляем задачу в список передаваемых
                self.logger.info(f"Начата передача задачи {task.id} на Процессор {processor.id}")
                cluster, core = core_allocation
                time.sleep(task.transfer_time)  # Моделируем время передачи
                self.logger.info(f"Завершена передача задачи {task.id} на Процессор {processor.id} ")
                self.logger.debug(
                    f"Создание потока для задачи {task.id} на Процессоре {processor.id}, Кластер {cluster}, Ядро {core}")
                # Создаём новый поток для выполнения задачи
                thread = threading.Thread(target=self._execute_task, args=(task, processor, cluster, core))
                thread.start()  # Запускаем поток
                task.time_started = time.time()  # Фиксируем время начала выполнения задачи
                task.waiting_time = task.time_started - task.time_added  # Вычисляем длительность нахождения задачи в очереди
                with self.processing_lock:
                    self.executing_tasks.append(task)   # Добавляем задачу в список выполняющихся
                self.threads.append(thread)  # Добавляем поток в список активных потоков
            else:
                # Если не удалось выделить ядро, возвращаем задачу обратно в очередь
                self.logger.debug(
                    f"Нет доступных ядер для задачи {task.id} на Процессоре {processor.id}, возвращаем в очередь")
                self.add_task(task) # Возвращаем задачу в очередь

            # Освобождаем ядро главного процессора после передачи
            cluster_cp, core_cp = central_core
            self.central_processor.release_core(cluster_cp, core_cp)

            # Удаляем задачу из списка передаваемых
            with self.processing_lock:
                self.transferring_tasks.remove(task)

        # Пытаемся выделить ядро на главном процессоре для отправки
        central_core = self.central_processor.allocate_core(task)
        if central_core:  # Если удалось выделить ядро
            # Создаём поток для передачи задачи
            transfer_thread = threading.Thread(target=transfer)
            transfer_thread.start()  # Запускаем поток
            self.threads.append(transfer_thread)  # Добавляем поток в список активных потоков
        else:
            # Если не удалось выделить ядро на главном процессоре, возвращаем задачу обратно в очередь
            self.logger.debug(f"Нет доступных ядер на центральном процессоре для задачи {task.id}, возвращаем в очередь")
            self.add_task(task)  # Возвращаем задачу в очередь
            time.sleep(0.01)    # Кратковременная задержка перед следующей попыткой

    def process_tasks(self, total_simulation_time=60):
        """
        Обрабатывает задачи из очереди в течение заданного времени симуляции total_simulation_time.
        """
        start_time = time.time()  # Фиксируем время начала симуляции
        while time.time() - start_time < total_simulation_time:  # Пока не истекло время симуляции
            with self.queue_lock:  # Блокируем доступ к очереди задач
                if not self.task_queue: # Если очередь пуста
                    with self.processing_lock:  # Если список выполняемых задач пуст
                        if not self.transferring_tasks: # Если нет передаваемых задач
                            break # Если очередь задач пуста и нет задач в передаче, выходим из цикла
                    continue    # Если очередь пуста, но есть задачи в передаче, продолжаем ожидание
                task = heapq.heappop(self.task_queue)  # Извлекаем задачу с наивысшим приоритетом
            current_time = time.time()  # Текущее время
            waiting_time = current_time - task.time_added   # Время нахождения задачи в очереди
            # Установка max_lifetime перед проверкой
            if task.max_lifetime == 0.0:
                task.execution_time = self.central_processor.calculate_execution_time(task) # Определяем время выполнения
                multiplier = self.central_processor.priority_multiplier_map.get(task.priority, 2.0) # Множитель для времени жизни
                task.max_lifetime = multiplier * task.execution_time    # Установка времени жизни задачи
                if task.max_lifetime < 0.5 : task.max_lifetime = 0.5
            if waiting_time > task.max_lifetime:
                # Время выполнения задачи превышает max_lifetime и должна быть удалена
                self.logger.info(
                    f"Task {task.id} dropped due to exceeding max lifetime ({waiting_time:.2f} > {task.max_lifetime:.2f} сек)")
                # Добавляем удаленную задачу в список выполненных
                with self.completed_lock:
                    self.completed_tasks.append({
                        "task_id": task.id,
                        "processor": None,
                        "cluster": None,
                        "core": None,
                        "type": task.type.name,
                        "priority": task.priority,
                        "execution_time": 0.0,
                        "transfer_time": 0.0,
                        "waiting_time": waiting_time,
                        "data_size": task.data_size,
                        "state": "DROPPED",
                        "time_completed": 0.0,
                        "max_lifetime": task.max_lifetime
                    })
                continue  # Переходим к следующей задаче
            # Расчет времени пересылки данных
            transfer_rate = 128e9  # Скорость передачи данных 128 Гбит/с
            task.transfer_time = task.data_size / transfer_rate  # Время пересылки в секундах
            processor = self._select_processor(task)  # Выбираем процессор для выполнения задачи
            if processor:  # Если процессор выделен
                self.send_task(task, processor)  # Отправляем задачу через центральный процессор
            else:
                # Если не удалось выбрать процессор, возвращаем задачу обратно в очередь
                self.logger.debug(f"Нет подходящего процессора для задачи {task.id}, возвращаем в очередь")
                self.add_task(task) # Возвращаем задачу в очередь
                time.sleep(0.1)  # Задаем интервал времени до следующей попытки нахождения свободного процессора

        # Ждём завершения всех потоков после окончания времени симуляции
        for thread in self.threads:
            thread.join()

    def _execute_task(self, task: Task, processor: RISCVProcessor, cluster: int, core: int):
        """
        Симуляция выполнения задачи в процессоре.
        """
        self.logger.info(f"Выполнение задачи {task.id} на Процессоре {processor.id}, Кластер {cluster}, Ядро {core}")

        time.sleep(task.execution_time)  # Моделирование времени выполнения задачи
        self.logger.info(
            f"Время выполнения задачи {task.id} на Процессоре {processor.id}, Кластер {cluster}, Ядро {core} = {task.execution_time:.6f} сек")
        # Фиксируем время завершения задачи
        task.time_completed = time.time()
        # Рассчитываем время обработки задачи как время ожидания + время выполнения
        task.lifetime = task.time_completed - task.time_added
        # Освобождаем ядро после завершения задачи
        processor.release_core(cluster, core)
        task.state = TaskState.COMPLETED  # Обновляем состояние задачи на COMPLETED

        # Добавляем информацию о завершённой задаче в список завершённых задач
        with self.completed_lock:  # Блокируем доступ к списку завершённых задач
            self.completed_tasks.append({
                "task_id": task.id,
                "processor": processor.id,
                "cluster": cluster,
                "core": core,
                "type": task.type.name,
                "priority": task.priority,
                "execution_time": task.execution_time,
                "transfer_time": task.transfer_time,
                "waiting_time": task.waiting_time,
                "data_size": task.data_size,
                "state": task.state.name,
                "time_completed": task.time_completed,
                "max_lifetime": task.max_lifetime
            })
        # Удаляем задачу из списка выполняющихся
        with self.processing_lock:
            self.executing_tasks.remove(task)
        # Логируем завершение задачи
        self.logger.info(f"Завершена задача {task.id} на Процессоре {processor.id}, Кластер {cluster}, Ядро {core}")

    def generate_task_report(self, top_n=100):
        """
        Генерирует отчёт о выполненных задачах.
        """
        with self.completed_lock:  # Блокируем доступ к списку завершённых задач
            return self.completed_tasks  # Возвращаем список завершённых задач

def generate_ieee_802_3ba_tasks(count=500, min_size=64, max_size=128):
    """
    Генерирует список случайных задач для симуляции.
    """
    tasks = []  # Список для хранения задач
    for i in range(count):
        task_type = random.choice(list(TaskType))  # Случайный выбор типа задачи
        data_size = random.randint(min_size, max_size)  # Размер данных в битах (случайный из диапазона)
        # Определяем количество операций на основе размера данных и типа задачи
        # Предполагаем, что одна операция обрабатывает 1 бит данных
        ops = max(1, data_size // 64)
        task = Task(
            priority=random.randint(1, 5),  # Случайный приоритет от 1 до 5
            id=i + 1,  # Уникальный идентификатор задачи (начиная с 1)
            data_size=data_size,
            type=task_type,
            operations=ops,
        )
        tasks.append(task)  # Добавляем задачу в список
    return tasks  # Возвращаем список задач

def generate_priority_distribution(df):
    """
        Генерирует график распределения приоритетов задач.
    """
    plt.figure(figsize=(10,6))
    sns.countplot(x='Приоритет', data=df, palette='viridis')
    plt.title('Распределение Приоритетов Задач')
    plt.xlabel('Приоритет')
    plt.ylabel('Количество Задач')
    plt.savefig('priority_distribution.png')
    plt.close()

def generate_execution_time_distribution(df):
    """
       Генерирует график распределения времени выполнения задач.
    """
    plt.figure(figsize=(10,6))
    sns.histplot(df['Время выполнения (сек)'], bins=30, kde=True, color='skyblue')
    plt.title('Распределение Времени Выполнения Задач')
    plt.xlabel('Время Выполнения (сек)')
    plt.ylabel('Частота')
    plt.savefig('execution_time_distribution.png')
    plt.close()

def generate_task_state_pie_chart(df):
    """
        Генерирует круговую диаграмму распределения состояний задач.
    """
    plt.figure(figsize=(8,8))
    state_counts = df['Состояние'].value_counts()
    plt.pie(state_counts, labels=state_counts.index, autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
    plt.title('Распределение Состояний Задач')
    plt.savefig('task_state_pie_chart.png')
    plt.close()

def generate_processor_task_distribution(df):
    """
    Генерирует гистограмму, показывающую количество задач, обработанных каждым процессором.
    """
    plt.figure(figsize=(10, 6))
    # Фильтруем только выполненные задачи
    completed_tasks = df[df['Состояние'] == 'COMPLETED']
    # Считаем количество задач по каждому процессору
    task_counts = completed_tasks['Процессор'].value_counts().sort_index()
    # Создаем барплот
    ax = sns.barplot(x=task_counts.index, y=task_counts.values, palette='viridis')
    plt.title('Количество Задач, Обработанных Каждым Процессором')
    plt.xlabel('Процессор')
    plt.ylabel('Количество Задач')

    # Добавляем аннотации на каждый столбец
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.tight_layout()
    plt.savefig('processor_task_distribution.png')
    plt.close()


def generate_core_task_distribution(df):
    """
    Генерирует тепловые карты распределения задач между ядрами на каждом процессоре.
    """
    # Фильтруем только выполненные задачи
    completed_tasks = df[df['Состояние'] == 'COMPLETED']

    # Получаем список уникальных процессоров
    processors = completed_tasks['Процессор'].dropna().unique()

    for processor in processors:
        # Фильтруем задачи для текущего процессора
        processor_tasks = completed_tasks[completed_tasks['Процессор'] == processor]

        # Получаем список уникальных кластеров для текущего процессора
        clusters = processor_tasks['Кластер'].dropna().unique()

        # Создаем пустую матрицу для тепловой карты
        heatmap_data = pd.DataFrame()

        for cluster in clusters:
            # Фильтруем задачи для текущего кластера
            cluster_tasks = processor_tasks[processor_tasks['Кластер'] == cluster]

            # Считаем количество задач по каждому ядру в кластере
            core_counts = cluster_tasks['Ядро'].value_counts().sort_index()

            # Добавляем данные в матрицу тепловой карты
            heatmap_data[f'Кластер {cluster}'] = core_counts

        # Заполняем отсутствующие значения нулями
        heatmap_data = heatmap_data.fillna(0).astype(int)

        # Создаем тепловую карту
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data.T, annot=True, fmt="d", cmap='YlGnBu', cbar_kws={'label': 'Количество задач'})
        plt.title(f'Распределение Задач между Ядрами на Процессоре {processor}')
        plt.xlabel('Ядра')
        plt.ylabel('Кластеры')
        plt.tight_layout()
        filename = f'processor_{processor}_core_distribution.png'
        plt.savefig(filename)
        plt.close()
        print(f"График распределения задач по ядрам для Процессора {processor} сохранён как {filename}")

def main():
    # Настройка базовой конфигурации логирования
    logging.basicConfig(
        level=logging.INFO,  # Уровень логирования INFO для вывода основных сообщений
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Формат сообщений логов
    )

    # Количество процессоров в системе (исключая центральный)
    amount_of_processors = 2
    rtos = MultiprocessorRTOS(amount_of_processors)  # Создаём экземпляр RTOS с заданной конфигурацией процессоров
    tasks = generate_ieee_802_3ba_tasks()  # Генерируем список задач
    for task in tasks:
        rtos.add_task(task)  # Добавляем каждую задачу в очередь RTOS

    rtos.process_tasks(total_simulation_time=60)  # Запускаем обработку задач на total_simulation_time секунд

    task_report = rtos.generate_task_report()  # Получаем отчёт о выполненных задачах

    print("Отчет о выполнении задач:")
    headers = [
        "Задача ID", "Процессор", "Кластер", "Ядро", "Приоритет",
        "Тип", "Время ожидания (сек)", "Время пересылки (сек)",
        "Время выполнения (сек)", "Размер (бит)", "ttl (сек)", "Состояние"
    ]
    # Формируем таблицу с данными о задачах
    table = [
        [
            task["task_id"],
            task["processor"],
            task["cluster"],
            task["core"],
            task["priority"],
            task["type"],
            f"{task['waiting_time']:.6f}",
            f"{task['transfer_time']:.6f}",
            f"{task['execution_time']:.6f}",
            task["data_size"],
            f"{task['max_lifetime']:.6f}" if task["max_lifetime"] else "-",
            task["state"]
        ]
        for task in task_report
    ]
    # Выводим таблицу в консоль с помощью tabulate
    print(tabulate(table, headers=headers, tablefmt="grid"))
    print(f"Всего выполнено задач: {len([t for t in task_report if t['state'] == 'COMPLETED'])}")
    print(f"Всего дропнуто задач: {len([t for t in task_report if t['state'] == 'DROPPED'])}")

    # Создаём DataFrame из списка словарей
    df = pd.DataFrame(task_report)

    # Определяем список столбцов, которые хотим включить в Excel
    desired_columns = [
        "task_id", "processor", "cluster", "core", "priority",
        "type", "waiting_time", "transfer_time",
        "execution_time", "data_size", "max_lifetime", "state"
    ]

    # Проверяем, что все желаемые столбцы присутствуют в DataFrame
    existing_columns = [col for col in desired_columns if col in df.columns]
    missing_columns = [col for col in desired_columns if col not in df.columns]
    if missing_columns:
        print(f"Внимание: отсутствуют столбцы {missing_columns} в данных отчёта.")

    # Выбираем только необходимые столбцы
    df_filtered = df[existing_columns]

    # Переименовываем столбцы для лучшей читаемости в Excel
    df_filtered.rename(columns={
        "task_id": "Задача ID",
        "processor": "Процессор",
        "cluster": "Кластер",
        "core": "Ядро",
        "priority": "Приоритет",
        "type": "Тип",
        "waiting_time": "Время ожидания (сек)",
        "transfer_time": "Время пересылки (сек)",
        "execution_time": "Время выполнения (сек)",
        "data_size": "Размер (бит)",
        "max_lifetime": "ttl (сек)",
        "state": "Состояние"
    }, inplace=True)

    # Сохраняем DataFrame в Excel-файл
    excel_filename = "task_report.xlsx"
    try:
        df_filtered.to_excel(excel_filename, index=False)
        print(f"Отчет успешно сохранён в файл {excel_filename}")
    except Exception as e:
        print(f"Ошибка при сохранении отчета в Excel: {e}")

    # Вызываем генерацию графиков
    generate_priority_distribution(df_filtered)
    generate_execution_time_distribution(df_filtered)
    generate_task_state_pie_chart(df_filtered)
    generate_processor_task_distribution(df_filtered)
    generate_core_task_distribution(df_filtered)
