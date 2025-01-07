import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import pandas as pd
import logging

# Импортируем из Вашего файла riscv_rtos.py:
from riscv_rtos import (
    MultiprocessorRTOS,
    Task, TaskType,
    generate_ieee_802_3ba_tasks,  # где ops = data_size // 64
    generate_priority_distribution,
    generate_execution_time_distribution,
    generate_task_state_pie_chart,
    generate_processor_task_distribution,
    generate_core_task_distribution
)

###############################################################################
#  Класс-Handler для вывода логов logging в Text-виджет
###############################################################################
class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        # Форматируем сообщение
        msg = self.format(record)

        # Выводим в Text (через after, чтобы не блокировать GUI)
        def append():
            self.text_widget.insert(tk.END, msg + "\n")
            self.text_widget.see(tk.END)
        self.text_widget.after(0, append)


###############################################################################
#  Основной класс GUI
###############################################################################
class RTOSGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RISC-V RTOS Simulation (GUI) with Log")

        # Параметры симуляции
        self.num_processors_var = tk.IntVar(value=2)
        self.sim_time_var = tk.IntVar(value=60)

        # Параметры для случайных задач
        self.random_count_var = tk.IntVar(value=5)
        self.random_min_data_size_var = tk.IntVar(value=64)
        self.random_max_data_size_var = tk.IntVar(value=128)

        # Список пользовательских задач
        self.custom_tasks = []
        self.task_id_counter = 1

        # Будущий объект RTOS и результаты
        self.rtos = None
        self.task_report_df = None

        # Поток для симуляции
        self.simulation_thread = None
        self.stop_simulation = False

        # 1) Создадим Canvas + общий вертикальный скролл
        self._create_scrollable_area()

        # 2) Создадим в main_frame все виджеты (LabelFrame, Treeview, Text и т.п.)
        self._create_widgets_in_main_frame()

        # 3) Настроим логгер (вывод в Text)
        self._setup_logger()

    ###########################################################################
    #  Создаём Canvas + Scrollbar, оборачиваем в main_frame
    ###########################################################################
    def _create_scrollable_area(self):
        # Canvas, занимающий всё окно (для вертикальной прокрутки)
        self.scroll_canvas = tk.Canvas(self, highlightthickness=0)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Вертикальный Scrollbar, прикреплённый к Canvas
        self.v_scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.scroll_canvas.configure(yscrollcommand=self.v_scrollbar.set)

        # Frame, куда будем складывать всё остальное
        self.main_frame = ttk.Frame(self.scroll_canvas)
        self.scroll_canvas.create_window((0, 0), window=self.main_frame, anchor="nw")

        # При изменении размера main_frame пересчитываем scrollregion
        self.main_frame.bind(
            "<Configure>",
            lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))
        )

    ###########################################################################
    #  Создаём все элементы (виджеты) внутри main_frame
    ###########################################################################
    def _create_widgets_in_main_frame(self):
        # ------------------- Настройки симуляции -------------------
        frame_settings = ttk.LabelFrame(self.main_frame, text="Настройки симуляции")
        frame_settings.pack(fill="x", padx=5, pady=5)

        ttk.Label(frame_settings, text="Количество процессоров:").grid(row=0, column=0, padx=5, pady=2, sticky="e")
        ttk.Entry(frame_settings, textvariable=self.num_processors_var, width=5).grid(row=0, column=1, padx=5, pady=2, sticky="w")

        ttk.Label(frame_settings, text="Время симуляции (сек):").grid(row=1, column=0, padx=5, pady=2, sticky="e")
        ttk.Entry(frame_settings, textvariable=self.sim_time_var, width=5).grid(row=1, column=1, padx=5, pady=2, sticky="w")

        # ------------------- Рандомные задачи -------------------
        frame_random = ttk.LabelFrame(self.main_frame, text="Случайные задачи")
        frame_random.pack(fill="x", padx=5, pady=5)

        ttk.Label(frame_random, text="Кол-во задач:").grid(row=0, column=0, padx=5, pady=2, sticky="e")
        ttk.Entry(frame_random, textvariable=self.random_count_var, width=5).grid(row=0, column=1, padx=5, pady=2, sticky="w")

        ttk.Label(frame_random, text="Min Size (бит):").grid(row=1, column=0, padx=5, pady=2, sticky="e")
        ttk.Entry(frame_random, textvariable=self.random_min_data_size_var, width=5).grid(row=1, column=1, padx=5, pady=2, sticky="w")

        ttk.Label(frame_random, text="Max Size (бит):").grid(row=2, column=0, padx=5, pady=2, sticky="e")
        ttk.Entry(frame_random, textvariable=self.random_max_data_size_var, width=5).grid(row=2, column=1, padx=5, pady=2, sticky="w")

        ttk.Button(frame_random, text="Добавить рандомные задачи", command=self.add_random_tasks).grid(
            row=0, column=2, rowspan=3, padx=10, pady=2, sticky="ns"
        )

        # ------------------- Ручное добавление задач -------------------
        frame_tasks = ttk.LabelFrame(self.main_frame, text="Ручное добавление задачи")
        frame_tasks.pack(fill="x", padx=5, pady=5)

        self.priority_var = tk.IntVar(value=3)
        self.type_var = tk.StringVar(value="COMPUTE")
        self.data_size_var = tk.IntVar(value=128)
        self.operations_var = tk.IntVar(value=2)  # ops = data_size//64 (по желанию)

        ttk.Label(frame_tasks, text="Приоритет (1..5):").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(frame_tasks, textvariable=self.priority_var, width=5).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(frame_tasks, text="Тип задачи:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        cmb_type = ttk.Combobox(frame_tasks, textvariable=self.type_var,
                                values=[t.name for t in TaskType], state="readonly", width=12)
        cmb_type.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(frame_tasks, text="Размер (бит):").grid(row=2, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(frame_tasks, textvariable=self.data_size_var, width=8).grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(frame_tasks, text="Операции (Ops):").grid(row=3, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(frame_tasks, textvariable=self.operations_var, width=8).grid(row=3, column=1, padx=5, pady=2)

        ops_hint_txt = "Обычно ops = data_size//64 (1 операция = 64 бит)"
        ttk.Label(frame_tasks, text=ops_hint_txt, foreground="gray").grid(row=4, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        ttk.Button(frame_tasks, text="Добавить задачу", command=self.add_manual_task).grid(row=5, column=0, columnspan=2, pady=5)

        # ------------------- Список задач + удаление -------------------
        frame_task_list = ttk.LabelFrame(self.main_frame, text="Список задач")
        frame_task_list.pack(fill="both", expand=True, padx=5, pady=5)

        self.task_listbox = tk.Listbox(frame_task_list, height=8, width=70)
        self.task_listbox.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(frame_task_list, orient="vertical", command=self.task_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.task_listbox.config(yscrollcommand=scrollbar.set)

        ttk.Button(frame_task_list, text="Удалить выбранную", command=self.delete_selected_task).pack(pady=5)
        ttk.Button(frame_task_list, text="Удалить все задачи", command=self.delete_all_tasks).pack(pady=5)

        # ------------------- Кнопки управления -------------------
        frame_control = ttk.Frame(self.main_frame)
        frame_control.pack(fill="x", padx=5, pady=5)

        ttk.Button(frame_control, text="Запустить симуляцию", command=self.start_simulation).pack(side="left", padx=5)
        ttk.Button(frame_control, text="Сохранить Excel", command=self.save_excel).pack(side="left", padx=5)
        ttk.Button(frame_control, text="Сгенерировать графики", command=self.generate_charts).pack(side="left", padx=5)

        # ------------------- Text для лога (выше кнопок) -------------------
        frame_log = ttk.LabelFrame(self.main_frame, text="Лог исполнения")
        frame_log.pack(fill="both", expand=False, padx=5, pady=5)

        self.log_text = tk.Text(frame_log, wrap="word", height=10)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        # ------------------- Таблица результатов (Treeview) -------------------
        frame_table = ttk.LabelFrame(self.main_frame, text="Результаты (все столбцы)")
        frame_table.pack(fill="both", expand=True, padx=5, pady=5)

        self.columns = (
            "task_id", "processor", "cluster", "core",
            "priority", "type", "waiting_time", "transfer_time",
            "execution_time", "data_size", "max_lifetime", "state"
        )
        self.column_headers = {
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
        }

        self.report_tree = ttk.Treeview(frame_table, columns=self.columns, show='headings', height=10)
        self.report_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        for col in self.columns:
            self.report_tree.heading(col, text=self.column_headers[col], anchor='center')
            self.report_tree.column(col, anchor='center', width=120)

        scrollbar_table = ttk.Scrollbar(frame_table, orient=tk.VERTICAL, command=self.report_tree.yview)
        scrollbar_table.pack(side=tk.RIGHT, fill=tk.Y)
        self.report_tree.configure(yscrollcommand=scrollbar_table.set)

    ###########################################################################
    #  Настраиваем глобальный логгер на вывод в self.log_text
    ###########################################################################
    def _setup_logger(self):
        logger = logging.getLogger()  # root logger
        logger.setLevel(logging.INFO)  # Будем видеть INFO и выше

        # Удаляем старые handlers (чтобы не дублировать в консоли)
        for h in logger.handlers[:]:
            logger.removeHandler(h)

        # Наш кастомный Handler для Text
        text_handler = TextHandler(self.log_text)
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        text_handler.setFormatter(formatter)
        logger.addHandler(text_handler)

    ###########################################################################
    # Логика кнопок (добавление задач, удаление, симуляция и т.д.)
    ###########################################################################

    def add_random_tasks(self):
        """
        Генерируем указанное количество случайных задач (ops = data_size//64).
        """
        n = self.random_count_var.get()
        min_ds = self.random_min_data_size_var.get()
        max_ds = self.random_max_data_size_var.get()

        random_tasks = generate_ieee_802_3ba_tasks(count=n, min_size=min_ds, max_size=max_ds)
        # Обновим ID, чтобы шли дальше по порядку
        for rt in random_tasks:
            rt.id = self.task_id_counter
            self.task_id_counter += 1
            self.custom_tasks.append(rt)
            self.task_listbox.insert(tk.END, f"ID={rt.id}, P={rt.priority}, Type={rt.type.name}, "
                                             f"Size={rt.data_size}, Ops={rt.operations}")

    def add_manual_task(self):
        """
        Добавляем задачу из полей ввода в список custom_tasks и в Listbox.
        """
        p = self.priority_var.get()
        t = self.type_var.get()
        ds = self.data_size_var.get()
        ops = self.operations_var.get()

        new_task = Task(
            priority=p,
            id=self.task_id_counter,
            data_size=ds,
            type=TaskType[t],
            operations=ops
        )
        self.custom_tasks.append(new_task)
        self.task_id_counter += 1

        self.task_listbox.insert(
            tk.END, f"ID={new_task.id}, P={p}, Type={t}, Size={ds}, Ops={ops}"
        )

    def delete_selected_task(self):
        """
        Удаляем выбранную задачу из listbox и из self.custom_tasks.
        """
        selection = self.task_listbox.curselection()
        if not selection:
            messagebox.showinfo("Нет выбора", "Сначала выберите задачу из списка.")
            return
        index = selection[0]
        self.task_listbox.delete(index)
        del self.custom_tasks[index]

    def delete_all_tasks(self):
        """
        Удаляем все задачи из списка.
        """
        self.custom_tasks.clear()
        self.task_listbox.delete(0, tk.END)
        self.task_id_counter = 1  # сбросим счётчик

    def start_simulation(self):
        """
        Запускаем симуляцию в отдельном потоке, чтобы не зависал GUI.
        """
        if self.simulation_thread and self.simulation_thread.is_alive():
            messagebox.showwarning("Идёт симуляция", "Симуляция уже запущена!")
            return

        # Создаём RTOS
        num_proc = self.num_processors_var.get()
        self.rtos = MultiprocessorRTOS(num_proc)

        # Добавляем задачи
        for t in self.custom_tasks:
            self.rtos.add_task(t)

        self.stop_simulation = False
        sim_time = self.sim_time_var.get()
        self.simulation_thread = threading.Thread(
            target=self.run_simulation_thread,
            args=(sim_time,),
            daemon=True
        )
        self.simulation_thread.start()

    def run_simulation_thread(self, sim_time):
        """
        Функция, которая реально запускает .process_tasks() + выводит результаты.
        """
        start_time = time.time()
        self.rtos.process_tasks(total_simulation_time=sim_time)
        end_time = time.time()

        # Получаем отчёт
        report = self.rtos.generate_task_report()
        self.task_report_df = pd.DataFrame(report)

        # Подсчитываем итоги
        completed = sum(1 for r in report if r["state"] == "COMPLETED")
        dropped = sum(1 for r in report if r["state"] == "DROPPED")

        # Просто Info в logger (уйдёт в лог TextHandler)
        logging.info(f"Симуляция завершена за {end_time - start_time:.2f} сек.")
        logging.info(f"Выполнено задач: {completed}, Дропнуто: {dropped}.")

        # Обновляем таблицу (Treeview)
        self._update_report_table(report)

    def _update_report_table(self, report):
        """
        Заполняем Treeview данными из отчёта.
        """
        # Сначала очистим предыдущие данные
        for row in self.report_tree.get_children():
            self.report_tree.delete(row)

        # Добавим строки
        for t in report:
            values = (
                t["task_id"],
                t["processor"],
                t["cluster"],
                t["core"],
                t["priority"],
                t["type"],
                f"{t['waiting_time']:.6f}",
                f"{t['transfer_time']:.6f}",
                f"{t['execution_time']:.6f}",
                t["data_size"],
                f"{t['max_lifetime']:.6f}" if t["max_lifetime"] else "-",
                t["state"]
            )
            self.report_tree.insert("", tk.END, values=values)

    def save_excel(self):
        """
        Сохраняем результаты в Excel (task_report.xlsx).
        """
        if self.task_report_df is None or self.task_report_df.empty:
            messagebox.showinfo("Нет данных", "Сначала запустите симуляцию, чтобы получить отчёт.")
            return

        df_filtered = self.rename_columns_for_report(self.task_report_df)
        excel_filename = "task_report.xlsx"
        try:
            df_filtered.to_excel(excel_filename, index=False)
            logging.info(f"Отчёт сохранён в файл {excel_filename}")
        except Exception as e:
            logging.error(f"Не удалось сохранить Excel: {e}")

    def generate_charts(self):
        """
        Генерируем 4 графика: Priority, Exec Time, Task State Pie, Processor Distribution
        """
        if self.task_report_df is None or self.task_report_df.empty:
            messagebox.showinfo("Нет данных", "Сначала запустите симуляцию, чтобы получить отчёт.")
            return

        df_filtered = self.rename_columns_for_report(self.task_report_df)
        generate_priority_distribution(df_filtered)
        generate_execution_time_distribution(df_filtered)
        generate_task_state_pie_chart(df_filtered)
        generate_processor_task_distribution(df_filtered)
        generate_core_task_distribution(df_filtered)


        logging.info("Графики сохранены в виде PNG-файлов.")

    def rename_columns_for_report(self, df):
        """
        Переименовываем столбцы под русские названия (как в исходном коде).
        """
        df_copy = df.copy()
        rename_map = {
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
        }
        df_copy.rename(columns=rename_map, inplace=True)
        return df_copy


def main():
    app = RTOSGUI()
    app.geometry("1000x800")  # Можно указать стартовый размер
    app.mainloop()

if __name__ == "__main__":
    main()
