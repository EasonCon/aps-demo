import random
import logging
import pandas as pd

from deap import base, creator, tools
from datetime import datetime, timedelta
from LoadData import tasks, resources
from TypeDefine import Resource, Task, ResourceSelectionStrategy, TaskSortStrategy, TaskType
from typing import List
from openpyxl import load_workbook
from openpyxl.styles import Alignment

random.seed(0)


class Scheduler(object):
    def __init__(self, tasks: List[Task], resources: List[Resource]):
        self.tasks = tasks  # tasks:level 1,但是记录了 child task
        self.resources = resources

    def _schedule(self, select_strategy: ResourceSelectionStrategy):
        # BOM只有两层 -> 简单处理
        execute_list = []
        for task in self.tasks:
            if task.children_task:
                execute_list.extend([child for child in task.children_task])
            execute_list.append(task)
        for task in execute_list:
            self._scheduling_one_task(task, select_strategy)

        for task in execute_list:
            for op in task.craft_paths:
                if not op.assigned_resource or not op.assigned_start_time or not op.assigned_end_time:
                    assert 0 == 1

    def based_schedule(self, select_strategy: ResourceSelectionStrategy, sort_stragety: TaskSortStrategy | None):
        """
        根据事先定义好的顺序按照一定的资源选择策略进行时间推导
        :return:
        """
        self._clear()
        for resource in self.resources:
            resource.local_time = datetime(2024, 9, 1)

        if sort_stragety == TaskSortStrategy.PLANENDDATE:
            self.tasks_sort()
        elif sort_stragety == TaskSortStrategy.CR:
            self.tasks_sort_by_CR()
        else:
            # self.tasks_sort_by_OPT()  # deap 优化不可用，设置为pass
            pass
        self._schedule(select_strategy)

    def tasks_sort(self):
        self.tasks = sorted(self.tasks, key=lambda x: x.planned_start_date)

    def tasks_sort_by_CR(self):
        self.tasks = sorted(self.tasks, key=lambda x: x.cr)

    def print_gantt(self):
        # data = []
        #
        # # 添加 tqdm 进度条来跟踪数据转换进度
        # for task in self.tasks:
        #     for op in task.craft_paths:
        #         data.append({
        #             'Task': task.id,
        #             'Operation': op.name,
        #             'Start': op.assigned_start_time,
        #             'Finish': op.assigned_end_time,
        #             'Resource': op.assigned_resource
        #         })
        #
        # df = pd.DataFrame(data)
        #
        # # 使用 plotly 创建甘特图
        # fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Resource", text="Operation")
        # fig.update_yaxes(categoryorder="total ascending")
        # fig.update_layout(
        #     title="Gantt Chart of Tasks",
        #     xaxis_title="Time",
        #     yaxis_title="Task",
        #     showlegend=True
        # )

        # 显示图表
        # fig.show(renderer='svg')
        # fig.show(renderer='webgl')

        # project = gantt.Project(name="Production Schedule")
        #
        # # 遍历任务并添加到项目中
        # for task in self.tasks:
        #     for operation in task.craft_paths:
        #         # 创建每个操作的甘特任务
        #         gantt_task = gantt.Task(
        #             name=f"{task.id} - {operation.name}",
        #             start=operation.assigned_start_time.date(),
        #             stop=operation.assigned_end_time.date(),
        #             resources=operation.assigned_resource,
        #             percent_done=100
        #         )
        #         # 将任务添加到项目中
        #         project.add_task(gantt_task)
        #
        # # 生成甘特图
        # project.make_svg_for_tasks(filename="gantt_chart.svg", today=datetime.now().date())
        # print("甘特图已保存为 gantt_chart.svg")
        pass

    def cal_kpi(self) -> tuple[dict[Resource, float], dict[Task, timedelta]]:
        # 资源生产效率
        resource_efficiency: dict[Resource, float] = {}
        for resource in self.resources:
            resource_efficiency[resource] = 0

        for resource in self.resources:
            start: datetime = datetime(2024, 9, 1)
            end: datetime = resource.local_time
            duration: timedelta = timedelta(seconds=0)
            for operation in resource.operation_list:
                duration += operation.assigned_end_time - operation.assigned_start_time

            if end == start:
                continue
            resource_efficiency[resource] = duration / (end - start)

        # 任务延期情况
        task_delay: dict[Task, timedelta] = {}
        all_tasks: List[Task] = []
        all_tasks.extend(self.tasks)
        for task in self.tasks:
            if task.children_task:
                all_tasks.extend(task.children_task)

        for task in all_tasks:
            task_delay[task] = task.delay

        return resource_efficiency, task_delay

    def out_put(self, display: bool = False, output_file: bool = False, path=None):
        if display:
            header_format = "{:<10} {:<5} {:<25} {:<19} {:<19}"
            row_format = "{:<10} {:<5} {:<25} {:<19} {:<19}"

            # 打印标题
            print(header_format.format("作业单号", "工序号", "工序名称", "开始时间", "结束时间"))
            print("=" * 100)  # 分隔线

            for task in self.tasks:
                for i, operation in enumerate(task.craft_paths, 1):
                    start_time = operation.assigned_start_time.strftime('%Y-%m-%d %H:%M')
                    end_time = operation.assigned_end_time.strftime('%Y-%m-%d %H:%M')
                    print(row_format.format(task.id, i, operation.name, start_time, end_time))

        # 计算并输出平均资源效率、总延期时间和延期任务数量
        efficient, delays = self.cal_kpi()

        average_efficiency = sum(efficient.values()) / len(efficient) if efficient else 0
        total_delay_hours = sum(delays.values())
        delayed_tasks_count = sum(1 for value in delays.values() if value > 0)

        print(f"{'平均资源效率:':<20} {average_efficiency:>10.2f}")
        print(f"{'任务总延期:':<20} {total_delay_hours:>10.2f} 小时")
        print(f"{'延期任务数量:':<20} {delayed_tasks_count:>10}")
        print("-" * 100)

        if output_file:
            # 资源视角 gantt
            resource_gantt_writer = pd.ExcelWriter(path + 'resource_gantt.xlsx', engine='openpyxl')
            for resource in self.resources:
                data = {
                    "作业单号": [op.parent_task.id for op in resource.operation_list],
                    "工序名称": [op.name for op in resource.operation_list],
                    "物料编码": [op.material_id for op in resource.operation_list],
                    "开始时间": [op.assigned_start_time for op in resource.operation_list],
                    "结束时间": [op.assigned_end_time for op in resource.operation_list],
                    "工作时间(时)": [round((op.assigned_end_time - op.assigned_start_time).total_seconds() / 3600, 2) for op in resource.operation_list],
                    "全部可用资源": [" ".join([re.name for re in op.available_resource]) for op in resource.operation_list]
                }
                df = pd.DataFrame(data)
                df.to_excel(resource_gantt_writer, sheet_name=resource.name, index=False)
            resource_gantt_writer._save()

            # 作业单视角 gantt
            task_gantt_writer = pd.ExcelWriter(path + 'task_gantt.xlsx', engine='openpyxl')
            all_tasks: List[Task] = []
            all_tasks.extend(self.tasks)
            for task in self.tasks:
                if task.children_task:
                    all_tasks.extend(task.children_task)
            all_tasks = sorted(all_tasks, key=lambda x: x.craft_paths[-1].assigned_start_time)

            type_dict = {
                TaskType.PARENT: "父级MO",
                TaskType.CHILD: "子级任务"
            }

            data = {
                "作业单号": [task.id for task in all_tasks],
                "物料编码": [task.material_id for task in all_tasks],
                "关联类型": [type_dict[task.type] for task in all_tasks],
                "生产数量": [task.uncleared_quantity for task in all_tasks],
                "子任务": [",".join([str(child.id) for child in task.children_task]) for task in all_tasks],
                "开始时间": [task.craft_paths[0].assigned_start_time for task in all_tasks],
                "结束时间": [task.craft_paths[-1].assigned_end_time for task in all_tasks],
            }
            df = pd.DataFrame(data)
            df.to_excel(task_gantt_writer, sheet_name="作业单", index=False)
            task_gantt_writer._save()

            # 工序视角: 先计算op 的分布，然后再关联op 的信息
            if path == "opt_out/":
                operation_gantt_writer = pd.ExcelWriter(path + 'opt_operation_gantt.xlsx', engine='openpyxl')
            else:
                operation_gantt_writer = pd.ExcelWriter(path + 'operation_gantt.xlsx', engine='openpyxl')

            date_range = pd.date_range(datetime(2024, 9, 1).date(), datetime(2027, 12, 31).date())  # 假定静态，时间长度不会超过这个范围
            df = pd.DataFrame(index=[op.id for task in all_tasks for op in task.craft_paths], columns=sorted(date_range, key=lambda x: x))

            # 填充df，计算工序数量
            for task in all_tasks:
                for op in task.craft_paths:
                    total_quantity = op.parent_task.uncleared_quantity
                    start_date = op.assigned_start_time.date()
                    end_date = op.assigned_end_time.date()
                    if start_date == end_date:
                        df.loc[op.id, pd.to_datetime(start_date)] = total_quantity
                        continue
                    op_range = pd.date_range(start_date, end_date)
                    quantity_range = []

                    for i, date in enumerate(op_range):
                        if i == 0:
                            first_day_working_minutes = (date + timedelta(days=1) - op.assigned_start_time).total_seconds() / 60
                            first_day_amount = round(first_day_working_minutes / op.minutes_per_bear * op.quantity_per_bear)
                            quantity_range.append(first_day_amount)
                        elif i == len(op_range) - 1:
                            last_day_working_minutes = (op.assigned_end_time - date).total_seconds() / 60
                            last_day_amount = round(last_day_working_minutes / op.minutes_per_bear * op.quantity_per_bear)
                            quantity_range.append(last_day_amount)
                        else:
                            working_minutes = timedelta(days=1).total_seconds() / 60
                            amount = round(working_minutes / op.minutes_per_bear * op.quantity_per_bear)
                            quantity_range.append(amount)

                    for quantity, date in zip(quantity_range, op_range):
                        df.loc[op.id, date] = quantity

            # debug
            # filtered_df = df.loc[df.index == '压铸']
            # non_empty_filtered_df = filtered_df.dropna(how='all')
            # print(non_empty_filtered_df)

            # 补充工序信息
            added_info = {
                "物料": [op.parent_task.material_id for task in all_tasks for op in task.craft_paths],
                "物料描述": [op.parent_task.material_describe for task in all_tasks for op in task.craft_paths],
                "作业单号": [op.parent_task.id for task in all_tasks for op in task.craft_paths],
                "需求日期": [pd.to_datetime(op.parent_task.planned_end_date) for task in all_tasks for op in task.craft_paths],
                "工序编码": [op.order for task in all_tasks for op in task.craft_paths],
                "工序名称": [op.name for task in all_tasks for op in task.craft_paths],
                "设备": [op.assigned_resource.name for task in all_tasks for op in task.craft_paths],
                "生产数量": [op.parent_task.uncleared_quantity for task in all_tasks for op in task.craft_paths]
            }
            added_info_df = pd.DataFrame(added_info)
            added_info_df = added_info_df.reset_index(drop=True)
            df.columns = pd.to_datetime(df.columns).date
            df = df.reset_index(drop=True)
            df = pd.concat([added_info_df, df], axis=1)  # 合并
            df_sorted = df.sort_values(by=['需求日期', "作业单号", '工序编码'])  # 排序
            df_sorted.reset_index(drop=True, inplace=True)
            df_sorted.to_excel(operation_gantt_writer, sheet_name="工序明细", index=True)
            operation_gantt_writer._save()

            # 重新打开: 设置列宽和对齐
            excel_path = operation_gantt_writer
            wb = load_workbook(excel_path)
            ws = wb.active  # 获取第一个工作表

            # 遍历所有列，设置宽度为20
            for col in ws.columns:
                max_column = col[0].column_letter  # 获取列的字母编号
                ws.column_dimensions[max_column].width = 20  # 设置列宽为20

                for cell in col:
                    cell.alignment = Alignment(horizontal='left')  # 设置左对齐

            # 保存修改后的 Excel 文件
            wb.save(excel_path)

    def _clear(self):
        """
        清除所有任务和资源的状态
        :return:
        """
        to_removed = []
        for task in self.tasks:
            if task.uncleared_quantity == 0:
                to_removed.append(task)
            for op in task.craft_paths:
                op.assigned_start_time = None
                op.assigned_end_time = None
                op.assigned_resource = None
        for resource in self.resources:
            resource.operation_list.clear()

        for task in to_removed:
            self.tasks.remove(task)

    def tasks_sort_by_OPT(self):
        """
        使用deap的简单排序，已经证明不可行
        """

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("indices", random.sample, range(len(self.tasks)), len(self.tasks))  # 随机生成一个排列
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self._objective)
        toolbox.register("mate", tools.cxOnePoint)  # 有序交叉 cxOrdered
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)  # 变异操作 mutShuffleIndexes
        toolbox.register("select", tools.selTournament, tournsize=3)  # selTournament

        def optimize_permutation_sequence(n_generations=100, population_size=100, cxpb=0.1, mutpb=0.1):
            # 初始化种群
            population = toolbox.population(n=population_size)

            # 遗传算法主循环
            for generation in range(n_generations):
                offspring = toolbox.select(population, len(population))
                offspring = list(map(toolbox.clone, offspring))

                # 应用交叉
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < cxpb:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # 应用变异
                for mutant in offspring:
                    if random.random() < mutpb:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

                # 评估适应度
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # 更新种群
                population[:] = offspring

                # 输出当前代的最优适应度
                fits = [ind.fitness.values[0] for ind in population]
                print(f"Generation {generation}: Min fitness: {min(fits):.2f}, Avg fitness: {sum(fits) / len(population):.2f}")

            # 返回最佳个体和其适应度
            best_individual = tools.selBest(population, 1)[0]
            return best_individual, best_individual.fitness.values[0]

        best_seq, best_fitness = optimize_permutation_sequence()
        print("优化后的序列：", best_seq)
        print("优化后的适应度：", best_fitness)

    def _objective(self, seq: list[int]):
        self.tasks = [self.tasks[i] for i in seq]
        self._schedule(ResourceSelectionStrategy.GREEDY)  # 默认贪心
        efficient, delay = self.cal_kpi()
        return sum(delay.values()),

    def _scheduling_one_task(self, task: Task, _stragty: ResourceSelectionStrategy):
        """工序接续关系全部采用SSEE,task之间使用ES"""
        constraint = None
        if task.type == TaskType.PARENT and task.children_task:
            constraint = max([op.assigned_end_time for child in task.children_task for op in child.craft_paths])

        for i, job in enumerate(task.craft_paths):
            if _stragty == ResourceSelectionStrategy.RANDOM:
                selected_resource: Resource = random.choice(job.available_resource)
            else:
                selected_resource: Resource = min(job.available_resource, key=lambda r: r.local_time)

            selected_resource.operation_list.append(job)

            if i == 0:
                if constraint is None:
                    job.assigned_start_time = selected_resource.local_time
                else:
                    # parent的开始时间 TODO:检查parent 开始是否满足约束，由于毛坯加工非常快，这里暂定假设满足
                    job.assigned_start_time = max(constraint, selected_resource.local_time)
            else:
                job.assigned_start_time = max(task.craft_paths[i - 1].assigned_start_time + timedelta(hours=1), selected_resource.local_time)

            job.assigned_end_time = job.assigned_start_time + timedelta(minutes=(task.uncleared_quantity * job.minutes_per_bear) / job.quantity_per_bear)

            # 若不满足SSEE,同时修正开始和结束时间
            if i > 0 and job.assigned_end_time < task.craft_paths[i - 1].assigned_end_time:
                job.assigned_start_time = task.craft_paths[i - 1].assigned_end_time + (task.craft_paths[i - 1].assigned_end_time - job.assigned_end_time)
                job.assigned_end_time = job.assigned_start_time + timedelta(minutes=(task.uncleared_quantity * job.minutes_per_bear) / job.quantity_per_bear)

            job.assigned_resource = selected_resource

            # 更新资源时间
            selected_resource.local_time = job.assigned_end_time
            assert job.assigned_start_time and job.assigned_resource, "Start time or resource is not assigned"


if __name__ == '__main__':
    solver = Scheduler(resources=resources, tasks=tasks)

    # 不同模式下的kpi对比
    resource_selecter = [ResourceSelectionStrategy.GREEDY, ResourceSelectionStrategy.RANDOM]
    task_sorter = [TaskSortStrategy.CR, TaskSortStrategy.PLANENDDATE]

    # for first_mode in resource_selecter:
    #     for second_mode in task_sorter:
    #         print(f"资源选择策略: {first_mode}, 当前任务排序策略: {second_mode}")
    #         solver.based_schedule(select_strategy=first_mode, sort_stragety=second_mode)
    #         solver.out_put(display=False,output_file=False)

    solver.based_schedule(select_strategy=ResourceSelectionStrategy.GREEDY, sort_stragety=TaskSortStrategy.PLANENDDATE)
    solver.out_put(display=False, output_file=False, path="output/")

    # 输出资源利用率
    # efficicent, delay = solver.cal_kpi()
    # for key,val in efficicent.items():
    #     print(f"资源{key.name}的利用率: {val}")
