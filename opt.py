import numpy as np
from openpyxl.styles.builtins import title

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.visualization.scatter import Scatter

from TypeDefine import ResourceSelectionStrategy, TaskSortStrategy
from scheduler import Scheduler
from LoadData import resources, tasks


class MyProblem(ElementwiseProblem):
    def __init__(self, solver: Scheduler):
        self.scheduler = solver
        self.original_order = {}
        for i, task in enumerate(self.scheduler.tasks):
            self.original_order[i] = task
        xl: np.ndarray = np.zeros(len(self.scheduler.tasks))
        xu: np.ndarray = np.ones(len(self.scheduler.tasks))
        super().__init__(n_var=len(self.scheduler.tasks), n_obj=2, n_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        idx = np.argsort(x)
        self.scheduler.tasks = [self.scheduler.tasks[i] for i in idx]
        self.scheduler.based_schedule(select_strategy=ResourceSelectionStrategy.GREEDY, sort_stragety=None)
        efficient, delay = self.scheduler.cal_kpi()
        out["F"] = [sum(delay.values()), - sum(efficient.values()) / len(efficient)]

        # 恢复现场
        self.scheduler.tasks = [val for key, val in self.original_order.items()]


if __name__ == "__main__":
    scheduler = Scheduler(resources=resources, tasks=tasks)
    problem = MyProblem(scheduler)

    algorithm = NSGA2(pop_size=100)
    res = minimize(
        problem,
        algorithm,
        ("n_gen", 100),
        verbose=10,
        seed=1,
    )

    scheduler.out_put(False, True, path="opt_out/")

    plot = Scatter(title="Optimization Results Pareto Front", labels=["MO Delivery Delay", "Average Equipment Load Rate"], )
    plot.add(res.F, edgecolor="blue", facecolor="none")
    plot.show()
