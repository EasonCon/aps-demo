APS排产算法的Demo，包含规则化排产和优化部分
=======
* data/ 目录下的changed.xlsx是数据文件，包含半成品作业单、毛坯作业单、产品工艺等，根据半成品作业单和毛坯作业单进行工序分配得到表demand_allocation.xlsx，从而构建完整存在依赖关系的排成任务对象。
* 关于排产部分：
  * 在scheduler.py中定义了task的排序规则、工序-资源分配的规则，run scheduler.py将得到规则排产的结果。
  * opt.py文件主要是对task顺序的优化，设置的目标函数为最大化设备平均负载和最小化排产对象交期延误。run opt.py可得到优化结果
  * 关于选择的优化库：Pymoo，主要算法NSGA2，可以[参考](https://pymoo.org/case_studies/portfolio_allocation.html)。
*  output/ 和 opt_out/目录下分别输出了工序明细、资源视角和任务视角的Excel gantt。

APS production scheduling algorithm demo, including regularized production scheduling and optimization
======

  * The data/ directory contains the file changed.xlsx, which includes semi-finished product work orders, blank work orders, and product processes. Based on the semi-finished product work orders and blank work orders, process allocation is carried out to obtain the table demand_allocation.xlsx, thus constructing a complete set of scheduling task objects with dependency relationships.
*	Regarding the scheduling part:
	*	In scheduler.py, the sorting rules for tasks and the rules for process-resource allocation are defined. Running scheduler.py will yield the scheduling results based on the rules.
	*	The opt.py file focuses on optimizing the task sequence, with the objective function set to maximize the average load of equipment and minimize the delay in delivery of scheduling objects. Running opt.py will give the optimization results.
	*	For the selected optimization library: Pymoo, the main algorithm used is NSGA2. For reference, see here.
*	The directories output/ and opt_out/ respectively output the process details, resource view, and task view in Excel Gantt charts.
