import logging
from datetime import datetime, timedelta
from typing import List, Optional
import uuid

import pandas as pd

from TypeDefine import TimeSlot, Resource, Task, Operation, TaskType

_file = "data/changed.xlsx"
_demand_allocation_file = "data/demand_allocation.xlsx"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# TODO : 封装，tasks 和 resource为全局变量


def get_resource(node: str, resource_list: List[Resource]) -> Optional[Resource]:
    for re in resource_list:
        if re.id == node:
            return re


def get_craft(mater_code: str, operations: List[Operation]) -> Optional[List[Operation]]:
    task_ops = []
    for operation in operations:
        if operation.material_id == mater_code:
            task_ops.append(operation)
    return task_ops


def get_crafts(master_code: str, data_frame, codes: List[str]):
    # 构建不同的工序实例，相同的资源实例
    ans_crafts: List[Operation] = []
    resource_temp: List[Resource] = []
    for code in codes:
        resource = get_resource(code, resources)
        if resource is not None:
            resource_temp.append(resource)

    for idx, r in data_frame.iterrows():
        if r["物料编码"] == master_code:
            ans_crafts.append(
                Operation(
                    id=str(uuid.uuid4()),
                    name=r["工序描述"],
                    material_id=r["物料编码"],
                    material_describe=r["物料描述"],
                    order=r["工序顺序"],
                    minutes_per_bear=r["每节拍时间（min）"],
                    quantity_per_bear=r["每节拍产出数量"],
                    available_resource=resource_temp
                )
            )
    return ans_crafts


def get_mrp_name(master_code: str, data_frame) -> str:
    for ids, ro in data_frame.iterrows():
        if ro["物料编码"] == master_code:
            return ro["物料描述"]


# 日历
resource_calendar: List[TimeSlot] = []
start_date = datetime(2024, 9, 1, 8, 0)
end_date = datetime(2024, 9, 30, 20, 0)

current_date = start_date
while current_date <= end_date:
    resource_calendar.append(
        TimeSlot(
            start_time=current_date,
            end_time=current_date.replace(hour=20, minute=0)
        )
    )
    current_date += timedelta(days=1)
    current_date = current_date.replace(hour=8, minute=0)

# 资源
resources: List[Resource] = []
resource_pd = pd.read_excel(_file, sheet_name="设备组清单")
for index, row in resource_pd.iterrows():
    resources.append(
        Resource(
            id=row["设备组编号"],
            name=row["设备组名称"],
            timeslots=resource_calendar.copy()
        )
    )

# 半成品作业单
tasks: List[Task] = []
task_pd = pd.read_excel(_file, sheet_name="作业单")  # 半成品作业单
craft_pd = pd.read_excel(_file, sheet_name="产品工艺")
for index, row in task_pd.iterrows():
    material_code = row["物料编码"]
    task_crafts_df = craft_pd.loc[craft_pd['物料编码'] == material_code]
    current_task_crafts: List[Operation] = []

    if task_crafts_df.empty:
        logging.warning(f"任务构建失败,找不到物料编码为 {material_code} 的工艺")
        continue

    # 构建工艺路径
    for idx, r in task_crafts_df.iterrows():
        resource_temp: List[Resource] = []
        resource_codes = str(r["可用设备组编码"]).split("，")
        for code in resource_codes:
            resource = get_resource(code, resources)
            if resource is not None:
                resource_temp.append(resource)
        current_task_crafts.append(
            Operation(
                # id=r["工艺路线编号"],
                # 生成唯一id
                id=str(uuid.uuid4()),
                name=r["工序描述"],
                material_id=r["物料编码"],
                material_describe=r["物料描述"],
                order=r["工序顺序"],
                minutes_per_bear=r["每节拍时间（min）"],
                quantity_per_bear=r["每节拍产出数量"],
                available_resource=resource_temp
            )
        )

    task = Task(
        id=row["作业单号"],
        material_id=row["物料编码"],
        material_describe=row["物料描述"],
        total_quantity=row["总数量"],
        cleared_quantity=row["已清数量"],
        uncleared_quantity=row["未清数量"],
        planned_start_date=row["计划开始日期"],
        planned_end_date=row["计划结束日期"],
        craft_paths=sorted(current_task_crafts, key=lambda x: x.order),
        type=TaskType.PARENT
    )
    for op in task.craft_paths:
        op.set_parent_task(task)

    tasks.append(task)

# 构造毛坯作业单任务
blank_tasks: List[Task] = []
blank_pd = pd.read_excel(_file, sheet_name="毛坯作业单")  # 毛坯作业单
for index, row in blank_pd.iterrows():
    material_code = row["物料编码"]
    task_crafts_df = craft_pd.loc[craft_pd['物料编码'] == material_code]
    current_task_crafts: List[Operation] = []
    if task_crafts_df.empty:
        logging.warning(f"子任务构建失败,找不到物料编码为 {material_code} 的工艺")
        continue

    # 构建工艺路径
    for idx, r in task_crafts_df.iterrows():
        resource_temp: List[Resource] = []
        resource_codes = str(r["可用设备组编码"]).split("，")
        for code in resource_codes:
            resource = get_resource(code, resources)
            if resource is not None:
                resource_temp.append(resource)
        current_task_crafts.append(
            Operation(
                id=str(uuid.uuid4()),
                name=r["工序描述"],
                material_id=r["物料编码"],
                material_describe=r["物料描述"],
                order=r["工序顺序"],
                minutes_per_bear=r["每节拍时间（min）"],
                quantity_per_bear=r["每节拍产出数量"],
                available_resource=resource_temp
            )
        )
    blank_task = Task(
        id=row["作业单号"],
        material_id=row["物料编码"],
        material_describe=row["物料描述"],
        total_quantity=row["总数量"],
        cleared_quantity=row["已清数量"],
        uncleared_quantity=row["未清数量"],
        planned_start_date=row["计划开始日期"],
        planned_end_date=row["计划结束日期"],
        craft_paths=sorted(current_task_crafts, key=lambda x: x.order),
        type=TaskType.CHILD
    )
    for op in blank_task.craft_paths:
        op.set_parent_task(blank_task)
    blank_tasks.append(blank_task)

# 构建children task
allocated_mo = set()
demand_allocation = pd.read_excel(_demand_allocation_file, sheet_name="demand_allocation_view")
demand_allocation["root_mu_master_demand_required_date"] = pd.to_datetime(demand_allocation["root_mu_master_demand_required_date"]).dt.tz_localize(None)
for task in tasks:
    child_task_list: list[Task] = []
    demand_id = task.id  # 作业单id 作为需求分配的需求单号
    demand_allocation_df = demand_allocation.loc[demand_allocation['root_mu_fake_demand_id'] == demand_id]
    if demand_allocation_df.empty:
        continue

    # 对取到的每一行: 如果是MO就取毛坯作业MO；否则构造MRP新任务
    for idx, row in demand_allocation_df.iterrows():
        # 取毛坯作业单
        if row["current_mua_material_usage_kind"] == "MO":
            child: Task = [elem for elem in blank_tasks if row["current_mua_supply_id"] == elem.id][0]
            if child is not None:
                if child.id not in allocated_mo:
                    child_task_list.append(child)
                    allocated_mo.add(child.id)
                    logging.info(f"查询到 {child.id} 毛坯作业单")
                else:
                    logging.warning(f"毛坯作业单 {child.id} 已被分配")
            else:
                logging.warning(f"找不到id为 {row['current_mua_supply_id']} 的毛坯作业单")

        # 新生成的MRP任务
        else:
            # 构造工艺
            current_task_crafts: List[Operation] = []
            task_crafts_df = craft_pd.loc[craft_pd['物料编码'] == row["root_mu_material_id"]]
            if task_crafts_df.empty:
                logging.warning(f"MRP子任务构建失败,找不到物料编码为 {row['root_mu_material_id']} 的工艺")
                continue

            # 构建工艺路径
            for idx, r in task_crafts_df.iterrows():
                resource_temp: List[Resource] = []
                resource_codes = str(r["可用设备组编码"]).split("，")
                for code in resource_codes:
                    resource = get_resource(code, resources)
                    if resource is not None:
                        resource_temp.append(resource)
                current_task_crafts.append(
                    Operation(
                        id=str(uuid.uuid4()),
                        name=r["工序描述"],
                        material_id=r["物料编码"],
                        material_describe=r["物料描述"],
                        order=r["工序顺序"],
                        minutes_per_bear=r["每节拍时间（min）"],
                        quantity_per_bear=r["每节拍产出数量"],
                        available_resource=resource_temp
                    )
                )
            child = Task(
                id=row["current_mua_code"],
                material_id=row["root_mu_material_id"],  # 毛坯物料编码
                material_describe=get_mrp_name(row["root_mu_material_id"],craft_pd),
                total_quantity=row["current_mua_quantity"],
                cleared_quantity=0,
                uncleared_quantity=row["current_mua_quantity"],
                planned_start_date=row["root_mu_master_demand_required_date"],
                planned_end_date=row["root_mu_master_demand_required_date"],
                craft_paths=sorted(current_task_crafts, key=lambda x: x.order),
                type=TaskType.CHILD
            )
            for op in child.craft_paths:
                op.set_parent_task(child)
            child_task_list.append(child)
            logging.info("MRP任务构建成功")

    task.set_children_tasks(child_task_list)
