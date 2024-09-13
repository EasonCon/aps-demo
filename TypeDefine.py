from __future__ import annotations

import datetime
import weakref
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


@dataclass
class TimeSlot:
    start_time: datetime
    end_time: datetime


@dataclass
class Resource:
    id: str
    name: str
    timeslots: List[TimeSlot] = None
    priority: int = 1

    # scheduler demand
    local_time: Optional[datetime.datetime] = None
    operation_list: list[Operation] = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)

    def __iter__(self):
        return iter(self.operation_list)


@dataclass
class Operation:
    id: str
    name: str
    material_id: str
    material_describe: str
    order: int
    minutes_per_bear: int
    quantity_per_bear: int

    available_resource: List[Resource]
    product_batch_size: int = 1
    lead_time: int = 1
    post_time: int = 1
    lead_time_occupied: int = 1
    post_time_occupied: int = 1

    # scheduler demand
    assigned_resource: Resource = field(default=None)
    assigned_start_time: datetime = None
    assigned_end_time: datetime = None
    _parent_task: Optional[weakref.ref] = field(default=None, repr=False, compare=False)  # 使用弱引用

    @property
    def parent_task(self) -> Optional[Task]:
        return self._parent_task() if self._parent_task is not None else None

    @parent_task.setter
    def parent_task(self, task: Optional[Task]):
        self._parent_task = weakref.ref(task) if task is not None else None

    def __hash__(self):
        return hash(self.id)

    def set_parent_task(self, task: Task):
        self.parent_task = task


@dataclass
class Task:
    id: str
    material_id: str
    material_describe: str
    total_quantity: int
    cleared_quantity: int
    uncleared_quantity: int
    craft_paths: List[Operation]
    type: TaskType
    planned_start_date: datetime = field(default=datetime)
    planned_end_date: datetime = field(default=datetime)

    # task关联
    children_task: List[Task] = field(default_factory=list)

    def set_craft_paths(self, craft_paths: List[Operation]):
        self.craft_paths = craft_paths

    @property
    def delay(self) -> float:
        duration: datetime.timedelta = max(
            max(
                [
                    op.assigned_end_time for op in self.craft_paths
                    if op.assigned_end_time is not None
                ]
            ) - self.planned_end_date,
            datetime.timedelta(days=0)
        )
        return duration.total_seconds() / 3600

    @property
    def cr(self):
        return (self.planned_end_date - datetime.datetime(2024, 9, 1)) / sum([self.uncleared_quantity * op.minutes_per_bear / op.quantity_per_bear for op in self.craft_paths])

    def __hash__(self):
        return hash(self.id)

    def set_children_tasks(self, children_task: List[Task]):
        self.children_task = children_task


class ResourceSelectionStrategy(Enum):
    GREEDY = "GREEDY"
    RANDOM = "RANDOM"


class TaskSortStrategy(Enum):
    CR = "CR"
    PLANENDDATE = "PLANENDDATE"
    OPT = "OPT"

class TaskType(Enum):
    PARENT = "PARENT"
    CHILD = "CHILD"