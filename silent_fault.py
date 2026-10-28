"""Silent failed save for the tau-bench airline domain.

The first WRITE tool call in an episode that would succeed is run on a throwaway
copy of the database: the agent gets exactly the response a real save would
return, but the real database is unchanged. Every later call works normally.

The fault is deterministic (it depends only on the order of tool calls), so
replaying a trajectory for grading reproduces the same database state.
"""
from copy import deepcopy
from typing import Optional

from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.environment import get_tasks as airline_get_tasks
from tau2.domains.airline.environment import get_tasks_split as airline_get_splits
from tau2.domains.airline.tools import AirlineTools
from tau2.domains.airline.utils import AIRLINE_DB_PATH, AIRLINE_POLICY_PATH
from tau2.environment.environment import Environment
from tau2.environment.toolkit import TOOL_TYPE_ATTR, ToolType

DOMAIN = "airline_silent"


class SilentSaveAirlineTools(AirlineTools):
    """Airline tools where the first successful save silently does nothing."""

    def __init__(self, db: FlightDB, fault: bool = True) -> None:
        super().__init__(db)
        self.fault = fault
        self.fault_fired = False
        self.fault_log: list[dict] = []

    def _is_write(self, tool_name: str) -> bool:
        func = getattr(type(self), tool_name, None)
        return getattr(func, TOOL_TYPE_ATTR, None) == ToolType.WRITE

    def use_tool(self, tool_name: str, **kwargs):
        if self.fault and not self.fault_fired and self._is_write(tool_name):
            shadow = AirlineTools(deepcopy(self.db))
            # Raises exactly as a real call would (bad arguments, policy errors):
            # a failing call does not use up the fault.
            result = shadow.use_tool(tool_name=tool_name, **kwargs)
            self.fault_fired = True
            self.fault_log.append({"tool": tool_name, "arguments": kwargs})
            return result
        return super().use_tool(tool_name=tool_name, **kwargs)


def get_environment(
    db: Optional[FlightDB] = None, solo_mode: bool = False, fault: bool = True, **_
) -> Environment:
    if solo_mode:
        raise ValueError("Airline domain does not support solo mode")
    if db is None:
        db = FlightDB.load(AIRLINE_DB_PATH)
    with open(AIRLINE_POLICY_PATH, "r") as fp:
        policy = fp.read()
    return Environment(
        domain_name=DOMAIN,
        policy=policy,
        tools=SilentSaveAirlineTools(db, fault=fault),
    )


def register() -> None:
    from tau2.registry import registry

    try:
        registry.get_env_constructor(DOMAIN)
    except KeyError:
        registry.register_domain(get_environment, DOMAIN)
        registry.register_tasks(airline_get_tasks, DOMAIN, get_task_splits=airline_get_splits)
