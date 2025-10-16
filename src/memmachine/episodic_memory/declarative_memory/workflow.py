import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Self


class Workflow:
    def __init__(
        self,
        executable: Callable[..., Awaitable],
        subworkflows: list[Self] = [],
        callback: Callable[..., Awaitable] | None = None,
    ):
        """
        Initialize a Workflow.

        Args:
            executable (Callable[..., Awaitable]):
                An asynchronous callable
                that performs the main operation of the workflow.
            subworkflows (list[Workflow], optional):
                A list of subworkflows to execute
                on the result of the main operation (default: []).
            callback (Callable[..., Awaitable], optional):
                An asynchronous callable that processes
                the results of the main operation
                and subworkflows (default: None).
        """
        self._executable = executable
        self._subworkflows = subworkflows
        self._callback = callback

    async def execute(self, arguments: Any) -> Any:
        """
        Execute the workflow with the provided arguments.

        Args:
            arguments (Any): Arguments to pass to the executable.

        Returns:
            Any:
                The result of the workflow execution,
                potentially processed by the callback if provided.
        """
        execution_result = await self._executable(arguments)

        subworkflow_results = await asyncio.gather(
            *[
                subworkflow.execute(execution_result)
                for subworkflow in self._subworkflows
            ]
        )

        if self._callback is not None:
            if subworkflow_results:
                return await self._callback(execution_result, subworkflow_results)
            else:
                return await self._callback(execution_result)

        return execution_result
