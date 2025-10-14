import functools
from collections.abc import Awaitable, Callable
from typing import Any, Self

def build_ingestion_workflow(
    config: dict[str, Any],
) -> Workflow:
    return Workflow(
        executable=functools.partial(
            DeclarativeMemory._assemble_episode_cluster,
            config["related_episode_postulator"],
        ),
        subworkflows=[
            build_derivative_derivation_workflow(derivative_derivation_workflow)
            for derivative_derivation_workflow in config[
                "derivative_derivation_workflows"
            ]
        ],
        callback=self._process_episode_cluster_assembly,
    )

def build_derivative_derivation_workflow(
    config: dict[str, Any],
) -> Workflow:
    return Workflow(
        executable=functools.partial(
            DeclarativeMemory._derive_derivatives,
            config["derivative_deriver"],
        ),
        subworkflows=[
            build_derivative_mutation_workflow(derivative_mutation_workflow)
            for derivative_mutation_workflow in config[
                "derivative_mutation_workflows"
            ]
        ],
        callback=self._process_derivative_derivation,
    )

def build_derivative_mutation_workflow(
    config: dict[str, Any],
) -> Workflow:
    return Workflow(
        executable=functools.partial(
            DeclarativeMemory._mutate_derivatives,
            config["derivative_mutator"],
        ),
        callback=self._process_derivative_mutation,
    )


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
