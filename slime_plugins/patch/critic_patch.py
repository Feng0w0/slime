from typing import (
    List,
    Mapping,
    TypeVar,
    Union,
)
import torch
from megatron.core.transformer.module import MegatronModule
HFPreTrained = TypeVar("HFPreTrained")
MegatronModel = TypeVar("MegatronModel", bound=MegatronModule)


def load_weights_hf_to_megatron_wrapper(
        self, hf_pretrained: HFPreTrained, megatron_model: Union[MegatronModel, List[MegatronModel]]
    ) -> List[MegatronModel]:
    """Load HuggingFace weights into Megatron models.

    This method orchestrates the complete weight loading process from HuggingFace
    format to Megatron's distributed format. It builds a conversion task and
    executes it with proper progress tracking and error handling.

    The actual weight transformations and distribution are delegated to the
    appropriate MegatronParamMapping instances based on the state mappings.

    Args:
        hf_pretrained (HFPreTrained): HuggingFace model or state source containing the
            weights to load.
        megatron_model (Union[MegatronModel, List[MegatronModel]]): Megatron model instance
            or list of model instances (one per virtual pipeline stage).

    Returns:
        List[MegatronModel]: The input megatron_model as a list with loaded weights.

    Process:
    1. Build a task mapping each Megatron parameter to its source
    2. For each parameter in the task:
        - Fetch source weights from HuggingFace state
        - Apply format transformation via the param mapping
        - Distribute to appropriate TP/PP ranks
        - Copy into the Megatron parameter

    Example:
        .. code-block:: python

            hf_model = PreTrainedCausalLM.from_pretrained("gpt2")
            megatron_model = create_megatron_model()  # Single model or list
            bridge.load_weights_hf_to_megatron(hf_model, megatron_model)

    Note:
        Progress is shown only on rank 0 to avoid cluttered output in
        distributed environments.

    Raises:
        ValueError: If hf_pretrained doesn't have state attribute or if weight shapes don't match.
        AttributeError: If required HF weights are missing.
    """
    if not isinstance(megatron_model, list):
        megatron_model = [megatron_model]

    hf_to_megatron_tasks = self.build_conversion_tasks(hf_pretrained, megatron_model)
    hf_state_dict: Mapping[str, torch.Tensor] = hf_pretrained.state if hasattr(hf_pretrained, "state") else {}

    description = f"Loading from {hf_pretrained.model_name_or_path}"
    for task in self._with_progress_tracking(hf_to_megatron_tasks, description):
        # None means megatron module not on current rank, skip if this task is not going to happen
        if task.megatron_module is None:
            continue
        # 1) Fetch source tensor(s) from HF state dict
        hf_weights = self.maybe_modify_loaded_hf_weight(task.mapping.hf_param, hf_state_dict)

        # 2) Delegate conversion & distribution to the bridge
        converted_weights = task.mapping.hf_to_megatron(hf_weights, task.megatron_module)

        # 3) Copy into Megatron param if this rank received a shard
        if converted_weights is not None:
            if task.param_weight is None:
                raise ValueError("param_weight is required for HF->Megatron conversion")

            if converted_weights.shape != task.param_weight.shape and task.param_name == 'language_model.output_layer.weight':
                continue

            # Check shape compatibility before copying
            if converted_weights.shape != task.param_weight.shape:
                raise ValueError(
                    f"Shape mismatch for megatron param {task.mapping.megatron_param}:\n"
                    f"  Expected shape: {task.param_weight.shape}\n"
                    f"  Got shape: {converted_weights.shape}\n"
                    f"  Bridge type: {type(task.mapping).__name__}\n"
                    f"  HF mapping: {task.mapping.hf_param}"
                )
            task.param_weight.data.copy_(converted_weights)

    self._broadcast_shared_embeddings(megatron_model)
    return megatron_model