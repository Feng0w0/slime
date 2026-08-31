
try:
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    _original_build_conversion_tasks = MegatronModelBridge.build_conversion_tasks

    def _patched_build_conversion_tasks(self, hf_pretrained, megatron_model):
        """
        Invoke the original build_conversion_tasks and filter out any actual None tasks.

        The original implementation might return List[None | WeightConversionTask].
        We consolidate it here into List[WeightConversionTask] to avoid errors
        when accessing None.task.xxx later.
        """
        tasks = _original_build_conversion_tasks(self, hf_pretrained, megatron_model)

        if tasks is None:
            return []

        filtered = [t for t in tasks if t is not None]

        return filtered

    MegatronModelBridge.build_conversion_tasks = _patched_build_conversion_tasks

except ImportError:
    pass
