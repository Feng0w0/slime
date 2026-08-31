import ctypes
import logging
from pathlib import Path

import torch

from slime.utils.common import is_npu
if is_npu():
    import megatron_adaptor

try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    if is_npu():
        deep_ep_opapi = (
            Path(deep_ep.__file__).resolve().parent
            / "vendors/hwcomputing/op_api/lib/libcust_opapi.so"
        )
        if not deep_ep_opapi.exists():
            raise ImportError(f"Cannot find DeepEP custom operator library: {deep_ep_opapi}")
        # DeepEP looks up custom aclnn entry points through RTLD_DEFAULT. Its
        # extension only links libopapi, so load the packaged custom API globally.
        _deep_ep_opapi_handle = ctypes.CDLL(str(deep_ep_opapi), mode=ctypes.RTLD_GLOBAL)

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        tms_impl = torch_memory_saver._impl
        if tms_impl is None:
            return old_init(self, *args, **kwargs)

        cdll = tms_impl._binary_wrapper.cdll
        original_interesting_region = cdll.tms_get_interesting_region()
        cdll.tms_set_interesting_region(False)
        try:
            old_init(self, *args, **kwargs)
            # DeepEP owns persistent buffers and may initialize them on its
            # internal streams. Make their lifetime independent of the TMS
            # disabled region before restoring allocation tracking.
            torch.cuda.synchronize()
        finally:
            cdll.tms_set_interesting_region(original_interesting_region)

    deep_ep.Buffer.__init__ = new_init

except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")

logging.getLogger("megatron").setLevel(logging.WARNING)

from . import megatron_patch  # noqa: F401, E402
