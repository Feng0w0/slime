"""GLM-4.7-Flash colocated training test with single-node PD + Mooncake."""

import os
import tempfile

import yaml

import slime.utils.external_utils.command_utils as U
HF_WEIGHTS_ROOT = os.environ["SLIME_TEST_HF_WEIGHTS_ROOT"]
DATASETS_ROOT = os.environ["SLIME_TEST_DATASETS_ROOT"]
MEGATRON_WEIGHTS_ROOT = os.environ["SLIME_TEST_MEGATRON_WEIGHTS_ROOT"]

MODEL_REPO = "zai-org/GLM-4.7-Flash"
MODEL_NAME = "GLM-4.7-Flash"
MODEL_TYPE = "glm4.7-30B-A3B"
NUM_GPUS = 8

def write_sglang_config() -> str:
    config = {
        "sglang": [
            {
                "name": "default",
                "server_groups": [
                    {
                        "worker_type": "prefill",
                        "num_gpus": 4,
                        "num_gpus_per_engine": 4,
                        "overrides": {"disaggregation_transfer_backend": "ascend"},
                    },
                    {
                        "worker_type": "decode",
                        "num_gpus": 4,
                        "num_gpus_per_engine": 4,
                        "overrides": {"disaggregation_transfer_backend": "ascend"},
                    },
                ],
            }
        ]
    }
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="sglang_pd_mooncake_", delete=False)
    with f:
        yaml.safe_dump(config, f, sort_keys=False)
    return f.name

def execute():
    sglang_config = write_sglang_config()

    ckpt_args = (
        f"--hf-checkpoint {HF_WEIGHTS_ROOT}/{MODEL_NAME} "
        f"--ref-load {MEGATRON_WEIGHTS_ROOT}/{MODEL_NAME}_torch_dist "
    )
    rollout_args = (
        f"--prompt-data {DATASETS_ROOT}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 2 "
        "--rollout-max-response-len 512 "
        "--rollout-temperature 1.0 "
        "--rollout-top-p 0.95 "
        "--global-batch-size 8 "
    )
    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )
    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )
    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 2 "
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 4 "
        "--expert-tensor-parallel-size 1 "
        "--decoder-last-pipeline-num-layers 23 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
    )
    sglang_args = (
        "--rollout-num-gpus 8 "
        "--rollout-num-gpus-per-engine 4 "
        "--sglang-enable-dp-attention "
        "--sglang-dp-size 4 "
        "--sglang-enable-dp-lm-head "
        "--sglang-ep-size 4 "
        "--sglang-moe-a2a-backend ascend_fuseep "
        "--sglang-moe-dense-tp-size 1 "
        "--sglang-mem-fraction-static 0.45 "
        "--sglang-disable-cuda-graph "
        "--sglang-cuda-graph-max-bs 8 "
        "--sglang-max-running-requests 16 "
        "--sglang-disaggregation-transfer-backend ascend "
        "--sglang-speculative-algorithm EAGLE "
        "--sglang-speculative-num-steps 3 "
        "--sglang-speculative-eagle-topk 1 "
        "--sglang-speculative-num-draft-tokens 4 "
        "--sglang-speculative-moe-a2a-backend ascend_fuseep "
        "--sglang-watchdog-timeout 1200 "
        "--sglang-router-request-timeout-secs 1200 "
        "--sglang-enable-metrics "
        f"--sglang-config {sglang_config} "
    )
    sglang_args += "--sglang-device npu "
    misc_args = (
        "--ci-test "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
        "--moe-token-dispatcher-type alltoall "
    )
    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
    )
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={
            "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:13200",
            "ASCEND_MF_TRANSFER_PROTOCOL": "sdma",
        },
    )

if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
