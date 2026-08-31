import os
import slime.utils.external_utils.command_utils as U
HF_WEIGHTS_ROOT = os.environ["SLIME_TEST_HF_WEIGHTS_ROOT"]
DATASETS_ROOT = os.environ["SLIME_TEST_DATASETS_ROOT"]
MEGATRON_WEIGHTS_ROOT = os.environ["SLIME_TEST_MEGATRON_WEIGHTS_ROOT"]

ENABLE_EVAL = bool(int(os.environ.get("SLIME_TEST_ENABLE_EVAL", "1")))

MODEL_NAME = "Moonlight-16B-A3B-Instruct"
MODEL_TYPE = "moonlight"
NUM_GPUS = 16

def execute():
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
        "--rm-type math "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--global-batch-size 16 "
    )

    eval_args = (
        f"{'--eval-interval 20 ' if ENABLE_EVAL else ''}"
        f"--eval-prompt-data aime {DATASETS_ROOT}/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 4096 "
        "--eval-top-k 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 2 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
        "--use-distributed-optimizer "
    )

    grpo_args = (
        "--advantage-estimator gspo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 4 "
        "--rollout-num-gpus 8 "
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-disable-cuda-graph "
        "--sglang-max-running-requests 512 "
    )
    sglang_args += "--sglang-device npu "

    ci_args = "--ci-test "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--use-flash-attn "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )

if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
