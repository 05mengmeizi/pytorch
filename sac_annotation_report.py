"""
SAC Annotation Report

Captures SAC (Selective Activation Checkpointing) annotations on joint forward
graphs for real HuggingFace models. Produces a deterministic text report that
can be diff'd across code changes to detect annotation regressions.

Usage:
    python sac_annotation_report.py                # print to stdout
    python sac_annotation_report.py -o report.txt  # write to file
    python sac_annotation_report.py --models gpt2  # specific model only
"""

import argparse
import functools
import sys

import torch
import torch._dynamo
from functorch.compile import min_cut_rematerialization_partition, nop
from torch._dynamo.backends.common import aot_autograd
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts


MATMUL_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
}


def sac_policy(ctx, func, *args, **kwargs):
    if func in MATMUL_OPS:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def make_capture_partition(all_regions):
    """Returns a partition_fn that captures SAC annotations before partitioning."""

    def capture_partition(joint_gm, joint_args, **kwargs):
        lines = []
        for node in joint_gm.graph.nodes:
            if node.op == "call_function":
                recompute = node.meta.get("recompute", None)
                if recompute is not None:
                    lines.append(f"{node.name}: {node.target} -> {recompute.name}")
        if lines:
            all_regions.append(lines)
        return min_cut_rematerialization_partition(joint_gm, joint_args, **kwargs)

    return capture_partition


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _make_gpt2():
    from transformers import GPT2Config
    from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

    config = GPT2Config(
        n_layer=2, n_head=2, n_embd=64, vocab_size=100, n_positions=32,
        attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config).train()
    input_ids = torch.randint(0, 100, (1, 16))
    return model, dict(input_ids=input_ids, labels=input_ids), "GPT2LMHeadModel (n_layer=2)"


def _make_bert():
    from transformers import BertConfig
    from transformers.models.bert.modeling_bert import BertForMaskedLM

    config = BertConfig(
        num_hidden_layers=2, hidden_size=64, num_attention_heads=2,
        intermediate_size=128, vocab_size=100, max_position_embeddings=32,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )
    model = BertForMaskedLM(config).train()
    input_ids = torch.randint(0, 100, (1, 16))
    return model, dict(input_ids=input_ids, labels=input_ids), "BertForMaskedLM (n_layer=2)"


MODEL_REGISTRY = {
    "gpt2": _make_gpt2,
    "bert": _make_bert,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_model(name, make_fn, out):
    model, inputs, desc = make_fn()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
        "use_reentrant": False,
        "context_fn": functools.partial(create_selective_checkpoint_contexts, sac_policy),
    })

    all_regions = []
    partition_fn = make_capture_partition(all_regions)
    backend = aot_autograd(fw_compiler=nop, bw_compiler=nop, partition_fn=partition_fn)
    torch._dynamo.reset()

    compiled = torch.compile(model, backend=backend)
    result = compiled(**inputs)
    loss = result.loss if hasattr(result, "loss") else result
    if isinstance(loss, torch.Tensor):
        loss.sum().backward()

    out.write(f"=== {desc} ===\n")
    for i, lines in enumerate(all_regions):
        out.write(f"--- Checkpoint Region {i} ---\n")
        for line in lines:
            out.write(line + "\n")
    out.write("\n")


def main():
    parser = argparse.ArgumentParser(description="SAC Annotation Report")
    parser.add_argument("-o", "--output", help="Write report to file instead of stdout")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_REGISTRY), default=list(MODEL_REGISTRY),
                        help="Models to include (default: all)")
    args = parser.parse_args()

    out = open(args.output, "w") if args.output else sys.stdout

    out.write("# SAC Annotation Report\n")
    out.write(f"# Policy: save matmuls (mm, addmm, bmm)\n")
    out.write("\n")

    for name in args.models:
        run_model(name, MODEL_REGISTRY[name], out)

    if args.output:
        out.close()
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
