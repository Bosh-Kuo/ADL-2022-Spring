import argparse
from transformers import SchedulerType

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a summarization task"
    )
    parser.add_argument(
        "--eval_rouge",
        action="store_true",
        help="If passed, calculate rouge score and store save as json file",
    )
    parser.add_argument(
        "--test_file", type=str, default="./data/public.jsonl", help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data/output_submission2.jsonl", help="Where to store the final model."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=256,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    # for t5 series model "summarize: "
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=True, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=64,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    # Text Generation parameters
    parser.add_argument(
        "--generation_method",
        type=str, 
        default="",
        help="What kind of NLG method",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,  # 5
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--do_sample",
        type=bool,
        default=False,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,  # 50
        help="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,  # 1.0
        help="If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,  # 1.0
        help=" The value used to module the next token probabilities.",
    )

    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="./mt5_small_model",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=24,  # 24 / 64
        help="Batch size (per device) for the tseting dataloader.",
    )
    parser.add_argument(
    "--learning_rate",
    type=float,
    default=5e-5,
    help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float,
                        default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts",
                "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--seed", type=int, default=31, help="A seed for reproducible training."
    )
    args = parser.parse_args()

    return args
