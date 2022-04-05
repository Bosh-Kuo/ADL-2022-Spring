import argparse
from transformers import SchedulerType

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task")
    # Data
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default='./data/multiple_choice_train.json', help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default='./data/multiple_choice_valid.json', help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default='./data/multiple_choice_valid.json', help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--output_file", type=str, default='./data/QA_test.json', help="A csv or a json file containing the predicted test data."
    )
    parser.add_argument(
        "--context_file", type=str, default='./data/context.json', help="A csv or a json file containing the context data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=384,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="If passed, go through the trian process",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="If passed, go through the validate process",
    )
    parser.add_argument(
        "--do_test",
        action="store_true",
        help="If passed, go through the test process",
    )

    # Model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-chinese",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        # required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float,
                        default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
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
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default="./multiple_choice",
                        help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42,
                        help="A seed for reproducible training.")

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )

    args = parser.parse_args()
    return args