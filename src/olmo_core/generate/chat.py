"""
CLI script for chatbot-style text generation using TransformerGenerationModule.

Example usage:
    # Basic usage
    python -m olmo_core.generate.chat path/to/checkpoint --max-new-tokens 512 --temperature 0.7

    # With custom chat template
    python -m olmo_core.generate.chat path/to/checkpoint --chat-template "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    # Greedy decoding (deterministic)
    python -m olmo_core.generate.chat path/to/checkpoint --do-sample False

    # Example usage from a Mac M4 with a public checkpoint (`--attention-backend torch --no-use-cache` to avoid flash attn dependency)
    python -m olmo_core.generate.chat https://olmo-checkpoints.org/ai2-llm/Olmo-3-1025-7B/stage2/step47684/ --attention-backend torch --no-use-cache
"""

import argparse
import json
import logging
import sys
from typing import Optional

from cached_path import cached_path
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from transformers import AutoTokenizer

from olmo_core.aliases import PathOrStr
from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.generate import GenerationConfig, TransformerGenerationModule
from olmo_core.io import join_path, normalize_path
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.utils import get_default_device, log_or_print, prepare_cli_environment

log = logging.getLogger(__name__)
console = Console()

# Just concatenate the messages, no system prompt, no assistant prompt.
DEFAULT_CHAT_TEMPLATE = """{% for message in messages %}{{ message['content'] }}{% endfor %}"""


def render_assistant_message(message: str) -> Panel:
    """Render an assistant message as a chat bubble."""
    text = Text(message, style="default")
    return Panel(
        text,
        title="[bold magenta]Assistant[/bold magenta]",
        title_align="left",
        border_style="magenta",
        padding=(0, 1),
        width=None,
    )


def render_system_message(message: str) -> Panel:
    """Render a system prompt message with a distinct style."""
    text = Text(message, style="dim")
    return Panel(
        text,
        title="[bold yellow]System[/bold yellow]",
        title_align="left",
        border_style="yellow",
        padding=(0, 1),
        width=None,
    )


def render_tokenizer_info(
    tokenizer_config: TokenizerConfig, tokenizer, chat_template: str
) -> Panel:
    """Render tokenizer configuration details."""
    # Olmo-core TokenizerConfig info
    left_lines = []
    left_lines.append("Olmo-core TokenizerConfig:")
    left_lines.append(f"  • Identifier: {tokenizer_config.identifier or 'N/A'}")
    left_lines.append(f"  • Vocab size: {tokenizer_config.vocab_size:,}")
    left_lines.append(f"  • EOS token ID: {tokenizer_config.eos_token_id}")
    left_lines.append(f"  • Pad token ID: {tokenizer_config.pad_token_id}")
    left_lines.append(f"  • BOS token ID: {tokenizer_config.bos_token_id}")

    # HuggingFace tokenizer info
    right_lines = []
    right_lines.append("HuggingFace Tokenizer:")
    right_lines.append(f"  • Vocab size: {tokenizer.vocab_size:,}")

    model_max_length = getattr(tokenizer, "model_max_length", None)
    right_lines.append(f"  • Model max length: {model_max_length}")

    # Special tokens
    all_special_tokens = getattr(tokenizer, "all_special_tokens", None)
    if all_special_tokens:
        special_tokens_str = ", ".join(all_special_tokens[:5])
        if len(all_special_tokens) > 5:
            special_tokens_str += f" ... ({len(all_special_tokens)} total)"
        right_lines.append(f"  • Special tokens: {special_tokens_str}")
    else:
        right_lines.append("  • Special tokens: N/A")

    # Token IDs and strings
    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    right_lines.append(f"  • EOS token: {eos_token} (ID: {eos_token_id})")
    pad_token = getattr(tokenizer, "pad_token", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    right_lines.append(f"  • Pad token: {pad_token} (ID: {pad_token_id})")
    bos_token = getattr(tokenizer, "bos_token", None)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    right_lines.append(f"  • BOS token: {bos_token} (ID: {bos_token_id})")

    left_text = Text("\n".join(left_lines))
    right_text = Text("\n".join(right_lines))
    columns = Columns([left_text, right_text], equal=True, expand=True)

    chat_template_panel = Panel(
        chat_template,
        title="[bold cyan]Chat Template[/bold cyan]",
        border_style="dim",
        padding=(0, 1),
    )
    combined_content = Group(columns, "", chat_template_panel)
    return Panel(
        combined_content, title="[bold green]Tokenizer Info[/bold green]", border_style="green"
    )


def render_generation_config(generation_config: GenerationConfig) -> Panel:
    """Render generation configuration details."""
    left_items = []
    right_items = []

    left_items.append(f"• Max new tokens: {generation_config.max_new_tokens}")
    right_items.append(f"• Max length: {generation_config.max_length}")

    if generation_config.do_sample:
        left_items.append("• Sampling: enabled")
        left_items.append(f"• Temperature: {generation_config.temperature}")
        top_k_str = "unlimited" if generation_config.top_k == -1 else str(generation_config.top_k)
        left_items.append(f"• Top-k: {top_k_str}")
        left_items.append(f"• Top-p: {generation_config.top_p}")
    else:
        left_items.append("• Sampling: disabled (greedy)")

    cache_status = "enabled" if generation_config.use_cache else "disabled"
    right_items.append(f"• KV cache: {cache_status}")

    left_text = Text("\n".join(left_items))
    right_text = Text("\n".join(right_items))
    columns = Columns([left_text, right_text], equal=True, expand=True)
    content = Group("[bold]Generation Parameters:[/bold]", columns)
    return Panel(content, title="[bold blue]Generation Config[/bold blue]", border_style="blue")


def load_tokenizer(tokenizer_config: TokenizerConfig):
    """Load the actual tokenizer from HF Hub using the tokenizer config identifier."""
    if tokenizer_config.identifier is None:
        raise ValueError("Tokenizer config has no identifier. Cannot load tokenizer. ")
    return AutoTokenizer.from_pretrained(tokenizer_config.identifier)


def load_tokenizer_config_from_checkpoint(checkpoint_dir: PathOrStr) -> Optional[TokenizerConfig]:
    """Load tokenizer config from checkpoint's config.json."""
    checkpoint_dir = normalize_path(checkpoint_dir)
    config_path = join_path(checkpoint_dir, "config.json")
    try:
        with cached_path(config_path).open() as f:
            config_dict = json.load(f)
        return TokenizerConfig.from_dict(config_dict["dataset"]["tokenizer"])
    except (KeyError, FileNotFoundError) as e:
        log_or_print(
            log,
            f"Could not load tokenizer config from checkpoint: {e}",
            level=logging.WARNING,
        )
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Chatbot-style text generation using OLMo TransformerGenerationModule",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m olmo_core.generate.chat /path/to/checkpoint

  # With custom generation parameters
  python -m olmo_core.generate.chat /path/to/checkpoint \\
      --max-new-tokens 512 --temperature 0.7 --top-p 0.9

  # Greedy decoding (deterministic)
  python -m olmo_core.generate.chat /path/to/checkpoint \\
      --max-new-tokens 256 --do-sample False

  # Override attention backend
  python -m olmo_core.generate.chat /path/to/checkpoint \\
      --attention-backend torch
        """,
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate (default: 1024)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum total length (prompt + generation). Overrides --max-new-tokens if set.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling (default: 1.0). Set to 0.0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling. -1 means no top-k filtering (default: -1)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.7,
        help="Top-p (nucleus) sampling (default: 0.7)",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sampling (default: True). Set --no-do-sample for greedy decoding.",
    )
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use KV cache for faster generation (default: True). Set --no-use-cache for no cache.",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        choices=[backend.value for backend in AttentionBackendName],
        help=f"Override attention backend (default: auto-chosen based on --use-cache). Options: {[backend.value for backend in AttentionBackendName]}",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=[dtype.value for dtype in DType],
        help=f"Data type for model parameters (default: bfloat16). Options: {[dtype.value for dtype in DType]}",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt to prepend to all conversations",
    )
    parser.add_argument(
        "--show-special-tokens",
        action="store_true",
        default=False,
        help="Show special tokens (e.g., <|endoftext|>, <|endofsequence|>) in generated text (default: False)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=DEFAULT_CHAT_TEMPLATE,
        help="Jinja2 chat template string. The default is to just concatenate the messages, no system prompt, no assistant prompt.",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level (default: INFO)",
    )

    args = parser.parse_args()

    device = get_default_device()
    log.info(f"Using device: {device}")

    # Load tokenizer config and tokenizer
    tokenizer_config = load_tokenizer_config_from_checkpoint(args.checkpoint_dir)
    if tokenizer_config is None:
        log.warning("Could not load tokenizer config from checkpoint. Falling back to dolma2.")
        tokenizer_config = TokenizerConfig.dolma2()

    try:
        tokenizer = load_tokenizer(tokenizer_config)
    except Exception as e:
        log.error(
            f"Failed to load tokenizer from identifier '{tokenizer_config.identifier}': {e}",
            exc_info=True,
        )
        raise e

    # Display tokenizer info
    console.print(render_tokenizer_info(tokenizer_config, tokenizer, args.chat_template))
    console.print()

    generation_config = GenerationConfig(
        pad_token_id=tokenizer_config.pad_token_id,
        eos_token_id=tokenizer_config.eos_token_id,
        max_new_tokens=args.max_new_tokens if args.max_length is None else None,
        max_length=args.max_length,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_cache=args.use_cache,
    )

    attention_backend = None
    if args.attention_backend:
        attention_backend = AttentionBackendName(args.attention_backend)
        log.info(f"Using explicitly specified attention backend: {attention_backend}")

    dtype = DType(args.dtype)
    log.info(f"Using dtype: {dtype}")

    log.info("Loading model from checkpoint...")
    try:
        generation_module = TransformerGenerationModule.from_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            generation_config=generation_config,
            device=device,
            dtype=dtype,
            attention_backend=attention_backend,
        )
    except Exception as e:
        log.error(f"Failed to load checkpoint: {e}", exc_info=True)
        sys.exit(1)

    console.print("[bold green]✓ Model loaded successfully![/bold green]")

    # Display generation config
    console.print(render_generation_config(generation_config))

    # Welcome message
    welcome_text = Text()
    welcome_text.append("Chatbot ready! ", style="bold green")
    welcome_text.append("Type your message and press Enter.\n\n", style="dim")
    welcome_text.append("Commands:\n", style="bold")
    welcome_text.append("  /quit or /exit ", style="cyan")
    welcome_text.append("- Exit the chatbot\n", style="dim")
    welcome_text.append("  /clear ", style="cyan")
    welcome_text.append("- Clear conversation history\n", style="dim")
    welcome_text.append("  /help ", style="cyan")
    welcome_text.append("- Show this help message", style="dim")

    console.print(Panel(welcome_text, title="[bold red]Welcome[/bold red]", border_style="red"))
    console.print()

    conversation_history: list[dict[str, str]] = []
    if args.system_prompt:
        conversation_history.append({"role": "system", "content": args.system_prompt})

    # Display system prompt if provided
    if args.system_prompt:
        console.print(render_system_message(args.system_prompt))
        console.print()

    try:
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
                console.print()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[bold yellow]Goodbye![/bold yellow]")
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["/quit", "/exit"]:
                console.print("[bold yellow]Goodbye![/bold yellow]")
                break
            elif user_input.lower() == "/clear":
                conversation_history = []
                if args.system_prompt:
                    conversation_history.append({"role": "system", "content": args.system_prompt})
                    console.print()
                    console.print(render_system_message(args.system_prompt))
                    console.print()
                console.print("[bold green]✓ Conversation history cleared.[/bold green]\n")
                continue
            elif user_input.lower() == "/help":
                help_text = Text()
                help_text.append("Commands:\n", style="bold")
                help_text.append("  /quit or /exit ", style="cyan")
                help_text.append("- Exit the chatbot\n", style="dim")
                help_text.append("  /clear ", style="cyan")
                help_text.append("- Clear conversation history\n", style="dim")
                help_text.append("  /help ", style="cyan")
                help_text.append("- Show this help message", style="dim")
                console.print(
                    Panel(help_text, title="[bold blue]Help[/bold blue]", border_style="blue")
                )
                console.print()
                continue

            # Add user message to conversation history
            conversation_history.append({"role": "user", "content": user_input})

            # Build prompt using chat template
            prompt = tokenizer.apply_chat_template(
                conversation_history,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=args.chat_template,
            )

            try:
                with console.status("[dim]Generating response...", spinner="dots"):
                    input_ids = tokenizer.encode(prompt, return_tensors="pt")
                    generated_ids, _, _ = generation_module.generate_batch(
                        input_ids, completions_only=True, log_timing=False
                    )
                    response_text = tokenizer.decode(
                        generated_ids[0], skip_special_tokens=not args.show_special_tokens
                    )

                console.print()
                console.print(render_assistant_message(response_text))
                conversation_history.append({"role": "assistant", "content": response_text})

            except Exception as e:
                log_or_print(log, f"Generation error: {e}", level=logging.ERROR)
                error_panel = Panel(
                    Text(f"Error: {e}", style="red"),
                    title="[bold red]Error[/bold red]",
                    border_style="red",
                    padding=(0, 1),
                )
                console.print()
                console.print(error_panel)
                conversation_history.pop()

    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        log.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    prepare_cli_environment()
    main()
