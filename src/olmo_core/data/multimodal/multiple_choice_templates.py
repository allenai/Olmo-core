"""Multiple-choice question templating (ported from mm_olmo)."""

from __future__ import annotations

import string

__all__ = ["template_mc_question"]


def build_option_name_instructions(option_type):
    assert option_type is not None
    return [
        f"Answer the question with only the option {option_type}.",
        f"Answer with only the option {option_type}.",
        f"Answer with only {option_type} for the correct option.",
        f"Respond with only the answer {option_type}",
        f"Return only the option {option_type}.",
        f"Your answer (option {option_type} only):",
        f"Select one option and provide only its {option_type}.",
        f"Choose the correct answer and return only its {option_type}.",
        f"Reply with just the option {option_type}, nothing else.",
        f"output only the {option_type} of your chosen answer to the question.",
        f"respond with the option {option_type} only.",
        f"Provide your answer as a single option {option_type} only.",
        f"Return the option {option_type} alone.",
        f"State only the option {option_type}.",
        f"please state the correct answer {option_type}.",
        f"Give only the option {option_type} in your response.",
    ]


def build_option_instructions(option_type):
    out = [
        "Which option best answers the question?",
        "Return one of the answer options",
        "Return the best answer option.",
        "Answer with one of the options.",
        "Answer the question using one of the options.",
        "Respond with only the correct option.",
        "Generate nothing but the correct option",
        "Your response should be one of the options.",
        "Your response should be one of the options",
        "state the correct option.",
        "Choose one of the options.",
        "Return one of the provided options.",
        "Please directly give the best option.",
        "Return the best option directly.",
    ]
    if option_type is not None:
        out += [
            f"Respond with only the option {option_type} and its full text.",
            f"Give me the option {option_type} and its text.",
        ]
    return out


def build_option_templates(q, opts, instr):
    return [
        f"Question: {q}\n\nOptions:\n{opts}\n\n{instr}",
        f"[QUESTION]\n{q}\n\n[OPTIONS]\n{opts}\n\n[INSTRUCTION]\n{instr}",
        f"{q}\n\n{opts}\n\n{instr}",
        f"{instr}\n\n{q}\n\n{opts}",
        f"{instr}\n{q}\n{opts}",
        f"Q: {q}\n\n{opts}\n\n{instr}",
        f"{q}\n\n---\n{opts}\n---\n\n{instr}",
        f"{q}\n\n{opts}\n\n{instr}",
        f"{q}\n{opts}\n{instr}",
        f"1. Question:\n{q}\n\n2. Options:\n{opts}\n\n3. Instructions:\n{instr}",
        f"{q} {opts}. {instr}",
        f"Select the correct answer:\n\n{q}\n\n{opts}\n\n{instr}",
        f"{q}\n\n\n{opts}\n\n\n{instr}",
        f"Question: {q}\n\n{opts}\n\nResponse: {instr}",
        f"*** Question ***\n{q}\n\n*** Options ***\n{opts}\n\n*** Notes ***\n{instr}",
    ]


def build_option_name_templates(question: str, options: str, option_type):
    return [
        f"{question}\n{options}\nReturn only the {option_type} of the best answer option",
        f"Answer this question by naming the {option_type} of one of the provided options:\n{question}\n{options}",
        f"Look at the options, then return the {option_type} of the option that best answers the question.\nQuestion: {question}\nOptions: {options}",
        f"Question: {question} Options: {options} Answer {option_type}:",
        f"Answer the question by selecting an answer {option_type}\nQuestion: {question}\nOptions: {options}",
        f"{question}\n{options}\nReturn only the {option_type} of the correct answer",
        f'Help me answer this question: "{question}", by stating the correct option {option_type}\n{options}',
        f"Question: {question}\n\nChoose one {option_type}:\n{options}",
        f"{question}\n{options}\nReturn the right answer {option_type} and nothing else.",
    ]


def build_instruction_templates(question: str, options: str, option_type):
    return [
        f"{question}\n{options}\nWhat option best answers the question?",
        f"{question}\n{options}\nReturn the best answer option",
        f"{question}\n{options}\nPick the best answer.",
        f"{question}\n{options}\nPlease return the best answer option.",
        f"{question} Select an answer option from:\n{options}",
        f"{question}\nSelect the best answer from:\n{options}\n\n",
        f"{question}\nReturn the correct answer without any explanation:\n{options}\n",
        f"{question}\nPick the right answer and return it:\n{options}\n",
        f"Question: {question} Options: {options}. Correct Answer Option:",
        f"Question: {question}\nOptions: {options}\nCorrect Answer Option:",
        f"Answer the question by selecting an answer option\nQuestion: {question}\nOptions: {options}",
        f"Answer the question by selecting an option\nQuestion: {question}\nOptions:\n{options}",
        f"{question}\n{options}\nReturn only the correct answer",
        f'Help me answer this question: "{question}", by stating the correct option from:\n{options}.',
        f'For the question "{question}", return the best option from:\n{options}.',
        f"Question: {question}\n\nChoose one:\n{options}",
        f"Question: {question}\nChoose from:\n{options}",
        f'Question: "{question}"\nChoose from:\n{options}',
        f"Question: {question}. Options: {options}. Best Options:\n",
    ]


COMMON_MARKERS = ["%s. ", "%s) ", "%s: ", "(%s) "]
WEIRD_MARKERS = [
    "%s; ",
    "%s ",
    "%s.  ",
    "%s)",
    "%s=",
    "%s\t",
    "%s    ",
    "{%s} ",
    "%s => ",
    "[%s] ",
    "<%s> ",
]
WEIRD_MARKERS = sorted(
    set(WEIRD_MARKERS + [x.strip() for x in COMMON_MARKERS + WEIRD_MARKERS if x.strip() != "%s"])
)


def template_mc_question(
    question,
    options,
    rng,
    unlabelled=False,
    p_inline=0.1,
    p_use_instruction_template=0.5,
    p_label_only_output=0.25,
    p_label_options=0.8,
):
    options = [str(x) for x in options]

    if unlabelled:
        label_only_response = False
        label_options = False
    else:
        label_only_response = rng.random() < p_label_only_output
        label_options = label_only_response or rng.random() < p_label_options

    inline = rng.random() < p_inline
    use_instruction_template = rng.random() < p_use_instruction_template

    if label_options:
        if rng.random() > 0.2:
            marker = COMMON_MARKERS[rng.randint(0, len(COMMON_MARKERS))]
        else:
            marker = WEIRD_MARKERS[rng.randint(0, len(WEIRD_MARKERS))]

        names = [
            (string.ascii_uppercase, "letter"),
            (string.ascii_lowercase, "letter"),
            ([str(i) for i in range(1, 21)], "number"),
            (["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"], "numeral"),
        ]
        option_names, option_name_type = names[rng.randint(0, len(names))]
        option_identifiers = [marker % n for n in option_names]
        option_text = [f"{opt_id}{opt}" for opt_id, opt in zip(option_identifiers, options)]
        if label_only_response:
            outputs = option_names[: len(options)]
        else:
            outputs = option_text
    else:
        option_name_type = None
        option_text = options
        option_names = options
        outputs = options
        marker = ""

    inline_seps = [
        x
        for x in [";", " ||| "]
        if x not in marker.strip() and not any(opt.endswith(x) for opt in options)
    ]
    if label_options:
        inline_seps.append("")
    if not inline_seps:
        inline = False
    if inline:
        sep = inline_seps[rng.randint(0, len(inline_seps))]
        tmp = [
            x if (ix == (len(option_text) - 1) or x.endswith(sep)) else x + sep
            for ix, x in enumerate(option_text)
        ]
        option_string = " ".join(tmp)
    else:
        if not label_options and rng.random() < 0.5:
            seps = ["•", "◦", "▪", "▫", "‣", "⁃", "∙", "○", "●", "□"]
            sep = f"\n{seps[rng.randint(0, len(seps))]} "
            option_string = sep.lstrip() + sep.join(option_text)
        else:
            sep = "\n"
            option_string = sep.join(option_text)

    if label_options:
        r = rng.random()
        if r < 0.6:
            option_name_type = option_name_type
        elif r < 0.8:
            option_name_type = "identifier"
        else:
            option_name_type = "label"

    if use_instruction_template:
        if label_only_response:
            instructions = build_option_name_instructions(option_name_type)
        else:
            instructions = build_option_instructions(option_name_type)
        instr = instructions[rng.randint(0, len(instructions))]
        templates = build_option_templates(question, option_string, instr)
    elif label_only_response:
        templates = build_option_name_templates(question, option_string, option_name_type)
    else:
        templates = build_instruction_templates(question, option_string, option_name_type)

    input_text = templates[rng.randint(0, len(templates))]
    return input_text, option_names[: len(option_text)], outputs[: len(option_text)]
