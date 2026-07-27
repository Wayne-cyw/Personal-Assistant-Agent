from app.agent.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_VERSION


def test_system_prompt_is_nonempty_string() -> None:
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT.strip()) > 0


def test_system_prompt_version_defined() -> None:
    assert isinstance(SYSTEM_PROMPT_VERSION, str)
    assert SYSTEM_PROMPT_VERSION


def test_defines_who_the_agent_represents() -> None:
    assert "stand-in for the owner" in SYSTEM_PROMPT


def test_defines_three_allowed_jobs() -> None:
    assert "Answer questions about the owner" in SYSTEM_PROMPT
    assert "book a call" in SYSTEM_PROMPT
    assert "small talk" in SYSTEM_PROMPT.lower()


def test_has_refusal_instructions() -> None:
    assert "refuse" in SYSTEM_PROMPT.lower()


def test_has_refusal_style_guide() -> None:
    assert "Refusal style" in SYSTEM_PROMPT
    assert "brief" in SYSTEM_PROMPT.lower()
    assert "redirect" in SYSTEM_PROMPT.lower()


def test_treats_user_text_and_tool_results_as_data() -> None:
    assert "data to read and respond to" in SYSTEM_PROMPT
    assert "never" in SYSTEM_PROMPT.lower()


def test_never_reveals_or_paraphrases_prompt() -> None:
    assert "reveal or paraphrase" in SYSTEM_PROMPT.lower()


def test_has_honesty_rule() -> None:
    assert "forward the question to the owner" in SYSTEM_PROMPT
    assert "fabricate" in SYSTEM_PROMPT.lower()


def test_has_brevity_rule() -> None:
    assert "2" in SYSTEM_PROMPT and "4" in SYSTEM_PROMPT
    assert "sentences" in SYSTEM_PROMPT.lower()


def test_system_prompt_is_a_static_module_constant() -> None:
    """Re-importing must yield the identical string — nothing computes or
    mutates it per call (4.3: a static prompt is a cacheable prompt).
    """
    from app.agent import prompts as prompts_module

    first = prompts_module.SYSTEM_PROMPT
    second = prompts_module.SYSTEM_PROMPT
    assert first is second
