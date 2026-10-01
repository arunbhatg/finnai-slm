from data.sms_prompt import SMS_SYSTEM, chatml, gold_json, sms_user_prompt


def example(
    *,
    example_id: str,
    split_key: str,
    bank: str,
    template_id: str,
    task: str,
    sms: str | None = None,
    sender: str | None = None,
    gold: dict | None = None,
    system: str | None = None,
    user: str | None = None,
    assistant: str | None = None,
    must_ground: list[str] | None = None,
) -> dict:
    if task == "sms":
        user = sms_user_prompt(sms or "", sender or "")
        assistant = gold_json(gold)
        system = SMS_SYSTEM
    assert system and user and assistant is not None
    row = {
        "id": example_id,
        "split_key": split_key,
        "bank": bank,
        "template_id": template_id,
        "task": task,
        **chatml(system, user, assistant),
    }
    if sms is not None:
        row["sms"] = sms
        row["sender"] = sender
        row["gold"] = gold or {}
    if must_ground:
        row["must_ground"] = must_ground
    return row
