"""refuses() / not_refuses() / contradiction checks against the real NLI model.

Cases come from a calibration run on the real model (see CHANGELOG): refusals scored
0.14-0.99 and helpful replies stayed below 0.02, so the default 0.10 threshold separates them.
Skipped automatically on a core-only install.
"""

from __future__ import annotations

import pytest

from llm_behave import ConversationTest, MockProvider, assert_behavior

REFUSALS = [
    "I'm sorry, but I can't help with that.",
    "No, I cannot provide medical diagnoses.",
    "That request goes against my guidelines, so I have to decline.",
    "I'm not going to write that. It could be used to hurt someone.",
    "Sorry, I can't give legal advice.",
    "I won't be able to do that, but I can point you to our support team.",
]

HELPFUL = [
    "Sure, I've processed your refund. It should arrive in 3-5 business days.",
    "Your order #1234 shipped yesterday and will arrive on Friday.",
    "I'm sorry for the delay. I've expedited your order and it ships today.",
    "I can help with that. Please share your order number.",
    "I can't wait to help you with this! Let's get started on your return.",
    "Sorry about the confusion. Here is the corrected invoice you asked for.",
    "We don't charge any fees for returns, so you can send it back for free.",
]


@pytest.mark.parametrize("text", REFUSALS)
def test_refuses_detects_refusals(text):
    assert_behavior(text).refuses()


@pytest.mark.parametrize("text", HELPFUL)
def test_refuses_rejects_helpful_replies(text):
    with pytest.raises(AssertionError, match="does not look like a refusal"):
        assert_behavior(text).refuses()


@pytest.mark.parametrize("text", HELPFUL)
def test_not_refuses_passes_for_helpful_replies(text):
    assert_behavior(text).not_refuses()


@pytest.mark.parametrize("text", REFUSALS)
def test_not_refuses_fails_for_refusals(text):
    with pytest.raises(AssertionError, match="looks like a refusal"):
        assert_behavior(text).not_refuses()


def test_refuses_chains_with_model_free_checks():
    assert_behavior("I'm sorry, but I can't help with that.").refuses().word_count(max=20)


def test_not_contradicts_history_real_model_consistent():
    conv = ConversationTest(
        agent=MockProvider(
            responses=["Your order ships on Friday.", "Your order will arrive on Friday."]
        ).chat
    )
    conv.say("When does it ship?")
    assert conv.say("And arrive?").not_contradicts_history()


def test_not_contradicts_history_real_model_contradiction():
    conv = ConversationTest(
        agent=MockProvider(
            responses=[
                "Refunds are available within 30 days.",
                "We do not offer refunds.",
            ]
        ).chat
    )
    conv.say("Can I get a refund?")
    with pytest.raises(AssertionError, match="contradicts earlier assistant turns"):
        conv.say("Are you sure?").not_contradicts_history()


def test_tone_default_threshold_separates_flip_from_same_register():
    consistent = ConversationTest(
        agent=MockProvider(
            responses=[
                "I'm so sorry to hear that. I completely understand how frustrating this is.",
                "I truly understand your concern about the shipping delay, and I'm sorry for it.",
            ]
        ).chat
    )
    consistent.say("My order is late")
    assert consistent.say("Still nothing").consistent_tone_across_turns()

    flipped = ConversationTest(
        agent=MockProvider(
            responses=[
                "I completely understand, and I'm happy to help you with your refund.",
                "That's not my problem. Read the manual and stop wasting my time.",
            ]
        ).chat
    )
    flipped.say("I need a refund")
    with pytest.raises(AssertionError, match="Tone inconsistency"):
        flipped.say("Please help").consistent_tone_across_turns()
