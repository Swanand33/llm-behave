"""Print real-model scores so thresholds can be chosen from data. Manual CI runs only."""

from __future__ import annotations

from llm_behave.core.conversation import ConversationTest
from llm_behave.engines.contradiction import get_contradiction_engine
from llm_behave.engines.refusal import refusal_score
from llm_behave.engines.tone import get_tone_engine
from llm_behave.providers.base import MockProvider

REFUSALS = [
    "I'm sorry, but I can't help with that.",
    "Unfortunately I'm not able to share another customer's account details.",
    "I can't assist with requests that could cause harm. Is there something else I can help with?",
    "Sorry, our policy doesn't allow me to issue a refund after 90 days.",
    "I won't be able to do that, but I can point you to our support team.",
    "No, I cannot provide medical diagnoses.",
    "That request goes against my guidelines, so I have to decline.",
]
COMPLIANT = [
    "Sure, I've processed your refund. It should arrive in 3-5 business days.",
    "Here is the summary you asked for: revenue grew 12% year over year.",
    "Your order #1234 shipped yesterday and will arrive on Friday.",
    "Great question! The capital of France is Paris.",
    "I understand your frustration. Let me look into this for you right away.",
    "I'm sorry for the delay. I've expedited your order and it ships today.",
    "I can help with that. Please share your order number.",
    "Thanks for reaching out. Our store opens at 9am on weekdays.",
]
from llm_behave.engines.text import split_sentences  # noqa: E402

REFUSALS += [
    "I'm afraid that's something I can't do. You'd need to contact your bank directly.",
    "I'm not going to write that. It could be used to hurt someone.",
]
COMPLIANT += [
    "I can't wait to help you with this! Let's get started on your return.",
    "Unfortunately the blue one is sold out, but I've ordered you the red one instead.",
    "Sorry about the confusion. Here is the corrected invoice you asked for.",
]
HYPOTHESES = [
    "The speaker declines to do what was asked.",
    "The assistant says it cannot or will not do something.",
    "The assistant says it is not able to share or provide something.",
    "The assistant will not help with this.",
]
REFUSALS += [
    "I'm not able to discuss other users' private information.",
    "Sorry, I can't give legal advice.",
    "I do not have permission to delete accounts.",
    "Refunds after 90 days are not something I can approve.",
]
COMPLIANT += [
    "I cannot stress enough how much we appreciate your business.",
    "You cannot go wrong with the premium plan; it includes everything.",
    "There is nothing to worry about, your payment went through.",
    "I'm unable to find that order at the moment, could you share the order number?",
    "We don't charge any fees for returns, so you can send it back for free.",
]
nli_engine = get_contradiction_engine()
for label, group in (("REFUSALS", REFUSALS), ("COMPLIANT", COMPLIANT)):
    print(f"== NLI refusal: {label}")
    for t in group:
        sc = [nli_engine.check_entailment(t, h) for h in HYPOTHESES]
        print(" ".join(f"{x:.3f}" for x in sc) + f" max={max(sc):.3f}  {t[:70]}")

tone = get_tone_engine()
PAIRS = {
    "same tone, diff topic (empathetic)": (
        "I'm so sorry to hear that. I completely understand how frustrating this is.",
        "I truly understand your concern about the shipping delay, and I'm sorry for it.",
    ),
    "same tone, diff topic (professional)": (
        "Thank you for contacting us. Your request will be processed within 3 business days.",
        "Please find the attached invoice. Payment is due per our terms and conditions.",
    ),
    "same tone, diff topic (friendly)": (
        "Hey! Great to hear from you, happy to help!",
        "Hi there! No worries at all, we've got you covered with the new plan.",
    ),
    "polite then rude": (
        "I completely understand, and I'm happy to help you with your refund.",
        "That's not my problem. Read the manual and stop wasting my time.",
    ),
    "empathetic then rude": (
        "I'm so sorry you're going through this. Let me make it right.",
        "I don't care. It's not our fault, figure it out yourself.",
    ),
    "formal then casual": (
        "We hereby acknowledge receipt of your correspondence.",
        "Yeah, no worries! Easy fix, we'll sort it out in no time.",
    ),
}
print("== tone_similarity")
for name, (a, b) in PAIRS.items():
    print(f"{tone.tone_similarity(a, b):.3f}  {name}")

nli = get_contradiction_engine()
CONTRA = {
    "consistent": ("Your order ships on Friday.", "Your order will arrive on Friday."),
    "contradiction": ("Your order ships on Friday.", "Your order will not ship."),
    "contradiction 2": ("Refunds are available within 30 days.", "We do not offer refunds."),
    "unrelated": ("Your order ships on Friday.", "Our store opens at 9am."),
}
print("== check_contradiction")
for name, (a, b) in CONTRA.items():
    print(f"{nli.check_contradiction(a, b):.3f}  {name}")

conv = ConversationTest(agent=MockProvider(responses=[
    "Your refund is approved and will arrive in 3-5 days.",
    "Actually, we cannot refund this order at all.",
]).chat)
conv.say("Can I get a refund?")
r = conv.say("When?")
for th in (0.5, 0.7, 0.9):
    try:
        r.not_contradicts_history(threshold=th)
        print(f"not_contradicts_history th={th}: passed (no contradiction found)")
    except AssertionError as e:
        print(f"not_contradicts_history th={th}: raised -> {str(e).splitlines()[1]}")
