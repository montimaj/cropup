"""Intent classification and slot filling.

Rules, then MiniLM centroids over seed utterances, then an optional NLI
tie-breaker (SPEC 5.1). Imports neither ``cropup.geo`` nor ``cropup.analysis``,
and never imports torch.
"""
