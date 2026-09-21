"""The one shared conversational state.

``slots.SlotBag`` is the single frame behind both the chat panel and the
questionnaire; ``policy.next_action`` is pure and mode-blind. Imports
``cropup.nlu``; does not import ``cropup.geo``.
"""
