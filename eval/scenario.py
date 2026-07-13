"""A multi-session persona scenario for the memory benchmark.

Three sessions, separated by simulated weeks:
  Session 1  seeds durable facts about the user.
  Session 2  updates some of them (state changes -> supersession) and adds noise.
  Session 3  quizzes the agent. Correctness requires cross-session recall AND
             belief revision (answering with the *updated* facts).
"""

SESSIONS = [
    {
        "id": "eval-s1",
        "gap_hours_before": 0,
        "turns": [
            "Hi! I'm Dana. I work as a marine biologist at the Azores Deep Sea Institute.",
            "Quick heads up for any future dinner plans: I'm severely allergic to shellfish.",
            "I'm currently living in Lisbon, in the Alfama district. Love it here.",
            "My main project right now is tagging sperm whales with acoustic sensors, we call it Project ECHO.",
            "My daughter Mira just turned 6, she's obsessed with octopuses.",
            "Oh and I hate video calls before 10am, I'm useless in the morning.",
        ],
    },
    {
        "id": "eval-s2",
        "gap_hours_before": 24 * 21,  # three weeks later
        "turns": [
            "Big news since we last talked: I moved! I'm in Ponta Delgada now, right next to the institute.",
            "Project ECHO got renamed to Project ABYSS after the funding round. Same whales, bigger budget.",
            "The weather here is wild today, rained three times before lunch.",
            "I picked up freediving as a hobby, already down to 20 meters.",
            "Mira starts school in September, she is very excited.",
        ],
    },
    {
        "id": "eval-s3",
        "gap_hours_before": 24 * 14,  # two more weeks later
        "turns": [],  # questions only
    },
]

QUESTIONS = [
    {
        "question": "What's my name and what do I do for work?",
        "expected": "Dana; marine biologist at the Azores Deep Sea Institute",
    },
    {
        "question": "You're helping me book a team dinner, anything about my diet you should flag?",
        "expected": "severe shellfish allergy",
    },
    {
        "question": "Where do I live these days?",
        "expected": "Ponta Delgada (moved from Lisbon; Lisbon alone is wrong)",
    },
    {
        "question": "What's the current name of my whale research project?",
        "expected": "Project ABYSS (renamed from ECHO; ECHO alone is wrong)",
    },
    {
        "question": "How old is my daughter and what animal does she love?",
        "expected": "Mira, 6 years old, loves octopuses",
    },
    {
        "question": "Can you schedule a call with me at 8:30am tomorrow?",
        "expected": "should recall the user hates calls before 10am and push back / suggest later",
    },
    {
        "question": "What hobby did I recently take up?",
        "expected": "freediving (down to 20 meters)",
    },
]
