"""
translator.py
-------------
Pure text mapping layer for PawTalk.
Converts classifier/analyzer output into human-readable, playful messages.
No audio logic lives here — edit this file to change the app's personality.

Bark translation structure
--------------------------
Each mood has three style banks: "cute", "funny", "emotional".
Each bank holds several (headline, message) pairs so results vary across clips.
The selection index is derived from the confidence score so the same result
always picks the same message — no randomness, fully deterministic.

get_bark_translation() returns:
  selected   — the primary translation (chosen style)
  alternates — the other two styles, ready for a "show me another" toggle
"""

# ---------------------------------------------------------------------------
# Message banks
# Structure: mood → style → list of {headline, message, emoji} entries
# Add more entries to a list to increase variety. Do not add new keys.
# ---------------------------------------------------------------------------

_STYLES = ("cute", "funny", "emotional")

_BANKS: dict = {

    # =========================================================================
    "excited": {
        "emoji":    "🐕💨",
        "fun_fact": (
            "Dogs can reach up to 30 mph during zoomies (officially called FRAPs — "
            "Frenetic Random Activity Periods). Scientists believe it's pure joy "
            "with a side of chaos."
        ),
        "cute": [
            {
                "headline": "Pure happiness, loading... ✨",
                "message":  (
                    "Oh my goodness, oh my goodness, oh my GOODNESS — "
                    "everything is amazing and wonderful and the BEST right now! "
                    "Somebody grab a toy before this little heart explodes."
                ),
            },
            {
                "headline": "Maximum wiggles detected 🌟",
                "message":  (
                    "Today is the greatest day that has ever happened in the history "
                    "of days! The tail is going, the ears are up, and every single "
                    "thing is a gift. Possibly because you exist."
                ),
            },
            {
                "headline": "Joy level: uncontainable 💛",
                "message":  (
                    "Everything is good! The world is good! You are good! "
                    "This bark is basically a tiny happy symphony dedicated to "
                    "the simple fact that right now, life is perfect."
                ),
            },
        ],
        "funny": [
            {
                "headline": "ZOOMIES INCOMING — EVACUATE THE FURNITURE 🛋️💨",
                "message":  (
                    "This pup is running a full-system reboot. Current status: "
                    "100% frantic, 0% chill. ETA to first lap of the living room: "
                    "approximately now. Please secure loose objects."
                ),
            },
            {
                "headline": "The hype train has no brakes 🚂",
                "message":  (
                    "Scientists have confirmed: this dog contains more enthusiasm "
                    "per square inch than any known substance. Warning label reads: "
                    "'Do not agitate.' Too late. Already agitated. Extremely agitated."
                ),
            },
            {
                "headline": "ALERT: EXCITEMENT LEVELS EXCEEDING SAFE LIMITS ⚠️",
                "message":  (
                    "We regret to inform you that calm has left the building. "
                    "In its place: spinning, barking, and a tail that may achieve "
                    "liftoff. Please stand clear of the blast radius."
                ),
            },
        ],
        "emotional": [
            {
                "headline": "This is what pure joy sounds like.",
                "message":  (
                    "Right now, in this moment, nothing is wrong. There is no worry, "
                    "no hesitation — just the full, uncomplicated happiness of a dog "
                    "who loves their life. Hold onto this sound."
                ),
            },
            {
                "headline": "This bark is a love letter.",
                "message":  (
                    "Dogs don't save their joy for special occasions. They give it "
                    "freely, loudly, and with everything they have. This bark is "
                    "for you. It has always been for you."
                ),
            },
            {
                "headline": "They are so glad you're here.",
                "message":  (
                    "To your dog, you coming home is the best thing that happens "
                    "every single day. This sound is what that feeling looks like "
                    "from the inside. It never gets old for them."
                ),
            },
        ],
    },

    # =========================================================================
    "playful": {
        "emoji":    "🎾🐶",
        "fun_fact": (
            "Dogs use a 'play bow' — front legs down, butt up — as a universal signal "
            "that whatever comes next is just for fun. It's basically doggo for "
            "'no hard feelings, okay?'"
        ),
        "cute": [
            {
                "headline": "Excuse me, I have a proposal 🐾",
                "message":  (
                    "This bark is a formal invitation. The terms are simple: "
                    "you throw the thing, I bring it back, we repeat until one of us "
                    "gets tired (spoiler: it won't be me). Please RSVP immediately."
                ),
            },
            {
                "headline": "Someone found their happy place 🌈",
                "message":  (
                    "This is a bark of pure invitation — a 'hey, do you want to do "
                    "the fun thing? Because I very much want to do the fun thing.' "
                    "The answer is always yes. Obviously."
                ),
            },
            {
                "headline": "Adventure is waiting and so am I 🌟",
                "message":  (
                    "The toy is ready. The paws are ready. The enthusiasm is "
                    "so, so ready. This bark is the starting pistol for the "
                    "best game you've ever played. Ready? Go!"
                ),
            },
        ],
        "funny": [
            {
                "headline": "PLAY WITH ME OR I WILL STARE AT YOU FOREVER 👀",
                "message":  (
                    "This dog has already chosen a toy. It has been dropped at your "
                    "feet twice. This bark is the third notice. "
                    "There will not be a fourth — only the stare."
                ),
            },
            {
                "headline": "HR has filed a complaint on your behalf 📋",
                "message":  (
                    "Your dog would like to formally notify you that you have been "
                    "sitting still for too long and this is unacceptable. "
                    "Please report to the backyard immediately for mandatory fetch."
                ),
            },
            {
                "headline": "This is not a request. This is a summons. 📜",
                "message":  (
                    "The ball has been placed in front of you. Eye contact has been "
                    "established. The bark has been issued. At this point you are "
                    "legally obligated to throw it."
                ),
            },
        ],
        "emotional": [
            {
                "headline": "Play is how they say 'I trust you.'",
                "message":  (
                    "When a dog invites you to play, they're not just bored — "
                    "they're choosing you. Out of everyone and everything, right "
                    "now they want to share their joy with you specifically."
                ),
            },
            {
                "headline": "This moment won't last forever.",
                "message":  (
                    "One day the toys will stay on the floor longer. One day the bark "
                    "will come a little less often. Today is not that day. "
                    "Go play. It matters more than you think."
                ),
            },
            {
                "headline": "They just want to be with you.",
                "message":  (
                    "The game doesn't really matter to them. The fetch isn't about "
                    "the ball. It's about moving through the world together, "
                    "laughing together, being alive at the same time as you."
                ),
            },
        ],
    },

    # =========================================================================
    "alert": {
        "emoji":    "🚨🐶",
        "fun_fact": (
            "Dogs can hear sounds at four times the distance of humans and can detect "
            "frequencies between 40 Hz and 65,000 Hz. That rustling you can't hear? "
            "They definitely can."
        ),
        "cute": [
            {
                "headline": "Something interesting has occurred 👀",
                "message":  (
                    "A thing has been detected. The nature of the thing is unclear "
                    "but it is definitely there and it definitely requires "
                    "both ears pointed forward at maximum attention. Stand by."
                ),
            },
            {
                "headline": "Filing a report on suspicious activity 📋",
                "message":  (
                    "There is something out there and this very good watchdog "
                    "wants you to know about it. Investigation is underway. "
                    "The household is safe. Probably. Stay close just in case."
                ),
            },
            {
                "headline": "Chief of Security, reporting for duty 🐾",
                "message":  (
                    "Something has changed in the environment and a thorough "
                    "examination is in progress. Do not be alarmed. Your dog "
                    "has the situation handled and will file a full report shortly."
                ),
            },
        ],
        "funny": [
            {
                "headline": "INTRUDER ALERT (probably a leaf) 🍃",
                "message":  (
                    "Code Red has been declared. The threat level: unclear. "
                    "The cause: possibly a plastic bag, possibly a stranger, "
                    "possibly the same fence post that was also suspicious yesterday."
                ),
            },
            {
                "headline": "The squirrel has been noted 🐿️",
                "message":  (
                    "Intelligence has been gathered. A creature of unknown motive "
                    "has entered the perimeter. It had the audacity to just "
                    "sit there. This will not stand. Loud notices have been filed."
                ),
            },
            {
                "headline": "Breaking news: something happened somewhere 📰",
                "message":  (
                    "Sources close to the window confirm that a sound occurred "
                    "in the general vicinity of outside. All available personnel "
                    "(one dog) have been deployed. Updates as they develop."
                ),
            },
        ],
        "emotional": [
            {
                "headline": "They just want to keep you safe.",
                "message":  (
                    "Every alert bark is a promise. 'I'm watching. I noticed. "
                    "I will always tell you when something changes.' "
                    "They take this job very seriously, because you matter to them."
                ),
            },
            {
                "headline": "Vigilance is a form of love.",
                "message":  (
                    "Your dog positions themselves between you and the unknown "
                    "without being asked, every single time. This bark is "
                    "the sound of someone who would never let anything hurt you."
                ),
            },
            {
                "headline": "They noticed before you did.",
                "message":  (
                    "There's a whole world of sensation happening around you "
                    "that you'll never experience the way they do. This bark "
                    "is them bridging that gap — sharing what they hear, so you're never alone in it."
                ),
            },
        ],
    },

    # =========================================================================
    "anxious": {
        "emoji":    "🫂🐕",
        "fun_fact": (
            "Dogs can sense human anxiety through smell — they detect cortisol changes "
            "in our sweat. Your calm genuinely helps them calm down too."
        ),
        "cute": [
            {
                "headline": "Someone needs a little reassurance 🌸",
                "message":  (
                    "This bark is soft around the edges — a little unsure, a little "
                    "wobbly. Something feels uncertain right now, and a gentle "
                    "presence, a quiet voice, or a small treat would help enormously."
                ),
            },
            {
                "headline": "Could use a snuggle right about now 💛",
                "message":  (
                    "The world feels a little big at the moment. This pup just "
                    "needs to know you're here, you're calm, and everything "
                    "is going to be okay. A lap would be very welcome."
                ),
            },
            {
                "headline": "Requesting proximity to a trusted human 🐾",
                "message":  (
                    "Something is off and it's a little overwhelming. "
                    "Not a dangerous thing — just an uncertain one. "
                    "Your dog just needs you nearby until the feeling passes."
                ),
            },
        ],
        "funny": [
            {
                "headline": "The anxiety is loud today 📣",
                "message":  (
                    "Current threat assessment: unclear, possibly everything. "
                    "Dog status: not handling it, extremely not handling it. "
                    "Prescribed treatment: belly rubs, stat, no questions asked."
                ),
            },
            {
                "headline": "I have concerns and they are many 📋",
                "message":  (
                    "This dog has prepared a list of worries. It is very long. "
                    "Item one: the mailman. Item two: the mailman's feelings. "
                    "Item three: whether you have thought about the mailman lately."
                ),
            },
            {
                "headline": "Requesting immediate emotional support 🚨",
                "message":  (
                    "The situation: unclear. The feelings: very clear and very large. "
                    "The solution: you, here, now, with your hand on this head "
                    "until further notice. No, the other hand. Both hands."
                ),
            },
        ],
        "emotional": [
            {
                "headline": "Something feels wrong and they can't explain it.",
                "message":  (
                    "Dogs can't tell us what scares them. They can only ask for "
                    "help the only way they know how. When you stay close and "
                    "stay calm, you're giving them something no one else can."
                ),
            },
            {
                "headline": "They came to you because you're their safe place.",
                "message":  (
                    "Out of everywhere they could go, they came to you. "
                    "That says everything about what you mean to them. "
                    "Be soft with them right now. They're trying their best."
                ),
            },
            {
                "headline": "This bark is a question: are you still there?",
                "message":  (
                    "Sometimes anxious barking isn't about fear of something outside. "
                    "It's about the fear of being alone. Answer the question. "
                    "Go to them. Let them hear your voice."
                ),
            },
        ],
    },

    # =========================================================================
    "warning": {
        "emoji":    "🐕🔔",
        "fun_fact": (
            "Warning barks tend to be lower-pitched and slower than excited barks. "
            "Dogs modulate pitch deliberately — deeper tones signal 'I'm serious' in "
            "canine communication across breeds."
        ),
        "cute": [
            {
                "headline": "Excuse me, I need to make an announcement 📣",
                "message":  (
                    "This pup has officially detected something that requires "
                    "everyone's attention. They are doing their job extremely well "
                    "and would appreciate it if you acknowledged this immediately."
                ),
            },
            {
                "headline": "On duty and taking it very seriously 🐾",
                "message":  (
                    "A thing is happening out there and your very dedicated dog "
                    "wants to make sure you are fully informed. They have considered "
                    "the situation and concluded: this is worth a bark."
                ),
            },
            {
                "headline": "Official notice has been issued 📋",
                "message":  (
                    "After careful deliberation, your dog has determined that "
                    "a bark is warranted. This was not done lightly. "
                    "Please take appropriate notice. Thank you for your attention."
                ),
            },
        ],
        "funny": [
            {
                "headline": "I am giving you exactly one warning. 🔔",
                "message":  (
                    "This bark arrived with gravitas. It was not a question. "
                    "It was not a request. It was a statement of intent from "
                    "a dog who has decided that something out there has crossed a line."
                ),
            },
            {
                "headline": "TERMS AND CONDITIONS: do not test me 📜",
                "message":  (
                    "Whatever is happening outside has officially been put on notice. "
                    "Your dog has read the situation, assessed the threat, and issued "
                    "a formal communication. The squirrel received it."
                ),
            },
            {
                "headline": "I have filed a strongly-worded bark 📰",
                "message":  (
                    "Legal proceedings have been initiated against the suspicious noise. "
                    "Your dog is representing themselves. Their argument is airtight. "
                    "The fence has been warned. Court adjourned."
                ),
            },
        ],
        "emotional": [
            {
                "headline": "They take their responsibility seriously.",
                "message":  (
                    "This bark isn't panic — it's purpose. Your dog has made "
                    "a deliberate decision to speak up, to stand between you "
                    "and uncertainty, because that's what they believe their job is."
                ),
            },
            {
                "headline": "Loyalty sounds like this.",
                "message":  (
                    "A warning bark is a declaration: I am here, I am watching, "
                    "and I will always let you know. There is no expectation of "
                    "reward in it. Just the deep instinct to protect the people they love."
                ),
            },
            {
                "headline": "They would face anything for you.",
                "message":  (
                    "Whatever is out there, your dog isn't running from it. "
                    "They're standing between it and you, speaking up in the only "
                    "language they have. That has always been enough."
                ),
            },
        ],
    },

    # =========================================================================
    "unknown": {
        "emoji":    "🤔🐶",
        "fun_fact": (
            "Dogs have over 100 different facial expressions and can make around "
            "10 distinct vocal sounds. Apparently your pup is pioneering sound #11."
        ),
        "cute": [
            {
                "headline": "This bark is a mystery 🌟",
                "message":  (
                    "The audio features weren't quite enough to be sure, but one "
                    "thing is certain: this pup had something important to say "
                    "and they said it with full confidence."
                ),
            },
        ],
        "funny": [
            {
                "headline": "Science has no answers 🔬",
                "message":  (
                    "Your dog may be inventing a new dialect. Or testing the acoustics. "
                    "Or reviewing the quarterly numbers. We simply cannot tell. "
                    "Try a slightly longer or cleaner recording."
                ),
            },
        ],
        "emotional": [
            {
                "headline": "Some things are beyond words.",
                "message":  (
                    "Not every feeling has a clear name. Sometimes a dog barks "
                    "and we can only guess at the depth of what's behind it. "
                    "The uncertainty doesn't make it any less real."
                ),
            },
        ],
    },
}

# Appended when confidence is below this threshold (0–100 int scale)
_LOW_CONFIDENCE_THRESHOLD = 35
_LOW_CONFIDENCE_DISCLAIMER = (
    " (Confidence is a little low on this one — but that's our best read. "
    "A cleaner recording might give a sharper result.)"
)

# ---------------------------------------------------------------------------
# Voice assessment → tip cards
# ---------------------------------------------------------------------------

_PITCH_TIPS = {
    "steady": {
        "icon": "🎵",
        "status": "good",
        "tip": (
            "Solid, steady pitch — great for commands like 'stay' and 'down'. "
            "Your dog hears a confident, consistent leader."
        ),
    },
    "expressive": {
        "icon": "🎶",
        "status": "good",
        "tip": (
            "Nice expressive tone! Dogs love a little vocal enthusiasm, especially "
            "for recall ('come!') and praise. Keep that energy."
        ),
    },
    "inconsistent": {
        "icon": "🎵",
        "status": "warning",
        "tip": (
            "Your pitch varied quite a bit — dogs learn by hearing the same cue "
            "the same way each time. Try saying 'sit' with a firm, even tone every time."
        ),
    },
    "undetectable": {
        "icon": "🎵",
        "status": "bad",
        "tip": (
            "Pitch couldn't be detected — your voice may be too quiet or breathy. "
            "Try speaking with more projection and keep the mic close."
        ),
    },
}

_LOUDNESS_TIPS = {
    "too_soft": {
        "icon": "🔇",
        "status": "bad",
        "tip": (
            "Speak up! Dogs have excellent hearing, but they respond better to a clear, "
            "confident voice. Imagine you're calling across a room."
        ),
    },
    "just_right": {
        "icon": "🔊",
        "status": "good",
        "tip": (
            "Perfect volume — confident without being overwhelming. "
            "This is exactly what your dog wants to hear."
        ),
    },
    "too_loud": {
        "icon": "📢",
        "status": "warning",
        "tip": (
            "A bit loud — yelling can stress your dog or make them less responsive. "
            "A firm, calm tone works better than volume."
        ),
    },
}

_DURATION_TIPS = {
    "too_short": {
        "icon": "⏱️",
        "status": "warning",
        "tip": (
            "That was very quick — aim for about 0.5–1.5 seconds. Long enough "
            "for your dog to process, short enough to stay crisp."
        ),
    },
    "ideal": {
        "icon": "✅",
        "status": "good",
        "tip": (
            "Ideal command length! Short and punchy is exactly right. "
            "Dogs respond to clarity, not length."
        ),
    },
    "slightly_long": {
        "icon": "⏱️",
        "status": "warning",
        "tip": (
            "Slightly long — try trimming to just the core word: "
            "'sit!', 'stay!', 'come!' Dogs key in on the sound, not the sentence."
        ),
    },
    "too_long": {
        "icon": "⏱️",
        "status": "bad",
        "tip": (
            "Keep it short and punchy! 'Sit!' beats 'Now, can you please sit down for me?' "
            "every single time. Dogs listen for familiar sounds, not full sentences."
        ),
    },
}

_GRADE_SUMMARIES = {
    "excellent": {
        "label": "Excellent",
        "color": "#06D6A0",
        "message": "Woof-worthy delivery! Your dog is lucky to have such a clear communicator.",
    },
    "good": {
        "label": "Good",
        "color": "#74B9FF",
        "message": "Solid command delivery — one small tweak and you'll be speaking fluent dog.",
    },
    "needs_work": {
        "label": "Needs Work",
        "color": "#F7DC6F",
        "message": "Not bad, but there's room to grow! Check the tips below to level up.",
    },
    "unclear": {
        "label": "Unclear",
        "color": "#B2BEC3",
        "message": "Hard to get a read on this one — try again with a clearer, louder recording.",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_bark_translation(
    mood:       str,
    confidence: int,
    style:      str  = "funny",
    dog_name:   str | None = None,
) -> dict:
    """
    Return a styled bark translation plus the two alternate styles.

    Parameters
    ----------
    mood       : str       — one of the classifier MOODS or 'unknown'
    confidence : int       — 0–100 score from bark_classifier
    style      : str       — 'cute', 'funny', or 'emotional' (default 'funny')
    dog_name   : str|None  — optional name; if given, prefixes the message

    Returns
    -------
    dict with keys:
        headline   : str   — short punchy title for the selected translation
        message    : str   — main body text (name-formatted if dog_name given)
        emoji      : str   — mood emoji
        fun_fact   : str   — dog behaviour tidbit
        style      : str   — which style was used ('cute'/'funny'/'emotional')
        alternates : list  — list of 2 dicts, each with keys:
                               style, headline, message
                             representing the other two styles
    """
    style = style if style in _STYLES else "funny"
    bank  = _BANKS.get(mood, _BANKS["unknown"])

    selected   = _pick_entry(bank, style,      confidence)
    alternates = [
        {**_pick_entry(bank, s, confidence), "style": s}
        for s in _STYLES if s != style
    ]

    message = selected["message"]
    if confidence < _LOW_CONFIDENCE_THRESHOLD:
        message += _LOW_CONFIDENCE_DISCLAIMER

    if dog_name:
        name = dog_name.strip().title()
        message  = f"{name} says: {message}"
        headline = f"{name} — {selected['headline']}"
    else:
        headline = selected["headline"]

    return {
        "headline":  headline,
        "message":   message,
        "emoji":     bank["emoji"],
        "fun_fact":  bank["fun_fact"],
        "style":     style,
        "alternates": alternates,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _pick_entry(bank: dict, style: str, confidence: int) -> dict:
    """
    Choose one entry from bank[style] deterministically.

    The index is derived from `confidence` so the same result always maps
    to the same message, but different confidence values rotate through
    the bank to provide variety across different recordings.
    """
    entries = bank.get(style, bank.get("funny", [{"headline": "...", "message": "..."}]))
    idx = confidence % len(entries)
    return entries[idx]


def get_voice_tips(assessment: dict) -> list:
    """
    Convert a voice_analyzer assessment dict into a list of tip card dicts.

    Each card has: 'category', 'icon', 'tip', 'status'
    Status is one of: 'good', 'warning', 'bad'
    """
    cards = []

    # Pitch card
    pitch_label = assessment.get("pitch_assessment", {}).get("label", "undetectable")
    pitch_data  = _PITCH_TIPS.get(pitch_label, _PITCH_TIPS["undetectable"])
    cards.append({
        "category": "Pitch",
        "icon":     pitch_data["icon"],
        "tip":      assessment["pitch_assessment"].get("tip", pitch_data["tip"]),
        "status":   pitch_data["status"],
    })

    # Loudness card
    energy_level  = assessment.get("loudness_assessment", {}).get("energy_level", "just_right")
    loudness_data = _LOUDNESS_TIPS.get(energy_level, _LOUDNESS_TIPS["just_right"])
    cards.append({
        "category": "Volume",
        "icon":     loudness_data["icon"],
        "tip":      assessment["loudness_assessment"].get("tip", loudness_data["tip"]),
        "status":   loudness_data["status"],
    })

    # Duration card
    dur_label  = assessment.get("duration_assessment", {}).get("label", "ideal")
    dur_data   = _DURATION_TIPS.get(dur_label, _DURATION_TIPS["ideal"])
    cards.append({
        "category": "Duration",
        "icon":     dur_data["icon"],
        "tip":      assessment["duration_assessment"].get("tip", dur_data["tip"]),
        "status":   dur_data["status"],
    })

    return cards


def get_grade_summary(overall_grade: str) -> dict:
    """
    Return a summary card dict for the overall command grade.
    Keys: 'label', 'color', 'message'
    """
    return _GRADE_SUMMARIES.get(overall_grade, _GRADE_SUMMARIES["unclear"])
