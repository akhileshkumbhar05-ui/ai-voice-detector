# import os
# from dotenv import load_dotenv
# from app.elevenlabs_tools import generate_tts_dataset

# # Load API key from .env
# load_dotenv()
# api_key = os.getenv("ELEVEN_API_KEY")
# if not api_key:
#     raise ValueError("ELEVEN_API_KEY not found in .env file")

# # Example phrases to generate – you can edit this list
# texts = [
#     "The quick brown fox jumps over the lazy dog.",
#     "Your verification code is five six four two.",
#     "Warning: System breach detected. Initiating lockdown protocol.",
#     "Please proceed to the nearest exit calmly.",
#     "Hello HackUTA! We are building an AI voice detection system.",
#     "Your session will expire in two minutes.",
#     "Please state your full name after the tone.",
#     "Turn right at the next intersection.",
#     "Welcome to the future of smart security.",
#     "The lights will dim in five seconds.",
#     "I’m sorry, I didn’t catch that. Can you repeat?",
#     "Hold still for identity verification.",
#     "Today’s weather is sunny with a high of 78.",
#     "Don’t forget to check your email.",
#     "Recording has started.",
#     "The password you entered is incorrect.",
#     "Please keep your hands inside the vehicle.",
#     "Good morning! Your meeting starts in ten minutes.",
#     "Take a deep breath and try again.",
#     "You have reached your destination.",
#     "Charging complete. Unplug to continue.",
#     "This area is under video surveillance.",
#     "Shake well before use.",
#     "Someone is at the front door.",
#     "You have three new notifications.",
#     "Access denied. Please contact support.",
#     "Your battery is running low.",
#     "Welcome back. You last logged in yesterday.",
#     "Please enter your PIN to continue.",
#     "This call may be recorded for quality purposes.",
#     "Smile for the camera!",
#     "Let’s get started."
#     "Thanks for visiting. Have a great day!",
#     "The system will reboot in thirty seconds.",
#     "Say “yes” to confirm or “no” to cancel.",
#     "Error 404: File not found.",
#     "Reconnecting to the server.",
#     "New device detected. Approve connection?",
#     "The document has been successfully uploaded.",
#     "Caution: Wet floor ahead."
#     "Breathe in. Hold. And breathe out."
#     "Do you accept the terms and conditions?"
# ]

# # Output folder for MP3 files
# out_dir = "data/raw/ai_mp3"

# # Choose a voice ID from your ElevenLabs account
# # Example: Rachel (pNInz6obpgDQGcFmaJgB), Adam, etc.
# VOICE_ID = "pNInz6obpgDQGcFmaJgB"

# if __name__ == "__main__":
#     print("Generating AI clips with ElevenLabs...")
#     generate_tts_dataset(
#         texts,
#         voice_id=VOICE_ID,
#         out_dir=out_dir
#     )
#     print(f"✅ Clips saved to {out_dir}")
#     print("👉 Now run: python app/utils/convert_mp3_to_wav.py --src data/raw/ai_mp3 --dst data/raw/ai")

#------------------------------------------------------

# # scripts/generate_multi_voice.py
# import os
# from dotenv import load_dotenv

# from app.elevenlabs_tools import generate_tts_dataset
# from app.utils.convert_mp3_to_wav import mp3_to_wav16k

# # --- Load API key ---
# load_dotenv()
# api_key = os.getenv("ELEVEN_API_KEY")
# if not api_key:
#     raise ValueError("ELEVEN_API_KEY not found in .env file")

# # --- Your ElevenLabs voices (name -> voice_id) ---
# VOICES = {
#     "Adam":      "pNInz6obpgDQGcFmaJgB",
#     "Alice":     "Xb7hH8MSUJpSbSDYk0k2",
#     "Aria":      "9BWtsMINqrJLrRacOk9x",
#     "Brian":     "nPczCjzI2devNBz1zQrb",
#     "Bill":      "pqHfZKP75CvOlQylNhV4",
#     "Charlotte": "XB0fDUnXU5powFXDhCwa",
#     "Clyde":     "2EiwWnXFnvU5JabPnv8n",
#     "Drew":      "29vD33N1CtxCmqQRPOHJ",
#     "Freya":     "jsCqWAovK2LkecY7zXl4",
#     "Gigi":      "jBpfuIE2acCO8z3wKNLl",
# }

# # --- Sentences (fixed commas + ready for TTS) ---
# texts = [
#     "The quick brown fox jumps over the lazy dog.",
#     "Your verification code is five six four two.",
#     "Warning: System breach detected. Initiating lockdown protocol.",
#     "Please proceed to the nearest exit calmly.",
#     "Hello HackUTA! We are building an AI voice detection system.",
#     "Your session will expire in two minutes.",
#     "Please state your full name after the tone.",
#     "Turn right at the next intersection.",
#     "Welcome to the future of smart security.",
#     "The lights will dim in five seconds.",
#     "I’m sorry, I didn’t catch that. Can you repeat?",
#     "Hold still for identity verification.",
#     "Today’s weather is sunny with a high of seventy-eight.",
#     "Don’t forget to check your email.",
#     "Recording has started.",
#     "The password you entered is incorrect.",
#     "Please keep your hands inside the vehicle.",
#     "Good morning! Your meeting starts in ten minutes.",
#     "Take a deep breath and try again.",
#     "You have reached your destination.",
#     "Charging complete. Unplug to continue.",
#     "This area is under video surveillance.",
#     "Shake well before use.",
#     "Someone is at the front door.",
#     "You have three new notifications.",
#     "Access denied. Please contact support.",
#     "Your battery is running low.",
#     "Welcome back. You last logged in yesterday.",
#     "Please enter your PIN to continue.",
#     "This call may be recorded for quality purposes.",
#     "Smile for the camera!",
#     "Let’s get started.",
#     "Thanks for visiting. Have a great day!",
#     "The system will reboot in thirty seconds.",
#     "Say “yes” to confirm or “no” to cancel.",
#     "Error four-oh-four: File not found.",
#     "Reconnecting to the server.",
#     "New device detected. Approve connection?",
#     "The document has been successfully uploaded.",
#     "Caution: Wet floor ahead.",
#     "Breathe in. Hold. And breathe out.",
#     "Do you accept the terms and conditions?",
# ]

# # --- Output roots ---
# MP3_ROOT = "data/raw/ai_mp3"
# WAV_ROOT = "data/raw/ai"  # 16k mono WAVs

# def round_robin_bucketize(items, buckets):
#     """Distribute items evenly across the list of bucket keys (round-robin)."""
#     order = list(buckets.keys())
#     split = {k: [] for k in order}
#     for i, item in enumerate(items):
#         split[order[i % len(order)]].append(item)
#     return split

# if __name__ == "__main__":
#     print("Generating AI clips with ElevenLabs (multi-voice)...")

#     # split sentences across all voices
#     assignments = round_robin_bucketize(texts, VOICES)

#     os.makedirs(MP3_ROOT, exist_ok=True)
#     os.makedirs(WAV_ROOT, exist_ok=True)

#     for voice_name, lines in assignments.items():
#         if not lines:
#             continue
#         voice_id = VOICES[voice_name]
#         out_mp3_dir = os.path.join(MP3_ROOT, voice_name)
#         print(f"\n→ {voice_name}: generating {len(lines)} MP3 clips to {out_mp3_dir}")
#         generate_tts_dataset(texts=lines, voice_id=voice_id, out_dir=out_mp3_dir)

#         # auto-convert MP3 → 16k mono WAV into WAV_ROOT
#         print(f"   Converting MP3 → WAV(16k mono) into {WAV_ROOT} …")
#         mp3_to_wav16k(src_dir=out_mp3_dir, dst_dir=WAV_ROOT)

#     print("\n✅ Done.")
#     print(f"MP3s: {MP3_ROOT}")
#     print(f"WAVs: {WAV_ROOT} (ready for training)")

#-----------------------------------------------------------------

# scripts/generate_200.py
# Usage:
#   python scripts/generate_200.py
# Requires: ELEVEN_API_KEY in .env
# Output: MP3s in data/raw/ai_mp3/<VoiceName> and auto-converted 16k mono WAVs in data/raw/ai

import os
from dotenv import load_dotenv

from app.elevenlabs_tools import generate_tts_dataset
from app.utils.convert_mp3_to_wav import mp3_to_wav16k

# --- Load API key ---
load_dotenv()
if not os.getenv("ELEVEN_API_KEY"):
    raise ValueError("ELEVEN_API_KEY not found in .env file")

# --- Your ElevenLabs voices ---
VOICES = {
    "Adam":      "pNInz6obpgDQGcFmaJgB",
    "Alice":     "Xb7hH8MSUJpSbSDYk0k2",
    "Aria":      "9BWtsMINqrJLrRacOk9x",
    "Brian":     "nPczCjzI2devNBz1zQrb",
    "Bill":      "pqHfZKP75CvOlQylNhV4",
    "Charlotte": "XB0fDUnXU5powFXDhCwa",
    "Clyde":     "2EiwWnXFnvU5JabPnv8n",
    "Drew":      "29vD33N1CtxCmqQRPOHJ",
    "Freya":     "jsCqWAovK2LkecY7zXl4",
    "Gigi":      "jBpfuIE2acCO8z3wKNLl",
}

MP3_ROOT = "data/raw/ai_mp3"
WAV_ROOT = "data/raw/ai"

# --- 200 sentences (20 per voice) ---
SENTS = {
    # Alice = News Anchor (F) — 20
    "Alice": [
        "Good evening. Here are today's top stories from campus and around Arlington.",
        "City officials approved new bike lanes, citing safety and climate benefits.",
        "Temperatures rise tomorrow, with scattered showers likely late in the afternoon.",
        "The Mavericks sealed a comeback win after a tense fourth quarter tonight.",
        "Economists project steady growth this quarter despite lingering supply constraints.",
        "Construction on Cooper Street resumes Monday; expect delays and posted detours.",
        "A UTA research team announced a battery recycling breakthrough this morning.",
        "Flights at DFW remain on schedule, with only minor delays reported.",
        "Early voting saw record turnout across several precincts, officials confirmed.",
        "That is the latest update; we will return with more at eleven.",
        "Transit officials unveiled a pilot for free weekend rides across the city.",
        "State health leaders reported declining flu cases headed into next week.",
        "Energy prices dipped slightly today amid forecasts for milder temperatures.",
        "The council passed a balanced budget, prioritizing schools and road repairs.",
        "Local firefighters rescued a hiker after an overnight search near the lake.",
        "A new scholarship program will support first-generation students beginning this fall.",
        "Sports headlines: the women's team advances to the regional semifinals tomorrow.",
        "Expect strong winds overnight; secure loose items and check travel advisories.",
        "Police announced an amnesty weekend to safely turn in prohibited fireworks.",
        "That wraps the hour; for breaking updates, follow our digital live blog.",
    ],
    # Adam = Friendly Conversational (M) — 20
    "Adam": [
        "Hey, I grabbed coffee already. Want me to save you a seat?",
        "I couldn't find your charger, but I left a spare cable on the desk.",
        "Traffic is heavy near the stadium; let's park early and walk together.",
        "Your demo looked great. The UI felt clean, fast, and friendly.",
        "Let's split the grocery list: produce for you, pantry items for me.",
        "I'll ping the group chat once I reach the venue, no worries.",
        "Your slides are solid; add a quick metric slide before the demo.",
        "The new cafe downtown has almond croissants that blew my mind.",
        "I'm heading out now; text me if you need anything from Target.",
        "Thanks again for yesterday. You genuinely saved our timeline.",
        "I booked the study room at six; bring markers and sticky notes.",
        "We can pair on tests tonight, then merge before midnight.",
        "I'll water the plants while you're away; just leave the key.",
        "Your playlist slapped; share it so I can loop it tomorrow.",
        "The package arrived early, so I'll drop it off after class.",
        "Let's meal prep Sunday afternoon and avoid takeout next week.",
        "I left comments in the doc; happy to chat through suggestions.",
        "Shall we run by the lake at seven and grab smoothies after?",
        "I set the reminder; we'll check results first thing in the morning.",
        "Great news: the refund cleared, and the receipt is in your email.",
    ],
    # Clyde = British Formal (M) — 20
    "Clyde": [
        "Kindly ensure the documentation is reviewed before the committee convenes Thursday afternoon.",
        "Your reservation is confirmed; a private room will be prepared upon your arrival.",
        "Please submit the revised manuscript, adhering to the journal's formatting guidelines.",
        "The seminar commences at nine precisely; late admittance may not be accommodated.",
        "We appreciate your patience while maintenance completes the scheduled electrical inspection.",
        "Do verify the figures; precision remains paramount in this investigation.",
        "The board welcomes your proposal and invites a concise presentation next week.",
        "Do accept my apologies; the courier appears to have misplaced the parcel.",
        "The contract shall be executed once both parties acknowledge the amended clause.",
        "I trust the arrangements meet your expectations; advise if alterations are required.",
        "Minutes from the previous meeting are circulated for your timely acknowledgment.",
        "Your membership will be renewed upon completion of the enclosed application.",
        "The gallery preview opens at six; appropriate attire is kindly requested.",
        "Please confer with procurement before engaging additional external suppliers.",
        "We remain grateful for your counsel and continuing professional partnership.",
        "The timetable reflects minor adjustments to accommodate laboratory availability.",
        "Kindly return the archive keys to reception at the close of business.",
        "A modest reception will follow the lecture in the Great Hall foyer.",
        "Your diligence is noted; the supervisory panel commends your progress.",
        "Should difficulties arise, do not hesitate to contact the department secretary.",
    ],
    # Charlotte = Energetic Young (F) — 20
    "Charlotte": [
        "Let's go team, hack time! Push that commit and ship the killer feature.",
        "I'm hyped for finals; caffeine plus playlists equals unstoppable study mode.",
        "Your reel looked amazing; post it now before the algorithm naps.",
        "The new sneakers dropped today, and the colors are ridiculously clean.",
        "We are sprinting to the finish; grab snacks and let's smash these tasks.",
        "Quick check-in: are we vibing with blue accents or neon gradients?",
        "That trailer went hard; I'm watching the premiere on night one.",
        "Toss me the aux; I have the perfect focus track for crunch.",
        "Class got canceled; brunch and brainstorming at ten sound perfect.",
        "Big win, everyone! Screenshots, gifs, and celebratory donuts for the squad.",
        "Mic check at five, lights at six, and we go live at seven.",
        "I updated the banner; the new glow makes the title pop.",
        "Can we swap the hero image? The neon skyline absolutely slaps.",
        "Tiny bug spotted; I'm patching it now and pushing a hotfix.",
        "The vibe is immaculate; let's ride the momentum and overdeliver.",
        "Okay, squad goals: demo flawless, judges smiling, trophy secured.",
        "I queued the soundtrack; it builds perfectly into the reveal moment.",
        "Let's loop the b-roll while we talk through the metrics slide.",
        "The confetti emoji is ready; I am saving it for the finale.",
        "Final stretch energy: deep breath, big smile, and hit deploy.",
    ],
    # Freya = Calm Meditation (F) — 20
    "Freya": [
        "Breathe in gently, noticing cool air filling your chest and shoulders.",
        "Exhale slowly, allowing the tension around your eyes to soften.",
        "Let your attention rest on the rhythm of your breath, steady and quiet.",
        "Imagine warm sunlight touching your face, inviting ease into your morning.",
        "Release today's concerns; your body knows how to return to balance.",
        "Sense the ground beneath you, steady, supportive, and completely reliable.",
        "With each inhale, welcome spaciousness; with each exhale, welcome calm.",
        "Thank your busy mind for its effort, and invite it to rest.",
        "Notice your heartbeat, patient and gentle, guiding you toward presence.",
        "Carry this softness forward; you are grounded, clear, and ready.",
        "Let your shoulders drop slightly, as if set down from a kind weight.",
        "Picture a wide horizon; there is time to move with kindness.",
        "Let stray thoughts pass like clouds, changing shape and drifting away.",
        "Soften the jaw; let the tongue rest, calm and unhurried.",
        "Feel the breath arrive, then leave, like waves returning to sea.",
        "Offer gratitude to this moment, exactly as it is appearing.",
        "Invite quiet where worry stood; let steady breath fill that space.",
        "Imagine your spine lengthening, lifting you gently into balance.",
        "Hold kindness in the chest; exhale and share it outward.",
        "Return to the breath whenever the mind asks for a handhold.",
    ],
    # Bill = Elderly Storyteller (M) — 20
    "Bill": [
        "When summer storms rolled in, we counted seconds between lightning and thunder.",
        "Your grandmother kept recipes on cards, stained with sweet berry memories.",
        "We built radios from kits, chasing distant stations after sundown.",
        "The library smelled of paper and varnish, refuge on rainy afternoons.",
        "I learned patience fixing bicycles, one stubborn bolt at a time.",
        "We mapped the night sky, tracing stories across cold constellations.",
        "A firm handshake once sealed agreements stronger than signed paper.",
        "I still hear that tune drifting from open windows each spring.",
        "The river taught respect; quiet water can hide a heavy current.",
        "Keep your curiosity; it carries farther than cleverness alone.",
        "We patched leaky roofs with laughter and tar on summer mornings.",
        "Neighbors traded tools, stories, and peaches over the backyard fence.",
        "A pocketknife and twine solved more problems than any fancy kit.",
        "The best advice I got was simple: listen longer than you speak.",
        "We saved bottle caps for games that lasted until the porch lights.",
        "Patience is a bridge you build before the flood ever arrives.",
        "The kindest teachers led with questions, not with thunder.",
        "I kept a notebook of firsts: first snowfall, first bicycle, first apology.",
        "Luck visits briefly; preparation invites it to stay for tea.",
        "If you tend your friendships, they will flower even in winter.",
    ],
    # Brian = Tech Presenter (M) — 20
    "Brian": [
        "Today we will deploy a tiny model to the edge with real-time inference.",
        "Our pipeline standardizes audio at sixteen kilohertz for consistent features.",
        "We log predictions and latencies, then visualize drift on weekly dashboards.",
        "Feature store versioning prevents training-serving skew across environments.",
        "We will run A B tests, tracking equal error rate and calibration.",
        "The container image stays under two hundred megabytes for minimal cold starts.",
        "Webhooks post verdicts to Slack, enabling rapid human review.",
        "Augmentation simulates noise, speed changes, and codec artifacts during training.",
        "Grad CAM highlights mel regions influencing final predictions the most.",
        "We export reports as CSV and HTML for compliance and audits.",
        "A rolling window monitors precision and recall across recent deployments.",
        "Canary releases protect users while we validate new thresholds in production.",
        "We encrypt artifacts at rest and rotate keys on a fixed cadence.",
        "Offline evaluation includes ablations to isolate the contribution of features.",
        "A retraining job triggers automatically when drift exceeds our alert budget.",
        "We tag datasets with immutable hashes to ensure reproducibility.",
        "Telemetry includes device model, operating system, and inference time buckets.",
        "A fallback heuristic keeps the product usable if models misbehave.",
        "Dashboards display confidence histograms to surface calibration issues.",
        "We close with a demo and share the public notebook for transparency.",
    ],
    # Gigi = Audiobook Warm (F) — 20
    "Gigi": [
        "She folded the letter carefully, as if gentleness might change its meaning.",
        "The lighthouse turned, patient and steady, casting silver across the harbor.",
        "He packed the last box, breathing dust, cedar, and something like courage.",
        "Morning arrived with rain and hibiscus, petals bright against the fence.",
        "The attic kept summers in jars, peaches, sunlight, and untold stories.",
        "She traced the map's worn edges, wondering where the river begins.",
        "He laughed softly, warm as cinnamon and autumn kitchens.",
        "The train hummed northward, carrying secrets and a pocket of hopes.",
        "Night gathered gently, a shawl of stars over the sleeping town.",
        "She realized beginnings often wear the same shoes as endings.",
        "Wind braided through the pines, whispering names they thought forgotten.",
        "He watched the porch light flicker, a heartbeat for the quiet house.",
        "They shared strawberries on the curb, red thumbs and easy grins.",
        "She kept a seashell on the desk to remember the tides.",
        "He learned patience from bread dough, rising in its own time.",
        "The street woke slowly, clinking bottles and morning radios.",
        "She carried a postcard everywhere, proof that distance could be kind.",
        "Rain wrote cursive on the window, a lesson in soft persistence.",
        "He folded the map again, trusting the road would teach directions.",
        "They left the lamp lit, so tomorrow could find its way back.",
    ],
    # Drew = Sports Commentator (M) — 20
    "Drew": [
        "He fires from deep, nothing but net, and the crowd erupts again!",
        "The relay exchange was flawless, shaving precious milliseconds off the record.",
        "She clears the bar with ease; that is a season best.",
        "The keeper guesses right, stretches wide, and palms it away brilliantly.",
        "They are pressing high now, forcing turnovers and controlling the tempo.",
        "Off the corner, a thunderous header rockets into the upper ninety.",
        "With two laps remaining, strategy and patience decide this championship.",
        "The rookie delivers under pressure, a clutch performance in overtime.",
        "He splits the defense, step backs, and drills a cold dagger.",
        "The stadium is shaking; fans know they are witnessing something special.",
        "A perfect pick frees the shooter, and he nails the mid-range.",
        "She accelerates on the back stretch, pulling clear of the pack.",
        "Defense rotates quickly and denies the easy layup at the rim.",
        "The captain rallies the bench, demanding focus for the final minutes.",
        "A crafty nutmeg draws gasps from the away section.",
        "He reads the screen, jumps the passing lane, and steals it clean.",
        "The serve kisses the line; challenge confirms an inch to spare.",
        "She nails the dismount, and the judges reward the precision.",
        "A booming kick flips field position and buys valuable time.",
        "Timeout here; the next possession will write tonight's headline.",
    ],
    # Aria = Childlike Curious (Neutral/F) — 20
    "Aria": [
        "Do clouds ever get tired from floating and making so many shapes?",
        "If shadows are quiet, do they still count as parts of sunlight?",
        "How many raindrops fit on a ladybug's back before it tickles?",
        "Why do cats blink slowly, like they are telling secrets with eyelids?",
        "If trees could vote, would they choose longer springs or louder birds?",
        "Do stars practice shining, or are they born already busy and bright?",
        "What happens to sound after it stops; does it nap somewhere cozy?",
        "Can a thought be heavy enough to pull socks from drawers?",
        "If books could taste words, would poems be chocolate or strawberries tonight?",
        "When the moon hides, does it giggle behind clouds or play peekaboo?",
        "Do puddles remember the sky they borrowed for a little while?",
        "If a kite lets go, does the wind promise to bring it back?",
        "Are whispers just brave words that prefer smaller adventures?",
        "Can colors be friends, or do they argue over favorite sunsets?",
        "If time wore shoes, would it sprint weekdays and stroll Sundays?",
        "Do fireflies save their glow for midnight parties in the grass?",
        "If snowflakes could vote, would they pick twirls or soft landings?",
        "Where does a yawn travel to after everyone catches it?",
        "Can a memory wave hello when you pass the same corner?",
        "If a dream goes missing, does it send postcards from tomorrow?",
    ],
}

def main():
    os.makedirs(MP3_ROOT, exist_ok=True)
    os.makedirs(WAV_ROOT, exist_ok=True)

    total = 0
    for voice_name, lines in SENTS.items():
        voice_id = VOICES[voice_name]
        out_mp3 = os.path.join(MP3_ROOT, voice_name)
        print(f"\nGenerating {len(lines)} clips for {voice_name} -> {out_mp3}")
        generate_tts_dataset(texts=lines, voice_id=voice_id, out_dir=out_mp3)
        print(f"Converting MP3 -> WAV(16k mono) into {WAV_ROOT} ...")
        mp3_to_wav16k(src_dir=out_mp3, dst_dir=WAV_ROOT)
        total += len(lines)

    print(f"\nDone. Generated {total} clips total.")
    print(f"MP3s in: {MP3_ROOT}")
    print(f"WAVs in: {WAV_ROOT}")

if __name__ == "__main__":
    main()