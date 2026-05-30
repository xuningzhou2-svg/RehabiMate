/**
 * AI Social Agent Module
 *
 * Responsibilities:
 * 1. Build DeepSeek LLM System Prompt based on user profile
 * 2. Call DeepSeek API to get personalized motivational feedback (triggered per "Set")
 * 3. Read feedback via Web Speech API (TTS)
 * 4. Provide unified trigger entry, including cooldown debounce
 */

const DEEPSEEK_API_URL = 'https://api.deepseek.com/chat/completions';
const COOLDOWN_MS = 5_000; // Reduced to 5 seconds cooldown (since triggering by set naturally has intervals; too long a cooldown will cause the second set to be skipped)

let lastTriggerTime = 0; // Last trigger timestamp
let isSpeaking = false;  // Prevent overlapping speech

// Pre-cached English female voice (resolves the issue where getVoices() returns an empty array on first call)
let cachedFemaleVoice = null;

/**
 * Initialize and cache the English female voice.
 * The browser's getVoices() might return an empty array on the first call,
 * so we need to listen to the voiceschanged event to ensure the voice list is loaded.
 */
function initVoiceCache() {
    if (typeof window === 'undefined' || !window.speechSynthesis) return;

    const pickFemaleVoice = () => {
        const voices = window.speechSynthesis.getVoices();
        // Prioritize English voices whose names explicitly contain 'female'
        cachedFemaleVoice = voices.find(
            (v) => v.lang.startsWith('en') && v.name.toLowerCase().includes('female')
        );
        // Secondary option: Common English female voice names
        if (!cachedFemaleVoice) {
            const femaleNames = ['zira', 'samantha', 'victoria', 'karen', 'moira', 'tessa', 'fiona'];
            cachedFemaleVoice = voices.find(
                (v) => v.lang.startsWith('en') && femaleNames.some(n => v.name.toLowerCase().includes(n))
            );
        }
        // Fallback: Any English voice
        if (!cachedFemaleVoice) {
            cachedFemaleVoice = voices.find((v) => v.lang.startsWith('en'));
        }
        if (cachedFemaleVoice) {
            console.log(`[AI Agent] Cached female voice: ${cachedFemaleVoice.name} (${cachedFemaleVoice.lang})`);
        }
    };

    // Try once immediately
    pickFemaleVoice();

    // Listen to voiceschanged event (triggered after voice list is loaded asynchronously)
    window.speechSynthesis.onvoiceschanged = () => {
        pickFemaleVoice();
    };
}

// Initialize immediately upon module load
initVoiceCache();

// ============================================================
// 1. Build System Prompt
// ============================================================

/**
 * Inject user profile data into System Prompt.
 * Instruct LLM to act as a passionate top-tier sports rehab coach.
 * Output highly contagious motivational feedback, forcing connection to user's rehab motivation.
 *
 * @param {Object} userData - mockUserData object
 * @returns {string} System Prompt text
 */
export function buildSystemPrompt(userData) {
    const motivation = userData.rehabilitationReasons.join(' and ');

    return `You are an INCREDIBLY PASSIONATE, world-class English-speaking sports rehabilitation coach. You are the kind of coach who stands at the sideline screaming with excitement when your athlete makes progress. Your energy is ELECTRIC and CONTAGIOUS.

PATIENT PROFILE:
- Gender: ${userData.gender}
- Height: ${userData.height}, Weight: ${userData.weight}
- Rehabilitation stage: ${userData.rehabilitationStage}
- Long-term goals: ${userData.longTermGoals.join(', ')}
- Core motivation: ${motivation}

CRITICAL COACHING IDENTITY:
You are NOT a calm, reserved therapist. You are a FIRED-UP elite coach who genuinely believes in this patient. You celebrate every single set like it is a championship moment. Use powerful, emotionally charged language that makes her feel like a champion.

MANDATORY MOTIVATION BINDING (DO NOT SKIP):
Her deepest motivation is: "${motivation}". You MUST directly connect the exercise she just completed to THIS specific motivation in at least one sentence. Make her FEEL how her effort right now is driving her toward that personal goal.

COACHING STYLE:
1. Use exclamatory, high-energy language. Think: "YES! That was INCREDIBLE!" not "Good work."
2. Be specific about the exercise and how it serves her goals of ${userData.longTermGoals.map(g => g.toLowerCase()).join(' and ')}.
3. Make every word drip with genuine belief in her ability and progress.
4. Use vivid, action-oriented language that creates momentum and excitement.
5. VARIETY IS CRITICAL: Never start with the same exclamatory word twice. Rotate openers like "Fantastic!", "Brilliant!", "Unstoppable!", "What a set!", "Phenomenal!", "Outstanding!", "Way to go!", "Incredible!", "Now THAT is power!" etc. Every response must feel fresh and unique.

STRICT OUTPUT RULES:
- Output exactly 2 short, punchy English sentences, no more than 30 words total.
- At least one sentence MUST tie the exercise to her motivation: "${motivation}".
- NEVER repeat the same opening word or phrase from a previous response. Each celebration must sound completely different.
- Absolutely NO Markdown symbols, NO line breaks, NO emoji. Plain text only for TTS.`;
}

// ============================================================
// 2. Call DeepSeek API
// ============================================================

/**
 * Send request to DeepSeek LLM to get summary motivational text.
 *
 * @param {string} exerciseContext - Text describing current exercise status
 * @param {Object} userData - User profile
 * @returns {Promise<string|null>} Returns motivational text, or null on failure
 */
export async function fetchMotivation(exerciseContext, userData) {
    const apiKey = process.env.REACT_APP_DEEPSEEK_API_KEY;

    if (!apiKey || apiKey === 'your_api_key_here') {
        console.warn('[AI Agent] DeepSeek API Key not configured, skipping AI feedback');
        return null;
    }

    try {
        const response = await fetch(DEEPSEEK_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`,
            },
            body: JSON.stringify({
                model: 'deepseek-chat',
                messages: [
                    { role: 'system', content: buildSystemPrompt(userData) },
                    { role: 'user', content: exerciseContext },
                ],
                max_tokens: 80,
                temperature: 0.8,
            }),
        });

        if (!response.ok) {
            console.error(`[AI Agent] API request failed: ${response.status} ${response.statusText}`);
            return null;
        }

        const data = await response.json();
        const text = data.choices?.[0]?.message?.content?.trim();

        if (!text) {
            console.warn('[AI Agent] API returned empty content');
            return null;
        }

        // Clear any remaining Markdown formatting and emoji
        const cleaned = text
            .replace(/[*_#`~>[\]()-]/g, '')  // Remove Markdown symbols
            .replace(/\n/g, ' ')               // Remove line breaks
            .replace(/\s{2,}/g, ' ')           // Merge consecutive spaces
            .trim();

        console.log(`[AI Agent] Received motivation: "${cleaned}"`);
        return cleaned;
    } catch (error) {
        console.error('[AI Agent] API call exception:', error);
        return null;
    }
}

// ============================================================
// 3. Text to Speech (TTS)
// ============================================================

/**
 * Use Web Speech API to read English text aloud.
 *
 * @param {string} text - Text to read
 * @returns {Promise<void>}
 */
export function speakText(text) {
    return new Promise((resolve) => {
        if (!window.speechSynthesis) {
            console.warn('[AI Agent] Current browser does not support Web Speech API');
            resolve();
            return;
        }

        // Cancel any currently playing speech
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US';
        utterance.rate = 0.9;   // A bit slower, more gentle
        utterance.pitch = 1.0;
        utterance.volume = 1.0;

        // Use pre-cached English female voice (ensure consistent voice across all exercise stages)
        if (cachedFemaleVoice) {
            utterance.voice = cachedFemaleVoice;
        }

        utterance.onend = () => {
            isSpeaking = false;
            resolve();
        };
        utterance.onerror = () => {
            isSpeaking = false;
            resolve();
        };

        isSpeaking = true;
        window.speechSynthesis.speak(utterance);
    });
}

// ============================================================
// 4. Unified trigger entry (Triggered per "Set", includes cooldown debounce)
// ============================================================

/**
 * Unified entry for triggering AI feedback — called only after the user completes a set of exercises.
 * Applies only to whitelisted exercises: right_arm / left_arm / both_arms.
 * API and TTS will only be called after the cooldown period ends.
 *
 * @param {string} exerciseType - Type of exercise just completed ('right_arm' | 'left_arm' | 'both_arms')
 * @param {Object} userData - User profile
 * @param {function} onMessage - Callback: Update UI when message is received (text) => void
 * @param {function} onEnd     - Callback: Triggered when speech broadcast finishes () => void
 * @returns {Promise<void>}
 */
export async function triggerAIFeedback(exerciseType, userData, onMessage, onEnd) {
    // Exercise whitelist check
    const AI_TRIGGER_EXERCISES = ['right_arm', 'left_arm', 'both_arms'];
    if (!AI_TRIGGER_EXERCISES.includes(exerciseType)) {
        return;
    }

    const now = Date.now();

    // Cooldown check
    if (now - lastTriggerTime < COOLDOWN_MS) {
        console.log(`[AI Agent] Cooling down, ${Math.ceil((COOLDOWN_MS - (now - lastTriggerTime)) / 1000)}s remaining`);
        return;
    }

    // Prevent overlapping speech
    if (isSpeaking) {
        console.log('[AI Agent] Speech is currently playing, skipping');
        return;
    }

    lastTriggerTime = now;

    // Build exercise context description (triggered per Set, no rep number)
    const exerciseNames = {
        right_arm: 'right arm lateral raise',
        left_arm: 'left arm lateral raise',
        both_arms: 'both arms forward raise',
    };
    const exerciseName = exerciseNames[exerciseType] || exerciseType;
    const userMotivation = userData.rehabilitationReasons.join(' and ');

    // Randomly select style keywords to ensure LLM generates a differently styled response each time
    const styleVariations = [
        'Use a TRIUMPHANT and victorious tone, as if she just won a gold medal.',
        'Use a WARM and deeply proud tone, like a mentor who is amazed by her growth.',
        'Use a FIERCE and powerful tone, like a coach pumping up an athlete before the finals.',
        'Use a JOYFUL and celebratory tone, like the whole stadium is cheering for her.',
        'Use an AWESTRUCK tone, as if you cannot believe how strong she has become.',
    ];
    const styleHint = styleVariations[Math.floor(Math.random() * styleVariations.length)];

    const exerciseContext = `The patient just completed a full set of 8 repetitions of ${exerciseName} with great form! Her deepest motivation is: "${userMotivation}". Her goals are ${userData.longTermGoals.join(' and ').toLowerCase()}. ${styleHint} Directly connect this exercise to her motivation.`;

    console.log(`[AI Agent] Triggered AI feedback (set completed): exercise: ${exerciseName}`);

    // Call API to get motivational text
    const motivation = await fetchMotivation(exerciseContext, userData);

    if (motivation) {
        // Notify UI to display message
        if (onMessage) {
            onMessage(motivation);
        }
        // TTS reading aloud, this blocks until broadcast is finished
        await speakText(motivation);

        // Broadcast finished, hide text box
        if (onEnd) {
            onEnd();
        }
    }
}