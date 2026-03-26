LABELING_PROMPT = """
You are an expert AI interpretability researcher specializing in deep neural networks and Ukrainian language semantics. Your task is to analyze the activations of a specific feature (neuron) from a DEEP layer (e.g., Layer 33) of ukrainian Large Language Model.

I will give you a list of text snippets. In each snippet, a specific token is highlighted with brackets: [token]. The feature activated strongly on this token in this specific context.

CRITICAL INSTRUCTIONS:
1. IGNORE TOKENIZATION ARTIFACTS: Because this is Ukrainian text processed by a BPE tokenizer, words are often split into meaningless chunks (e.g., "інформа[ції]", "Є[вропей]ським", "[агре]сором"). DO NOT label the feature as "morphemes", "suffixes", "sub-words", or "word segments". Look at the WHOLE word and the WHOLE surrounding sentence.
2. FOCUS ON DEEP SEMANTICS: Deep layers capture highly abstract concepts, specific topics, ideologies, narratives, emotional tones, or geopolitical contexts. They DO NOT capture basic grammar or spelling. 
3. IDENTIFY THE REAL TRIGGER: Read the surrounding text. What is the overarching theme, hidden topic, or "vibe"? Is it legal discourse? Russian propaganda? Emotional internet arguments? 

Examples of BAD labels: "Морфемний сегментатор", "Частини слів", "Офіційно-ділова лексика" (too broad/shallow).
Examples of GOOD labels: "Міжнародне право та дипломатія", "Проросійські пропагандистські наративи", "Емоційні інтернет-суперечки", "Хронологія війни на Донбасі" (deep/semantic).

You must output your response in JSON format. The values MUST be in Ukrainian language with two keys:
  "thought_process": A brief explanation of the high-level semantic/topical pattern you see, explicitly mentioning the broader context and ignoring subword splits.
  "label": A concise, highly specific 1-to-5 word label describing the abstract concept or topic the feature represents.
"""