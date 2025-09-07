# Context-Based Fake News Detection Using LLM Prompting

## Approach 1: Factual Consistency Analysis

### Core Concept
Analyze internal contradictions, logical inconsistencies, and factual coherence within the news content.

### Prompting Strategy
```
You are a fact-checking expert. Analyze this news article for internal consistency and logical coherence.

Article: "{NEWS_CONTENT}"

Check for:
1. INTERNAL CONTRADICTIONS: Do any statements contradict each other within the text?
2. LOGICAL FLOW: Does the sequence of events make logical sense?
3. CAUSAL RELATIONSHIPS: Are cause-and-effect claims supported by the context?
4. FACTUAL GAPS: Are there missing crucial details that would be present in real news?
5. IMPLAUSIBLE CLAIMS: Are any claims highly unlikely given the context?

Analysis Format:
- Contradictions Found: [List specific examples]
- Logical Issues: [Describe problems in reasoning]
- Missing Context: [What essential information is absent]
- Implausibility Score (1-10): [Rate how believable the overall narrative is]
- Verdict: [CONSISTENT/QUESTIONABLE/INCONSISTENT]
```

## Approach 2: Source and Attribution Analysis

### Core Concept
Examine how sources are cited, quoted, and referenced within the content context.

### Prompting Strategy
```
Analyze the source attribution patterns in this news article:

Content: "{NEWS_CONTENT}"

Evaluate:
1. SOURCE QUALITY: Are sources named, credible, and relevant?
2. QUOTE AUTHENTICITY: Do quotes sound natural and contextually appropriate?
3. EXPERT CREDIBILITY: Are claimed experts real and appropriately qualified?
4. VERIFICATION PATHS: Can claims be traced back to verifiable sources?
5. ATTRIBUTION PATTERNS: Does sourcing follow journalistic standards?

Output:
- Named Sources: [List and assess credibility]
- Anonymous Sources: [Appropriate usage or red flags]
- Quote Analysis: [Natural vs. fabricated indicators]  
- Verifiability: [HIGH/MEDIUM/LOW]
- Source Red Flags: [Specific concerns]
```

## Approach 3: Contextual Knowledge Verification

### Core Concept
Cross-reference claims against known facts and general world knowledge.

### Prompting Strategy
```
You have extensive knowledge up to your training cutoff. Analyze this news content for factual accuracy against known information:

News Content: "{NEWS_CONTENT}"

Fact-check against your knowledge:
1. VERIFIABLE FACTS: Which claims can you confirm or refute?
2. CONTEXTUAL ACCURACY: Do dates, locations, names align with known facts?
3. PLAUSIBILITY: Given your knowledge, how likely are the events described?
4. ANACHRONISMS: Are there any timeline inconsistencies?
5. DOMAIN EXPERTISE: Do technical/specialized claims make sense?

Assessment:
- Confirmed Facts: [List what you can verify]
- Disputed Claims: [What contradicts your knowledge]
- Unverifiable Claims: [What you cannot assess]
- Context Accuracy: [ACCURATE/PARTIALLY ACCURATE/INACCURATE]
- Knowledge Confidence: [How certain are you of this assessment]
```

## Approach 4: Narrative Structure Analysis

### Core Concept
Examine the storytelling patterns and narrative construction that differ between real and fake news.

### Prompting Strategy
```
Analyze the narrative structure of this news article:

Article: "{NEWS_CONTENT}"

Examine:
1. STORY ARC: Does it follow realistic news narrative patterns?
2. DETAIL DISTRIBUTION: Are details concentrated suspiciously in certain areas?
3. DRAMATIC ELEMENTS: Are there soap-opera-like dramatic peaks?
4. PACING: Does information unfold naturally or artificially?
5. RESOLUTION: Does the story conclude realistically?

Narrative Analysis:
- Structure Type: [Breaking news/Investigation/Feature/Opinion]
- Realism Score (1-10): [How naturally does the story unfold]
- Fabrication Indicators: [Artificial storytelling elements]
- Missing Elements: [What would real news include that's absent]
- Narrative Assessment: [NATURAL/ARTIFICIAL/FABRICATED]
```

## Approach 5: Claims Hierarchy and Evidence Chain

### Core Concept
Map the relationship between primary claims and supporting evidence within the content.

### Prompting Strategy
```
Map the evidence structure in this news article:

Content: "{NEWS_CONTENT}"

Create an evidence hierarchy:
1. PRIMARY CLAIMS: What are the main assertions?
2. SUPPORTING EVIDENCE: What evidence backs each claim?
3. EVIDENCE QUALITY: How strong is each piece of evidence?
4. LOGICAL CHAIN: Do evidence pieces logically support conclusions?
5. EVIDENCE GAPS: Where is evidence missing or weak?

Evidence Map:
- Main Claims: [List 3-5 primary assertions]
- Evidence per Claim: [What supports each claim]
- Evidence Strength: [STRONG/MODERATE/WEAK/ABSENT]
- Logical Gaps: [Where reasoning breaks down]
- Overall Support Level: [WELL-SUPPORTED/PARTIALLY SUPPORTED/UNSUPPORTED]
```

## Approach 6: Contextual Embedding Similarity

### Core Concept
Use semantic similarity to detect content that doesn't fit expected patterns for the claimed topic/domain.

### Prompting Strategy
```
Analyze whether this article's content matches its apparent topic and domain:

Article Content: "{NEWS_CONTENT}"
Claimed Topic/Category: "{TOPIC}"

Assess:
1. TOPIC COHERENCE: Does content consistently relate to the claimed topic?
2. DOMAIN APPROPRIATENESS: Does language/approach fit the news domain?
3. CONTEXTUAL DRIFT: Are there unexplained topic shifts?
4. SEMANTIC CONSISTENCY: Do all parts contribute to the main narrative?
5. RELEVANCE SCORE: How relevant is each paragraph to the core story?

Coherence Analysis:
- Topic Alignment: [How well content matches claimed focus]
- Content Unity: [Whether all parts belong together]
- Contextual Red Flags: [Sections that seem out of place]
- Semantic Score (1-10): [Overall coherence rating]
- Content Assessment: [COHERENT/MIXED/INCOHERENT]
```

## Approach 7: Multi-Step Reasoning Chain (this got implemenetd int he semantics.py)

### Core Concept
Use chain-of-thought prompting to reason through multiple aspects systematically.

### Prompting Strategy
```
Use step-by-step reasoning to assess this news article's authenticity:

Article: "{NEWS_CONTENT}"

Step 1 - Content Analysis:
What is this article claiming happened? List the key events/claims.

Step 2 - Plausibility Check:
For each claim, assess: Is this plausible given normal circumstances?

Step 3 - Evidence Evaluation:
What evidence is provided for each claim? Is it sufficient?

Step 4 - Context Assessment:
Does the broader context make sense? Are there missing pieces?

Step 5 - Consistency Review:
Are all parts of the story consistent with each other?

Step 6 - Reality Check:
Based on your knowledge, could this realistically happen as described?

Final Assessment:
- Reasoning Chain: [Your step-by-step analysis]
- Weak Links: [Where the story breaks down]
- Credibility Rating (1-10): [Overall assessment]
- Classification: [CREDIBLE/QUESTIONABLE/LIKELY FABRICATED]
```

