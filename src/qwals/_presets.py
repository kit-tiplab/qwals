"""Task-specific feature presets from the qWALS paper (Appendix A).

These are the WALS feature subsets that the leave-one-feature-out (LOFO)
optimisation procedure converged on for four cross-lingual NLP tasks:

- ``"abusive"``  — abusive language identification (53 features, ρ ≈ −0.82)
- ``"sentiment"`` — sentiment analysis           (21 features, ρ ≈ −0.80)
- ``"ner"``      — named entity recognition     (63 features, ρ ≈ −0.81)
- ``"dep"``      — dependency parsing           (75 features, ρ ≈ −0.99)

Reference: Eronen, J. et al. (2026). "Language Models Are Polyglots:
Language Similarity Predicts Cross-Lingual Transfer Learning Performance."
Mach. Learn. Knowl. Extr. 8(3), 65, Appendix A — Table A1.

Use via :meth:`qwals.QwalsCalculator.use_features`:

    calc.use_features("dep")            # apply DEP-optimised feature set
    calc.distance("Polish", "English")  # now uses only those 75 features
    calc.reset_features()               # back to using every feature

Notes
-----
The exact feature-name strings here mirror the canonical WALS
``Parameter_name`` values as cleaned by qwals' loader (commas stripped,
whitespace collapsed). A handful of names in the paper's appendix appear
to drop a trailing apostrophe (e.g. "When' Clauses" → "'When' Clauses");
qwals normalises both forms to the same key so the preset still applies.
Any name in a preset that is not present in the loaded WALS data is
silently skipped — useful for small synthetic test fixtures.
"""
from __future__ import annotations


# ---------- Abusive language identification (paper: 53 features) ----------
ABUSIVE: tuple[str, ...] = (
    "Zero Copula for Predicate Nominals",
    "Expression of Pronominal Subjects",
    "Semantic Distinctions of Evidentiality",
    "Inflectional Synthesis of the Verb",
    "Suppletion in Imperatives and Hortatives",
    "SNegVO Order",
    "Glottalized Consonants",
    "Passive Constructions",
    "Situational Possibility",
    "Purpose Clauses",
    "Definite Articles",
    "Reason Clauses",
    "Reciprocal Constructions",
    "The Associative Plural",
    "Comitatives and Instrumentals",
    "Action Nominal Constructions",
    "The Velar Nasal",
    "Absence of Common Consonants",
    "Order of Person Markers on the Verb",
    "Postnominal relative clauses",
    "Distributive Numerals",
    "Postverbal Negative Morphemes",
    "Order of Object Oblique and Verb",
    "Presence of Uncommon Consonants",
    "SONegV Order",
    "Person Marking on Adpositions",
    "Number of Cases",
    "Order of Demonstrative and Noun",
    "Verbal Number and Suppletion",
    "Coding of Nominal Plurality",
    "Utterance Complement Clauses",
    "The Position of Negative Morphemes in SOV Languages",
    "Perfective/Imperfective Aspect",
    "The Prohibitive",
    "Symmetric and Asymmetric Standard Negation",
    "Subtypes of Asymmetric Standard Negation",
    "Position of Case Affixes",
    "SVNegO Order",
    "Negative Indefinite Pronouns and Predicate Negation",
    "Syncretism in Verbal Person/Number Marking",
    "Order of Numeral and Noun",
    "NegSVO Order",
    "Coding of Evidentiality",
    "M-T Pronouns",
    "Voicing in Plosives and Fricatives",
    "Para-Linguistic Usages of Clicks",
    "Imperative-Hortative Systems",
    "'When' Clauses",
    "Overlap between Situational and Epistemic Modal Marking",
    "Reduplication",
    "Voicing and Gaps in Plosive Systems",
    "Vowel Quality Inventories",
    "Adjectives without Nouns",
)


# ---------- Sentiment analysis (paper: 21 features) ----------
SENTIMENT: tuple[str, ...] = (
    "Number of Genders",
    "SNegVO Order",
    "Glottalized Consonants",
    "Position of Negative Word With Respect to Subject Object and Verb",
    "Purpose Clauses",
    "Action Nominal Constructions",
    "Postnominal relative clauses",
    "Fixed Stress Locations",
    "Order of Relative Clause and Noun",
    "Sex-based and Non-sex-based Gender Systems",
    "Epistemic Possibility",
    "Position of Tense-Aspect Affixes",
    "Indefinite Articles",
    "Position of Pronominal Possessive Affixes",
    "Systems of Gender Assignment",
    "NegSVO Order",
    "Voicing in Plosives and Fricatives",
    "'When' Clauses",
    "Reduplication",
    "Voicing and Gaps in Plosive Systems",
    "Vowel Quality Inventories",
)


# ---------- Named entity recognition (paper: 63 features) ----------
NER: tuple[str, ...] = (
    "Gender Distinctions in Independent Personal Pronouns",
    "Zero Copula for Predicate Nominals",
    "Order of Degree Word and Adjective",
    "Number of Genders",
    "Tone",
    "Nominal and Locational Predication",
    "Situational Possibility",
    "Uvular Consonants",
    "Third Person Pronouns and Demonstratives",
    "Consonant Inventories",
    "Pronominal and Adnominal Demonstratives",
    "SOVNeg Order",
    "Minor morphological means of signaling negation",
    "The Associative Plural",
    "Finger and Hand",
    "Action Nominal Constructions",
    "Multiple Negative Constructions in SVO Languages",
    "Order of Adverbial Subordinator and Clause",
    "Postnominal relative clauses",
    "'Want' Complement Subjects",
    "Order of Object Oblique and Verb",
    "Prefixing vs. Suffixing in Inflectional Morphology",
    "M in Second Person Singular",
    "Genitives Adjectives and Relative Clauses",
    "Presence of Uncommon Consonants",
    "Position of negative words relative to beginning and end of clause and with respect to adjacency to verb",
    "SONegV Order",
    "Order of Adjective and Noun",
    "Conjunctions and Universal Quantifiers",
    "Predicative Adjectives",
    "Fixed Stress Locations",
    "Order of Relative Clause and Noun",
    "Obligatory Possessive Inflection",
    "Red and Yellow",
    "Order of Demonstrative and Noun",
    "Verbal Number and Suppletion",
    "Noun Phrase Conjunction",
    "Coding of Nominal Plurality",
    "Sex-based and Non-sex-based Gender Systems",
    "Green and Blue",
    "Periphrastic Causative Constructions",
    "Order of Adposition and Noun Phrase",
    "Preverbal Negative Morphemes",
    "The Position of Negative Morphemes in SOV Languages",
    "Perfective/Imperfective Aspect",
    "The Prohibitive",
    "Possessive Classification",
    "Position of Pronominal Possessive Affixes",
    "Order of Subject and Verb",
    "Nonperiphrastic Causative Constructions",
    "Lateral Consonants",
    "NegSOV Order",
    "Position of Case Affixes",
    "Systems of Gender Assignment",
    "Productivity of the Antipassive Construction",
    "Negative Indefinite Pronouns and Predicate Negation",
    "Order of Numeral and Noun",
    "NegSVO Order",
    "Voicing in Plosives and Fricatives",
    "Imperative-Hortative Systems",
    "Order of Negative Morpheme and Verb",
    "Reduplication",
    "Negative Morphemes",
)


# ---------- Dependency parsing (paper: 75 features) ----------
DEP: tuple[str, ...] = (
    "Zero Copula for Predicate Nominals",
    "Order of Degree Word and Adjective",
    "Number of Genders",
    "Tone",
    "Glottalized Consonants",
    "Relativization on Subjects",
    "Passive Constructions",
    "Situational Possibility",
    "Uvular Consonants",
    "Third Person Pronouns and Demonstratives",
    "Consonant Inventories",
    "Purpose Clauses",
    "Pronominal and Adnominal Demonstratives",
    "SOVNeg Order",
    "Minor morphological means of signaling negation",
    "Reason Clauses",
    "Finger and Hand",
    "Comitatives and Instrumentals",
    "Order of Adverbial Subordinator and Clause",
    "Absence of Common Consonants",
    "Order of Person Markers on the Verb",
    "Postnominal relative clauses",
    "'Want' Complement Subjects",
    "Order of Object Oblique and Verb",
    "Prefixing vs. Suffixing in Inflectional Morphology",
    "M in Second Person Singular",
    "Genitives Adjectives and Relative Clauses",
    "SONegV Order",
    "Order of Adjective and Noun",
    "Person Marking on Adpositions",
    "Inclusive/Exclusive Distinction in Independent Pronouns",
    "Vowel Nasalization",
    "Number of Possessive Nouns",
    "Predicative Adjectives",
    "Locus of Marking in Possessive Noun Phrases",
    "Order of Relative Clause and Noun",
    "Obligatory Possessive Inflection",
    "Red and Yellow",
    "Order of Demonstrative and Noun",
    "Noun Phrase Conjunction",
    "Coding of Nominal Plurality",
    "Sex-based and Non-sex-based Gender Systems",
    "Green and Blue",
    "Ordinal Numerals",
    "Order of Adposition and Noun Phrase",
    "Preverbal Negative Morphemes",
    "Epistemic Possibility",
    "Predicative Possession",
    "Utterance Complement Clauses",
    "Perfective/Imperfective Aspect",
    "The Prohibitive",
    "Possessive Classification",
    "Position of Pronominal Possessive Affixes",
    "Comparative Constructions",
    "Nonperiphrastic Causative Constructions",
    "Lateral Consonants",
    "Occurrence of Nominal Plurality",
    "Position of Case Affixes",
    "SVNegO Order",
    "Systems of Gender Assignment",
    "Productivity of the Antipassive Construction",
    "Order of Numeral and Noun",
    "Hand and Arm",
    "NegSVO Order",
    "Voicing in Plosives and Fricatives",
    "Position of Interrogative Phrases in Content Questions",
    "Imperative-Hortative Systems",
    "Order of Negative Morpheme and Verb",
    "Overlap between Situational and Epistemic Modal Marking",
    "Reduplication",
    "Voicing and Gaps in Plosive Systems",
    "Vowel Quality Inventories",
    "Plurality in Independent Personal Pronouns",
    "Negative Morphemes",
    "Adjectives without Nouns",
)


TASK_FEATURES: dict[str, tuple[str, ...]] = {
    "abusive": ABUSIVE,
    "sentiment": SENTIMENT,
    "ner": NER,
    "dep": DEP,
}

# Public list of task names — handy for CLI argparse choices and ``__all__``.
TASKS: tuple[str, ...] = tuple(TASK_FEATURES.keys())
