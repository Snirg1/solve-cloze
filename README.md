# solve-cloze
NLP - statistical language models – solving a cloze

This task deals with solving a cloze: a short text where certain words were removed, the goal is to fill in the missing items from a given list. 

### Input:
(1) a large well-formed English corpus
(2) a file with cloze: a short text where removed words were replaced with “__________“
(3) a file with words removed from the text (in a random order) – use these words to solve the cloze
(4) a file with top-50K most frequent English words in the given Wikipedia sample – that’s the lexicon to restrict the solution to, in order to ensure reasonable runtime

### Output:
print to the console a list of words in the order they are assigned to placeholders in (2)
For example, this input paragraph:

The wooden bridge, dating from the Middle Ages, across the Aare was destroyed by floods three __________ in thirty years, and was replaced with a steel suspension __________ in 1851. This was replaced by a __________ bridge in 1952. The city was linked up to the Swiss Central Railway in 1856.
And the list of words: bridge, concrete, times   

The output should be (a list with words):
[‘times’, ‘bridge’, ‘concrete’]
