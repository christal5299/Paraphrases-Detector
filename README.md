# Paraphrases-Detector

Quora is the one of the most popular platforms where people
actively ask questions on. However, due to a great amount of information on Quora,
people sometimes are not able to find the information they look for and would ask
the highly similar questions that were already solved or answered by others.
Therefore, the objective of this project is to recognize whether these pair questions
are paraphrased or not.

This project applied Natural Language Processing (NLP) algorithms
including Word Overlap algorithm (Jaccard Coefficient, Common Noun Similarity),
Word Order algorithm (Longest Common Subsequence, Maximum Common
Subsequence), N-Gram algorithm (Word N-Gram Overlap) to create features (the
similarity scores) for a logistic classification algorithm. The classifier then will help
find the best weights that predict the probability of the sentences being paraphrases
of each other.
