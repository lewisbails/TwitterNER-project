# Cross-domain evaluation of a non-neural named entity tagger

Named entity recognition for user-generated text is difficult for a couple of reasons. The first being that there's just not that much labelled data out there. The second being that it's high variance with long-tail entities and spelling/grammatical mistakes galore. We hypothesise that we can bump up the number of training examples by mixing the existing annotated user-generated data with more formal text (from newswire), which is much more abundant.

The repo contains all the code for scraping tweets for a test set, labelling them with spaCy (to go to Doccano), restructuring the doccano JSON1, and getting the feature vectors for each token to go into a csv (for modelling later).  

The modelling part of the code is a notebook that sets up the grid searches etc. and find the best model. Analysis on the test set (of tweets) is at the bottom of the notebook.

The report discusses the process of annotating the tweets, modelling, and analysing the results on a per-attribute level.
