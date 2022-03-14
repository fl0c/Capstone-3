# Capstone Three - Text Summarization using Deep Learning
An abstractive summarization method to summarize review texts and news article texts was explored. 

The code was adapted from: https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/

## 1. Data
Two datasets were explored to train the deep learning model:
  1) Amazon Fine Foods Reviews, with over 500,000 reviews and summaries over 10 years
     Extracted Source: https://www.kaggle.com/snap/amazon-fine-food-reviews

  2) BBC articles with 2,225 text articles and human written summaries from 2004-2005
     Used in the paper of D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006; 
     All rights, including copyright, in the content of the original articles are owned by the BBC, http://mlg.ucd.ie/datasets/bbc.html 
     Extracted Source: https://www.kaggle.com/namrarehman/bbc-articles?select=BBCarticles_csv.csv

## 2. Method
### Training Phase
The Seq2Sqe Model is set up with encoders and decoders using keras, to handle the problem of different length input and output sequences. The model is trained on the training set to predict the reference sequence offset by one timestep. The encoder consists of an embedding layer built on the feature vocabulary, three sequential LSTM layers and dropout. The decoder consists of an embedding layer with the target vocab, a LSTM layer that takes the hidden state of the last encoder timestep as initial states. The output is a prediction for the next word in the sequence. An Global attention layer is added to consider all the hidden states of the encoder to derive the attended context vector. This is used to overcom the difficulty for the encoder to memorize long sequences into a fixed length vector. Start and end tokens are added to signal the beginning and end of a sentence. 

### Inference Phase
The trained model is tested on the testing set. The inference process consists of encoding the input sequence and initalizing the decoder with interal states of the encoder. The start token is passed as input to the decoder, which will decode for one timestep to output the next word with highest probability. The sampled word becomes the input to the decoder in the next timestep, updating the internal states with the current timestep. This is ended once the end token is generated or the maximum length of reference text has been reached.

## 3. Data Cleaning
The source files are cleaned to convert to lower case, remove words within parentheses, apostrophe ‘s, punctuation and stop words. Start and end tokens are added to signal the beginning and end of a sentence. 

## 4. EDA
For the Amazon Fine Foods dataset, the 95th percentile of summary data contained word lengths of 8 words. So the maximum word length for the summaries was set to 8, and the maximum word length of the text was set at 30 words to reduce computational requrements. 

![Word Lengths] (https://github.com/fl0c/Capstone-3/blob/65c2b1039bcab9a62e1039a1dfbf2a06acc4b513/eda.png)

Various maximum word length combinations for the text and summaries were explored for the BBC articles dataset. If articles of meaningful length were kept, the model training parameters ballooned very quickly. If the word lengths were reduced, the dataset size became very small. Due to computational limits, the model could not be trained on the BBC dataset.

## 5. Model Predictions and Performance Metrics
The model was trained on a training size of 4,771 and remaining 10% was for testing. Even with a short summary word lenth of 8 words, and 30 words for the text, the model had over 3 million trainable parameters. This was over 8 million for the BBC dataset. 

![Model Summary](https://github.com/fl0c/Capstone-3/blob/f97f1d85fe5827474704faa7f26087d60c9a2e57/model.jpg)

![Diagnostic Plot](https://github.com/fl0c/Capstone-3/blob/65c2b1039bcab9a62e1039a1dfbf2a06acc4b513/diagnostic_plot.png)

The model early stopping criteria was set at validation loss increases. This was reached at the 14th epoch with this dataset. 

A sample of the output is shown below. Many outputs generated 'great' as the only summary.

![Sample Output](https://github.com/fl0c/Capstone-3/blob/ac65eccab32feb3e1f5059682fb0759e2c0aea09/sample%20output.jpg)

Looking at the model metrics, the training set produced fairly low BLEU scores (precision) of 0.0151 and ROUGE-2 scores (recall) of 0.0197. The testing set BLEU score was 0.009 and ROUGE-2 score was 0.033. The ROUGE-1 F1 score was about 0.5, so the model is not much better than a 50-50 guess.

## Future Work
1) Make it work on more complex datasets
   By modifying Keras preprocessing tokenization filter, we can give unique tokens to      place names, figures and percentages. While this is useful, this increases the          vocabulary size of the model and number of trainable parameters. This also makes        generalization more difficult for summarization. This  attempt was excluded here due    to limited computation power. A cloud cluster method could be explored to handle        longer sentences and more complex tokens.
   
2) Train on more data
   While this seems obvious, but due to the computational limits, the model can take        very long to compile in some tested cases.

3) Bidirectional LSTM model to capture the context from both directions within a          sentence

4) Beam search decoder for the test sequence decoding instead of argmax
