# Deep-Learning-Hackathon-Techsoc-2021

Achieved highest accuracy in the competition and finished 2nd overall

# Approach

Drive link containing the weights and the csv files: [link](https://drive.google.com/drive/folders/16B7f3Uz749Yhh93kKAq5SB1afJluyVEi?usp=sharing)

Inference notebook: [inference](inference_notebook.ipynb)
- # Preprocessing
    - [Notebook](notebooks/preprocessing.ipynb)
    - First I created a new column('info') which contained text from both content and title
    - For cleaning the data, I used pre trained glove embeddings. Basically, I checked how much of the data is available in glove and created a list of the words that aren't there along with the number of times they occur in the corpus.
    - Next, I cleaned the most occurring unknown words ( these were mostly extra spaces and "\t")
    - Another observation that I made was that some products are repeated multiple times and some products have also been assigned multiple categories.
    - I deleted all the duplicate rows but kept different rows of same product if they were assigned to different categories

- # Models  
    
    notebook for non-pre trained models: [notebook](notebooks/lstm_training.ipynb)

    - ## LSTM: (Test accuracy of 0.32)
        - [script](scripts/lstm.py)
        - First layer of this model is the embedding layer. 
        - Next, the embeddings of the title and content are passed through separate LSTMs. Let's call this output as the representations of the title and content.
        - Next, I took the mean of the representations along dimension 1 (mean of the outputs at all timesteps), rather than just using the last output because, the content sequence was very long and taking just the last step generally means that the information from the beginning is generally lost.
        - After taking the mean, I concatenated the two representations and passed it through a fully connected layer to obtain the probabilities. 
    
    - ## LSTM with attention (Test accuracy of 0.35)
        - [script](scripts/lstm-attention.py)
        - The architecture is similar to the LSTM one, but I added an additional attention layer.
        - Instead of taking the mean for all the steps in the content representation, I applied attention at each step of the content representation with the mean title representation. (reason being: We check which part of the content is relevant to the title and give preference to that)
        - It performed slightly better than just lstm.

    - ## LSTM followed by CNN (Test accuracy of 0.39)   
        - [script](scripts/lstm-cnn.py)
        - used the "info" column rather than having two inputs of title and content.
        - We first pass the embeddings through the lstm layer to obtain the representations.
        - Next, instead of doing the mean or using attention, we use a 1D CNN followed by pooling to obtain the final representation which is passed through the fully connected layer.
        - It performed better than lstm + attention
    
    - ## Kim CNN (test accuracy of 0.42)
        - [script](scripts/kim-cnn.py)
        - Looking at the increase of performance by using a CNN after the LSTM, i thought of completely using CNNs.
        - I used the kim cnn model with 3 filters.
        - This outperformed LSTM + CNN

    - ## BERT, RoBerta, XLNet, GPT2
        - Notebooks - [bert](notebooks/bert_training.ipynb), [xlnet](notebooks/xlnet_training.ipynb), [roberta](notebooks/roberta_training.ipynb), [gpt2](notebooks/gpt2_training.ipynb) 
        - After trying the above architectures, I tried fine tuning pre trained models.
        - I used a fully connected layer on top of the pre-trained model and only trained this layer.
        - Bert and roberta gave a test accuracy of 0.49, XLNet gave a test accuracy of 0.48 and GPT2 gave a test accuracy of 0.46.

    - ## Final Ensembling ( test accuracy of 0.509)
        - [notebook](notebooks/ensembling.ipynb)
        - The pretrained models clearly outperformed all the other architectures. Ensembling them with the other models actually reduced the performance.
        - Even using GPT2 in the ensemble reduced the performance.
        - Thus, in the final ensemble, I used Bert, Roberta and XLNet which gave best results when all 3 were used, which shows that each of them learnt slightly different methods to predict.
        - I added the probabilities predicted by each of the models and chose the highest probable one.

