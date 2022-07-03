# Criminal Offence Classifier
## Domain -  Information Retrieval

### 1. data
• description of closed lawsuits of Indian Courts\
• labels have 20 charges with its detail according to Indian Penal Code\
• texts are annotated as set of fact descriptions with sentence and document level offence

### 2. ptembs
• word2vec.kv: pre-trained key vectors of word2vec model with embedding dimension = 128

### 3. model ( COClassifier )
• treats text as a hierarchy of sentences and words\
• constructs intermediate sentence embeddings for each sentence\
• constructs document embedding for the entire text

### 4. training
• model is trained in minibatches and learning rate is modified as per F1 score\
• multi-task learning using GRU and RNNs to optimize both the sentence-level and document-level losses simultaneously