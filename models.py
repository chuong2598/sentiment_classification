import torch.nn as nn


class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, word_embeddings=None):
        """
        :param vocab_size: the number of different embeddings to make (need one embedding for every unique word).
        :param embedding_dim: the dimension of each embedding vector.
        :param num_classes: the number of target classes.
        :param word_embeddings: optional pre-trained word embeddings. If not given word embeddings are trained from
        random initialization. If given then provided word_embeddings are used and the embeddings are not trained.
        """
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size,embedding_dim, padding_idx=0)
        if word_embeddings is not None:
            self.embeddings = self.embeddings.from_pretrained(word_embeddings, freeze=True, padding_idx=0)
        self.W = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        :param x: a LongTensor of shape [batch_size, max_sequence_length]. Each row is one sequence (movie review),
        the i'th element in a row is the (integer) ID of the i'th token in the original text.
        :return: a FloatTensor of shape [batch_size, num_classes]. Predicted class probabilities for every sequence
        in the batch.
        """

        # Get representation of each word in each sentence
        embedded_words = self.embeddings(x)
        # Aggregate to get representation of each sentence.
        embedded_sentence_avg = embedded_words.mean(dim=1)
        # Use the embedding sentence as an input for a linear model to predict logic for each class
        y_predict = self.W(embedded_sentence_avg)
        return y_predict
