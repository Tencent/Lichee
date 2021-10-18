# -*- coding: utf-8 -*-
import torch

from lichee.module.torch.layer.normalization import LayerNorm
from lichee.utils import common


class WordEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, w2v_vectors_path=None, init_outer_w2v=True, freeze=True):
        super(WordEmbedding, self).__init__()

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

        if init_outer_w2v:
            vector_matrix = common.load_word2vec_vector(w2v_vectors_path, embedding_dim)
            assert vector_matrix.shape[1] == embedding_dim
            vector_matrix = torch.tensor(vector_matrix)
            self.embedding.weight.data.copy_(vector_matrix)

        self.embedding.weight.requires_grad = not freeze

    def forward(self, token_ids):
        return self.embedding(token_ids)


class BERTEmbedding(torch.nn.Module):
    """
    Construct the embedding module with config.

    Params:
    -------
    cfg: Dict:
        config of model
    """

    def __init__(self, cfg):
        super(BERTEmbedding, self).__init__()
        # Construct the embedding module from word, position and token_type embeddings.
        self.word_embeddings = torch.nn.Embedding(
            cfg["CONFIG"]["VOCAB_SIZE"], cfg["CONFIG"]["HIDDEN_SIZE"])
        self.position_embeddings = torch.nn.Embedding(
            cfg["CONFIG"]["MAX_POSITION_EMBEDDINGS"], cfg["CONFIG"]["HIDDEN_SIZE"])
        self.token_type_embeddings = torch.nn.Embedding(
            cfg["CONFIG"]["TYPE_VOCAB_SIZE"], cfg["CONFIG"]["HIDDEN_SIZE"])

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(cfg["CONFIG"]["HIDDEN_SIZE"])
        self.dropout = torch.nn.Dropout(cfg["CONFIG"]["HIDDEN_DROPOUT_PROB"])

    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertEmbeddingMixGrained(torch.nn.Module):
    """
    Construct the mix-grained embedding module with config.

    Params:
    -------
    cfg: Dict:
        config of model
    """

    def __init__(self, cfg):
        super(BertEmbeddingMixGrained, self).__init__()
        # Construct the embedding module from word, coarse, position and token_type embeddings.
        self.word_embeddings = torch.nn.Embedding(
            cfg["CONFIG"]['VOCAB_SIZE'], cfg["CONFIG"]['CONFIG.EMBEDDING_SIZE'])
        self.coarse_embeddings = torch.nn.Embedding(
            cfg["CONFIG"]['COARSE_VOCAB_SIZE'], cfg["CONFIG"]['COARSE_EMBEDDING_SIZE'])
        self.position_embeddings = torch.nn.Embedding(
            cfg["CONFIG"]["MAX_POSITION_EMBEDDINGS"], cfg["CONFIG"]["HIDDEN_SIZE"])
        self.token_type_embeddings = torch.nn.Embedding(
            cfg["CONFIG"]['TYPE_VOCAB_SIZE'], cfg["CONFIG"]["HIDDEN_SIZE"])

        self.LayerNorm = LayerNorm(cfg["CONFIG"]["HIDDEN_SIZE"])
        self.dropout = torch.nn.Dropout(cfg["CONFIG"]["HIDDEN_DROPOUT_PROB"])

        self.mix_mode = cfg["CONFIG"]["MIX_MODE"]
        print('Embeddinng mix mode', self.mix_mode)

    def forward(self, mix_input_ids, token_type_ids):
        input_ids = mix_input_ids[0]
        coarse_input_ids = mix_input_ids[1]

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        word_embeddings = self.word_embeddings(input_ids)
        coarse_embeddings = self.coarse_embeddings(coarse_input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if self.mix_mode == 'cat':
            mix_embeddings = torch.cat([word_embeddings, coarse_embeddings], dim=-1)
        else:
            mix_embeddings = word_embeddings + coarse_embeddings

        embeddings = mix_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
