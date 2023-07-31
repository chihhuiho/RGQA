import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

MAX_GQA_LENGTH = 40

class MLP(nn.Module):
    def __init__(self, dims, use_weight_norm=True):
        """Simple utility class defining a fully connected network (multi-layer perceptron)"""
        super(MLP, self).__init__()

        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            if use_weight_norm:
                layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            else:
                layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # [bsz, *, dims[0]] --> [bsz, *, dims[-1]]
        return self.mlp(x)


class WordEmbedding(nn.Module):
    def __init__(self, ntoken, dim, dropout=0.0):
        """Initialize an Embedding Matrix with the appropriate dimensions --> Defines padding as last token in dict"""
        super(WordEmbedding, self).__init__()
        self.ntoken, self.dim = ntoken, dim

        self.emb = nn.Embedding(ntoken + 1, dim, padding_idx=ntoken)
        self.dropout = nn.Dropout(dropout)

    def load_embeddings(self, weights):
        """Set Embedding Weights from Numpy Array"""
        assert weights.shape == (self.ntoken, self.dim)
        self.emb.weight.data[: self.ntoken] = torch.from_numpy(weights)

    def forward(self, x):
        # x : [bsz, seq_len] --> [bsz, seq_len, emb_dim]
        return self.dropout(self.emb(x))


class QuestionEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, nlayers=1, bidirectional=False, dropout=0.0, rnn="GRU"):
        """Initialize the RNN Question Encoder with the appropriate configuration"""
        super(QuestionEncoder, self).__init__()
        self.in_dim, self.hidden, self.nlayers, self.bidirectional = in_dim, hidden_dim, nlayers, bidirectional
        self.rnn_type, self.rnn_cls = rnn, nn.GRU if rnn == "GRU" else nn.LSTM

        # Initialize RNN
        self.rnn = self.rnn_cls(
            self.in_dim, self.hidden, self.nlayers, bidirectional=self.bidirectional, dropout=dropout, batch_first=True
        )

    def forward(self, x):
        # x: [bsz, seq_len, emb_dim] --> ([bsz, seq_len, ndirections * hidden], [bsz, nlayers * ndirections, hidden])
        output, hidden = self.rnn(x)  # Note that Hidden Defaults to 0

        # If not Bidirectional --> Just Return last Output State
        if not self.bidirectional:
            # [bsz, hidden]
            return output[:, -1]

        # Otherwise, concat forward state for last element and backward state for first element
        else:
            # [bsz, 2 * hidden]
            f, b = output[:, -1, : self.hidden], output[:, 0, self.hidden :]
            return torch.cat([f, b], dim=1)


class Attention(nn.Module):
    def __init__(self, image_dim, question_dim, hidden, dropout=0.2, use_weight_norm=True):
        """Initialize the Attention Mechanism with the appropriate fusion operation"""
        super(Attention, self).__init__()

        # Attention w/ Product Fusion
        self.image_proj = MLP([image_dim, hidden], use_weight_norm=use_weight_norm)
        self.question_proj = MLP([question_dim, hidden], use_weight_norm=use_weight_norm)
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(hidden, 1), dim=None) if use_weight_norm else nn.Linear(hidden, 1)

    def forward(self, image_features, question_emb):
        # image_features: [bsz, k, image_dim = 2048]
        # question_emb: [bsz, question_dim]

        # Project both image and question embedding to hidden and repeat question_emb
        num_objs = image_features.size(1)
        image_proj = self.image_proj(image_features)
        question_proj = self.question_proj(question_emb).unsqueeze(1).repeat(1, num_objs, 1)

        # Fuse w/ Product
        image_question = image_proj * question_proj

        # Dropout Joint Representation
        joint_representation = self.dropout(image_question)

        # Compute Logits --> Softmax
        logits = self.linear(joint_representation)
        return nn.functional.softmax(logits, dim=1)


class GQABUTD(nn.Module):
    def __init__(self, num_answers, dictionary, dropout=True):
        super().__init__()
        self.num_answers = num_answers
        self.dictionary = dictionary

        # default hyperparameters
        # https://github.com/siddk/vqa-outliers/blob/main/src/models/butd.py
        self.emb_dim = 300
        self.emb_dropout = 0.
        self.hidden = 1024
        self.rnn_layers = 1
        self.bidirectional = False
        self.q_dropout = 0.
        self.rnn = "GRU"
        if dropout:
            self.attention_dropout = 0.2
            self.answer_dropout = 0.5
        else:
            self.attention_dropout = 0.
            self.answer_dropout = 0.
        self.weight_norm = True

        # https://github.com/siddk/vqa-outliers/blob/main/src/preprocessing/gqa/obj_dataset.py
        self.v_dim = 2048

        self.build_model()
    
    def build_model(self):
        # Build Word Embeddings (for Questions)
        self.w_emb = WordEmbedding(
            ntoken=self.dictionary.ntoken, dim=self.emb_dim, dropout=self.emb_dropout
        )

        # Build Question Encoder
        self.q_enc = QuestionEncoder(
            in_dim=self.emb_dim,
            hidden_dim=self.hidden,
            nlayers=self.rnn_layers,
            bidirectional=self.bidirectional,
            dropout=self.q_dropout,
            rnn=self.rnn,
        )

        # Build Attention Mechanism
        self.att = Attention(
            image_dim=self.v_dim + 4,
            question_dim=self.q_enc.hidden,
            hidden=self.hidden,
            dropout=self.attention_dropout,
            use_weight_norm=self.weight_norm,
        )

        # Build Projection Networks
        self.q_project = MLP([self.q_enc.hidden, self.hidden], use_weight_norm=self.weight_norm)
        self.img_project = MLP(
            [self.v_dim + 4, self.hidden], use_weight_norm=self.weight_norm
        )

        # Build Answer Classifier
        self.ans_classifier = nn.Sequential(
            *[
                weight_norm(nn.Linear(self.hidden, 2 * self.hidden), dim=None)
                if self.weight_norm
                else nn.Linear(self.hidden, 2 * self.hidden),
                nn.ReLU(),
                nn.Dropout(self.answer_dropout),
                weight_norm(nn.Linear(2 * self.hidden, self.num_answers), dim=None)
                if self.weight_norm
                else nn.Linear(2 * self.hidden, self.num_answers),
            ]
        )
    
    def tokenize(self, sentences):
        """Tokenize and Front-Pad the Questions in the Dataset"""
        batch_tokens = []
        for sentence in sentences:
            tokens = self.dictionary.tokenize(sentence, False)
            tokens = tokens[:MAX_GQA_LENGTH]
            if len(tokens) < MAX_GQA_LENGTH:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (MAX_GQA_LENGTH - len(tokens))
                tokens = padding + tokens
            assert len(tokens) == MAX_GQA_LENGTH, "Tokenized & Padded Question != Max Length!"
            batch_tokens.append(tokens)
        batch_tokens = torch.from_numpy(np.asarray(batch_tokens))
        return batch_tokens
    
    def forward(self, feat, pos, sent, attention=False):
        # tokenize
        toks = self.tokenize(sent).cuda()

        # Embed and Encode Question --> [bsz, q_hidden]
        w_emb = self.w_emb(toks)
        q_enc = self.q_enc(w_emb)

        # Create new Image Features --> KEY POINT: Concatenate Spatial Features!
        image_features = torch.cat([feat, pos], dim=2)

        # Attend over Image Features and Create Image Encoding --> [bsz, img_hidden]
        att = self.att(image_features, q_enc)
        img_enc = (image_features * att).sum(dim=1)

        # Project Image and Question Features --> [bsz, hidden]
        q_repr = self.q_project(q_enc)
        img_repr = self.img_project(img_enc)

        # Merge
        joint_repr = q_repr * img_repr

        # Compute and Return Logits
        if not attention:
            return self.ans_classifier(joint_repr)
        else:
            return self.ans_classifier(joint_repr), att


class GQABUTD_branched(nn.Module):
    def __init__(self, num_answers, dictionary, dropout=True):
        super().__init__()
        self.num_answers = num_answers
        self.dictionary = dictionary

        # default hyperparameters
        # https://github.com/siddk/vqa-outliers/blob/main/src/models/butd.py
        self.emb_dim = 300
        self.emb_dropout = 0.
        self.hidden = 1024
        self.rnn_layers = 1
        self.bidirectional = False
        self.q_dropout = 0.
        self.rnn = "GRU"
        if dropout:
            self.attention_dropout = 0.2
            self.answer_dropout = 0.5
        else:
            self.attention_dropout = 0.
            self.answer_dropout = 0.
        self.weight_norm = True

        # https://github.com/siddk/vqa-outliers/blob/main/src/preprocessing/gqa/obj_dataset.py
        self.v_dim = 2048

        self.build_model()
    
    def build_model(self):
        # Build Word Embeddings (for Questions)
        self.w_emb = WordEmbedding(
            ntoken=self.dictionary.ntoken, dim=self.emb_dim, dropout=self.emb_dropout
        )

        # Build Question Encoder
        self.q_enc = QuestionEncoder(
            in_dim=self.emb_dim,
            hidden_dim=self.hidden,
            nlayers=self.rnn_layers,
            bidirectional=self.bidirectional,
            dropout=self.q_dropout,
            rnn=self.rnn,
        )

        # Build Attention Mechanism
        self.att = Attention(
            image_dim=self.v_dim + 4,
            question_dim=self.q_enc.hidden,
            hidden=self.hidden,
            dropout=self.attention_dropout,
            use_weight_norm=self.weight_norm,
        )

        # Build Projection Networks
        self.q_project = MLP([self.q_enc.hidden, self.hidden], use_weight_norm=self.weight_norm)
        self.img_project = MLP(
            [self.v_dim + 4, self.hidden], use_weight_norm=self.weight_norm
        )

        # Build Answer Classifier
        self.ans_classifier = nn.Sequential(
            *[
                weight_norm(nn.Linear(self.hidden, 2 * self.hidden), dim=None)
                if self.weight_norm
                else nn.Linear(self.hidden, 2 * self.hidden),
                nn.ReLU(),
                nn.Dropout(self.answer_dropout),
                weight_norm(nn.Linear(2 * self.hidden, self.num_answers), dim=None)
                if self.weight_norm
                else nn.Linear(2 * self.hidden, self.num_answers),
            ]
        )

        # Build Confidence Branch
        self.conf = nn.Sequential(
            *[
                weight_norm(nn.Linear(self.hidden, 2 * self.hidden), dim=None)
                if self.weight_norm
                else nn.Linear(self.hidden, 2 * self.hidden),
                nn.ReLU(),
                nn.Dropout(self.answer_dropout),
                weight_norm(nn.Linear(2 * self.hidden, 1), dim=None)
                if self.weight_norm
                else nn.Linear(2 * self.hidden, 1),
            ]
        )
    
    def tokenize(self, sentences):
        """Tokenize and Front-Pad the Questions in the Dataset"""
        batch_tokens = []
        for sentence in sentences:
            tokens = self.dictionary.tokenize(sentence, False)
            tokens = tokens[:MAX_GQA_LENGTH]
            if len(tokens) < MAX_GQA_LENGTH:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (MAX_GQA_LENGTH - len(tokens))
                tokens = padding + tokens
            assert len(tokens) == MAX_GQA_LENGTH, "Tokenized & Padded Question != Max Length!"
            batch_tokens.append(tokens)
        batch_tokens = torch.from_numpy(np.asarray(batch_tokens))
        return batch_tokens
    
    def forward(self, feat, pos, sent, attention=False):
        # tokenize
        toks = self.tokenize(sent).cuda()

        # Embed and Encode Question --> [bsz, q_hidden]
        w_emb = self.w_emb(toks)
        q_enc = self.q_enc(w_emb)

        # Create new Image Features --> KEY POINT: Concatenate Spatial Features!
        image_features = torch.cat([feat, pos], dim=2)

        # Attend over Image Features and Create Image Encoding --> [bsz, img_hidden]
        att = self.att(image_features, q_enc)
        img_enc = (image_features * att).sum(dim=1)

        # Project Image and Question Features --> [bsz, hidden]
        q_repr = self.q_project(q_enc)
        img_repr = self.img_project(img_enc)

        # Merge
        joint_repr = q_repr * img_repr

        # Compute and Return Logits
        if not attention:
            return self.ans_classifier(joint_repr), self.conf(joint_repr)
        else:
            return self.ans_classifier(joint_repr), self.conf(joint_repr), att


class GQABUTD_maha(nn.Module):
    def __init__(self, num_answers, dictionary, dropout=True):
        super().__init__()
        self.num_answers = num_answers
        self.dictionary = dictionary

        # default hyperparameters
        # https://github.com/siddk/vqa-outliers/blob/main/src/models/butd.py
        self.emb_dim = 300
        self.emb_dropout = 0.
        self.hidden = 1024
        self.rnn_layers = 1
        self.bidirectional = False
        self.q_dropout = 0.
        self.rnn = "GRU"
        if dropout:
            self.attention_dropout = 0.2
            self.answer_dropout = 0.5
        else:
            self.attention_dropout = 0.
            self.answer_dropout = 0.
        self.weight_norm = True

        # https://github.com/siddk/vqa-outliers/blob/main/src/preprocessing/gqa/obj_dataset.py
        self.v_dim = 2048

        self.build_model()
    
    def build_model(self):
        # Build Word Embeddings (for Questions)
        self.w_emb = WordEmbedding(
            ntoken=self.dictionary.ntoken, dim=self.emb_dim, dropout=self.emb_dropout
        )

        # Build Question Encoder
        self.q_enc = QuestionEncoder(
            in_dim=self.emb_dim,
            hidden_dim=self.hidden,
            nlayers=self.rnn_layers,
            bidirectional=self.bidirectional,
            dropout=self.q_dropout,
            rnn=self.rnn,
        )

        # Build Attention Mechanism
        self.att = Attention(
            image_dim=self.v_dim + 4,
            question_dim=self.q_enc.hidden,
            hidden=self.hidden,
            dropout=self.attention_dropout,
            use_weight_norm=self.weight_norm,
        )

        # Build Projection Networks
        self.q_project = MLP([self.q_enc.hidden, self.hidden], use_weight_norm=self.weight_norm)
        self.img_project = MLP(
            [self.v_dim + 4, self.hidden], use_weight_norm=self.weight_norm
        )

        # Build Answer Classifier
        self.ans_classifier = nn.Sequential(
            *[
                weight_norm(nn.Linear(self.hidden, 2 * self.hidden), dim=None)
                if self.weight_norm
                else nn.Linear(self.hidden, 2 * self.hidden),
                nn.ReLU(),
                nn.Dropout(self.answer_dropout),
                weight_norm(nn.Linear(2 * self.hidden, self.num_answers), dim=None)
                if self.weight_norm
                else nn.Linear(2 * self.hidden, self.num_answers),
            ]
        )
    
    def tokenize(self, sentences):
        """Tokenize and Front-Pad the Questions in the Dataset"""
        batch_tokens = []
        for sentence in sentences:
            tokens = self.dictionary.tokenize(sentence, False)
            tokens = tokens[:MAX_GQA_LENGTH]
            if len(tokens) < MAX_GQA_LENGTH:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (MAX_GQA_LENGTH - len(tokens))
                tokens = padding + tokens
            assert len(tokens) == MAX_GQA_LENGTH, "Tokenized & Padded Question != Max Length!"
            batch_tokens.append(tokens)
        batch_tokens = torch.from_numpy(np.asarray(batch_tokens))
        return batch_tokens
    
    def forward(self, feat, pos, sent, attention=False):
        # tokenize
        toks = self.tokenize(sent).cuda()

        # Embed and Encode Question --> [bsz, q_hidden]
        w_emb = self.w_emb(toks)
        q_enc = self.q_enc(w_emb)

        # Create new Image Features --> KEY POINT: Concatenate Spatial Features!
        image_features = torch.cat([feat, pos], dim=2)

        # Attend over Image Features and Create Image Encoding --> [bsz, img_hidden]
        att = self.att(image_features, q_enc)
        img_enc = (image_features * att).sum(dim=1)

        # Project Image and Question Features --> [bsz, hidden]
        q_repr = self.q_project(q_enc)
        img_repr = self.img_project(img_enc)

        # Merge
        joint_repr = q_repr * img_repr

        # Compute and Return Logits
        if not attention:
            return self.ans_classifier(joint_repr), joint_repr
        else:
            return self.ans_classifier(joint_repr), att