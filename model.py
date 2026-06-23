import torch
import torch.nn as nn
import nltk
import torchtext.vocab

# Match the notebook's constants
START = '<s>'
END = '</s>'
MAX_LEN = 50
HIDDEN_SIZE = 128
DEVICE = 'cpu'  # local CPU


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def initialize_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

    def forward(self, input_sequence, hidden_state):
        embedded = self.embedding(input_sequence)
        output, hidden_state = self.gru(embedded, hidden_state)
        return hidden_state


class Decoder(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, prev_word, hidden_state):
        embedded = self.embedding(prev_word)
        output, hidden_state = self.gru(embedded, hidden_state)
        output = self.softmax(self.out(output[0]))
        return output, hidden_state


def _ensure_nltk():
    for pkg in ('punkt', 'punkt_tab'):
        try:
            nltk.data.find(f'tokenizers/{pkg}')
        except LookupError:
            nltk.download(pkg)


def load_data(file_pathname):
    """Identical preprocessing to the notebook so vocab indices match the weights."""
    _ensure_nltk()
    src, tgt = [], []
    with open(file_pathname, encoding='utf-8') as inf:
        next(inf)  # skip header
        for line in inf:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            mr_str, ref_str = parts[0], parts[1]
            mr_tokens = [t.strip() for t in mr_str.split(',')]
            src.append(mr_tokens)
            ref_tokens = nltk.word_tokenize(ref_str.lower())
            if len(ref_tokens) > MAX_LEN - 2:
                ref_tokens = ref_tokens[:MAX_LEN - 2]
            tgt.append([START] + ref_tokens + [END])
    return src, tgt


def build_vocabs(train_path='train.txt'):
    train_src, train_tgt = load_data(train_path)
    src_vocab = torchtext.vocab.build_vocab_from_iterator(iter(train_src))
    tgt_vocab = torchtext.vocab.build_vocab_from_iterator(iter(train_tgt))
    return src_vocab, tgt_vocab


def load_models(src_vocab, tgt_vocab,
                encoder_path='hw5.encoder', decoder_path='hw5.decoder'):
    encoder = Encoder(len(src_vocab), HIDDEN_SIZE).to(DEVICE)
    decoder = Decoder(HIDDEN_SIZE, len(tgt_vocab)).to(DEVICE)
    # map_location=cpu lets GPU-trained weights load on a CPU machine
    encoder.load_state_dict(torch.load(encoder_path, map_location=DEVICE))
    decoder.load_state_dict(torch.load(decoder_path, map_location=DEVICE))
    encoder.eval()
    decoder.eval()
    return encoder, decoder


def generate(input_tokens, src_vocab, tgt_vocab, encoder, decoder, max_length=MAX_LEN):
    """input_tokens: list of MR strings, e.g. ['name[The Eagle]', 'food[French]']."""
    src_stoi = src_vocab.get_stoi()
    tgt_stoi = tgt_vocab.get_stoi()
    tgt_itos = tgt_vocab.get_itos()

    # Skip any MR token not seen in training (avoids KeyError on unknown values)
    indices = [src_stoi[w] for w in input_tokens if w in src_stoi]
    if not indices:
        return [], [w for w in input_tokens if w not in src_stoi]

    unknown = [w for w in input_tokens if w not in src_stoi]
    input_sequence = torch.tensor(indices, device=DEVICE).unsqueeze(1)

    with torch.no_grad():
        encoder_hidden = encoder.initialize_hidden_state()
        encoder_hidden = encoder(input_sequence, encoder_hidden)

        decoder_hidden = encoder_hidden
        prev_word = torch.tensor([[tgt_stoi[START]]], device=DEVICE)

        decoded_words = []
        for _ in range(max_length):
            decoder_output, decoder_hidden = decoder(prev_word, decoder_hidden)
            _, topi = decoder_output.topk(1)
            idx = topi.item()
            word = tgt_itos[idx]
            if word == END:
                break
            decoded_words.append(word)
            prev_word = topi.detach()

    return decoded_words, unknown
