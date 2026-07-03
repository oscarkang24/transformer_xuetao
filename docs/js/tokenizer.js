// Word-level tokenizer over the toy vocabulary.
// Must match the tokenize() in scripts/export_toy_model.py exactly:
// lowercase words and "." are tokens; anything not in the vocab -> <|unk|>.

const Tokenizer = {
  UNK: "<|unk|>",

  split(text) {
    return (text.toLowerCase().match(/[a-z]+|\./g) || []);
  },

  // Returns [{word, token, id, unk}] — `word` is what the user typed,
  // `token` is the vocab entry actually used.
  encode(text, vocab) {
    const stoi = Tokenizer.stoi(vocab);
    return Tokenizer.split(text).map((word) => {
      const unk = !(word in stoi);
      const token = unk ? Tokenizer.UNK : word;
      return { word, token, id: stoi[token], unk };
    });
  },

  decode(ids, vocab) {
    return ids.map((i) => vocab[i]).join(" ").replace(/ \./g, ".");
  },

  stoi(vocab) {
    if (!Tokenizer._stoi || Tokenizer._vocab !== vocab) {
      Tokenizer._stoi = Object.fromEntries(vocab.map((s, i) => [s, i]));
      Tokenizer._vocab = vocab;
    }
    return Tokenizer._stoi;
  },
};
