import torch
import numpy as np
from transformers import AutoModel, AutoModelForCausalLM
from torch.nn.functional import softmax

class GPT():    
    """wrapper for https://huggingface.co/openai-gpt
    """
    def __init__(self, path, vocab=None, device = 'cpu'): 
        self.device = device
        self.path = path
        # Hidden-state extraction does not need LM logits, so keep the lighter base model resident.
        self.model = AutoModel.from_pretrained(path).eval().to(self.device)
        self.lm_model = None
        
        if vocab is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(path)
            self.vocab = list(tokenizer.get_vocab().keys())
            self.word2id = tokenizer.get_vocab()
        else:
            self.vocab = vocab
            self.word2id = {w : i for i, w in enumerate(self.vocab)}
            
        self.UNK_ID = self.word2id.get('<unk>', 0)

    def encode(self, words):
        """map from words to ids
        """
        return [self.word2id[x] if x in self.word2id else self.UNK_ID for x in words]
        
    def get_story_array(self, words, context_words):
        """get word ids for each phrase in a stimulus story
        """
        nctx = context_words + 1
        story_ids = self.encode(words)
        story_array = np.full((len(story_ids), nctx), self.UNK_ID, dtype = np.int64)
        for i in range(len(story_array)):
            segment = story_ids[i:i+nctx]
            story_array[i, :len(segment)] = segment
        return torch.tensor(story_array).long()

    def get_context_array(self, contexts):
        """get word ids for each context
        """
        context_array = np.array([self.encode(words) for words in contexts])
        return torch.tensor(context_array).long()

    def get_hidden(self, ids, layer):
        """get hidden layer representations
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids = ids.to(self.device), 
                                 attention_mask = mask.to(self.device), output_hidden_states = True)
        return outputs.hidden_states[layer].detach().cpu().numpy()

    def get_probs(self, ids):
        """get next word probability distributions
        """
        if self.lm_model is None:
            self.lm_model = AutoModelForCausalLM.from_pretrained(self.path).eval().to(self.device)
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.lm_model(input_ids = ids.to(self.device), attention_mask = mask.to(self.device))
        probs = softmax(outputs.logits, dim = 2).detach().cpu().numpy()
        return probs
