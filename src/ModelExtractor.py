from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import pdb
from typing import List, Tuple
from peft import PeftModel, PeftConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
TOKEN = ""

def get_model(model_name: str):
    """Returns the model object given the model name."""
    if model_name == "gpt2":
        return GPTBScorer()
    elif model_name == "gpt2-large":
        return GPTLScorer()
    elif model_name == "gpt2-b-ft-def-klex":
        return GPTBScorerFTKlex(optim="default")
    elif model_name == "gpt2-b-ft-optim-klex":
        return GPTBScorerFTKlex(optim="optim")
    elif model_name == "gpt2-b-ft-def-wiki":
        return GPTBScorerFTWiki(optim="default")
    elif model_name == "gpt2-b-ft-optim-wiki":
        return GPTBScorerFTWiki(optim="optim")
    elif model_name == "llama-7b":
        return Llama7Scorer()
    elif model_name == "llama-7b_finetuned_wiki":
        return Llama7ScorerFTWiki()
    elif model_name == "llama-7b_finetuned_kids":
        return Llama7ScorerFTKids()
    elif model_name == "llama-13b":
        return Llama13Scorer()
    elif model_name == "mixtral":
        return MixtralScorer()
    else:
        raise ValueError(f"Model {model_name} not supported.")

class LMScorerBase:
    def __init__(self, optim: str = ""):
        self.optim = optim
        self.model = self.load_model()
        self.tokenizer = self.get_tokenizer()
        self.BOS = self.check_if_BOS()
        self.EOS = self.check_if_EOS()
        self.STRIDE = 200

    def load_model(self):
        raise NotImplementedError("Subclasses must implement this method")

    def get_tokenizer(self):
        raise NotImplementedError("Subclasses must implement this method")
            
    def check_if_BOS(self):
        test = self.tokenizer('test').input_ids
        return True if test[0] == self.tokenizer.bos_token_id else False
    
    def check_if_EOS(self):
        test = self.tokenizer('test').input_ids
        return True if test[-1] == self.tokenizer.eos_token_id else False

    def score(self, sentence) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], List[np.ndarray]]:
        with torch.no_grad():
            all_probs = torch.tensor([], device=self.model.device)
            all_entropies = torch.tensor([], device=self.model.device)
            offset_mapping = []
            start_ind = 0
            # add BOS and EOS tokens if they are not already in the sentence
            bos_id = [self.tokenizer.bos_token_id] if not self.BOS == True else []
            bos_mask = [1] if not self.BOS == True else []
            eos_id = [self.tokenizer.eos_token_id] if not self.EOS == True else []
            eos_mask = [1] if not self.EOS == True else []
            encodings = self.tokenizer(
                sentence[start_ind:],
                max_length=1022,
                truncation=True,
                return_offsets_mapping=True,
            )
            tensor_input = torch.tensor(
                    [bos_id + encodings["input_ids"] + eos_id],
                    device=self.model.device,
                )
            attention_mask = torch.tensor(
                    [bos_mask + encodings.attention_mask + eos_mask],
                    device=self.model.device,
                )
            # encodings.offset_mapping: maps subwords to the original words via the subword's start and end
            # position relative to the original token it was split from
            # text: 'Wohin wird sich das Klima entwickeln?'
            # tokenized text: ['Wohin', 'Ġwird', 'Ġsich', 'Ġdas', 'ĠKlima', 'Ġentwickeln', '?']
            # offset_mapping: [(0, 5), (5, 10), (10, 15), (15, 19), (19, 25), (25, 36), (36, 37)]
            output = self.model(
                tensor_input,
                labels=tensor_input,
                output_attentions=True,
                attention_mask=attention_mask,
            )
            # output.logits: tensor of shape [bsz, seq_len, vocab]
            # output.past_key_values: contains pre-computed hidden states
            # output.attentions: Tuple of torch.FloatTensor (one for each layer) of shape [bsz, n heads, seq len, seq len]
            # shift logits because at each step k, the logits represent the probabilities of the tokens at step k+1
            # leave away logits for EOS token (the refer to nothing)
            shift_logits = output["logits"][..., :-1, :].contiguous()
            # leave away BOS token; now the shift labels match with the logits (the logits at index 0 are the logits
            # for the BOS token but they refer to the probabilities of the first token in the sentence, so they
            # match with the shift labels)
            shift_labels = tensor_input[..., 1:].contiguous().cpu().numpy()

            # get probabilities: softmax over shift logits 
            all_probabilities = torch.nn.functional.softmax(output.logits, dim=-1)

            # Use advanced indexing to select the probabilities
            subtoken_probabilities = all_probabilities[0, range(shift_labels.shape[1]), shift_labels[0]]
            # remove the last element because it's P(EOS|last token)
            subtoken_probabilities = subtoken_probabilities[:-1]
            # different way to get log probs via cross entropy: 
            # in this case, use shift_labels = tensor_input[..., 1:].contiguous() 
            # get probs by np.exp(-log_probs.cpu())
            # log_probs = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')

            # flatten probabilities and compute entropy
            flat_probabilities = all_probabilities.view(-1, all_probabilities.size(-1))
            entropies = -torch.sum(flat_probabilities * torch.log(flat_probabilities), dim=1)
            # first entry in entropies will always be the same because it's the entropy of the BOS token (P(w1|BOS))
            # also, we don't need the last entry because it's the entropy of the EOS token (P(EOS|wN))
            entropies = entropies[1:-1]

            ### attentions ###
            # initially also returned the CLS query but it was always 1 for the CLS key and 0 otherwise;
            # TODO look at individual attention heads?
            attentions_list = list()
            attentions = output.attentions
            #attentions_dict = dict()
            for layer_idx, layer in enumerate(attentions):
                # average over attention heads and squeeze 'empty' batch size dimension
                avg_ah = torch.mean(layer, dim=1).squeeze(0)
                # CLS query
                #att_cls = avg_ah[0, :]
                # average over all queries and get rid of BOS and EOS attention scores
                att_avg = torch.mean(avg_ah, dim=0)[1:-1]
                # get rid of BOS and EOS attention scores
                # attentions_dict[layer_idx] = {
                #     'att_cls': att_cls[1:-1],
                #     'att_avg': att_avg[1:-1],
                # }
                attentions_list.append(np.asarray(att_avg.cpu()))

            all_probs = torch.cat([all_probs, subtoken_probabilities])
            all_entropies = torch.cat([all_entropies, entropies])
            offset_mapping = encodings.offset_mapping
            # if BOS or EOS were in the original encodings, remove them from the offset_mapping
            if self.BOS == True:
                offset_mapping.pop(0)
            if self.EOS == True:
                offset_mapping.pop(-1)
            return np.asarray(all_probs.cpu()), np.asarray(all_entropies.cpu()), offset_mapping, attentions_list


class GPTBScorer(LMScorerBase):
    def __init__(self):
        super().__init__()
        self.name = "gpt2-base"

    def load_model(self):
        model = GPT2LMHeadModel.from_pretrained("benjamin/gerpt2")
        model.to(device)
        model.eval()
        return model

    def get_tokenizer(self):
        return GPT2TokenizerFast.from_pretrained("benjamin/gerpt2")
    

class GPTLScorer(LMScorerBase):
    def __init__(self):
        super().__init__()
        self.name = "gpt2-large"

    def load_model(self):
        model = GPT2LMHeadModel.from_pretrained("benjamin/gerpt2-large")
        model.to(device)
        model.eval()
        return model

    def get_tokenizer(self):
        return GPT2TokenizerFast.from_pretrained("benjamin/gerpt2-large")


class GPTBScorerFTKlex(LMScorerBase):
    def __init__(self, optim: str):
        super().__init__(optim)
        self.name = f"gpt2-base-finetuned-{self.optim}-klexicon"

    def load_model(self):
        model = GPT2LMHeadModel.from_pretrained(f"/srv/scratch1/bolliger_haller/population-biases/finetuning/gerpt2/{self.optim}_params/klexikon_sentences")
        model.to(device)
        model.eval()
        return model

    def get_tokenizer(self):
        return GPT2TokenizerFast.from_pretrained("benjamin/gerpt2")


class GPTBScorerFTWiki(LMScorerBase):
    def __init__(self, optim: str):
        super().__init__(optim)
        self.name = f"gpt2-base-finetuned-{self.optim}-wiki"
        
    def load_model(self):
        model = GPT2LMHeadModel.from_pretrained(f"/srv/scratch1/bolliger_haller/population-biases/finetuning/gerpt2/{self.optim}_params/wiki_sentences")
        model.to(device)
        model.eval()
        return model

    def get_tokenizer(self):
        return GPT2TokenizerFast.from_pretrained("benjamin/gerpt2")


class Llama7Scorer(LMScorerBase):
    def __init__(self):
        super().__init__()
        self.name = "llama-7b"

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            "LeoLM/leo-hessianai-7b",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
        model.eval()
        return model
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained("LeoLM/leo-hessianai-7b")
    

class Llama7ScorerFTWiki(LMScorerBase):
    def __init__(self):
        super().__init__()
        self.name = "llama-7b_finetuned_wiki"

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            "LeoLM/leo-hessianai-7b",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            )
        model = PeftModel.from_pretrained(
            model,
            "/srv/scratch1/bolliger_haller/population-biases/finetuning/leo-hessianai-7b/wiki_sentences/checkpoint-5500"
        )
        model.eval()
        return model
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained("LeoLM/leo-hessianai-7b")

class Llama7ScorerFTKids(LMScorerBase):
    def __init__(self):
        super().__init__()
        self.name = "llama-7b_finetuned_kids"

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            "LeoLM/leo-hessianai-7b",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            )
        model = PeftModel.from_pretrained(
            model,
            "/srv/scratch1/bolliger_haller/population-biases/finetuning/leo-hessianai-7b/klexikon_sentences"
        )
        model.eval()
        return model
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained("/srv/scratch1/bolliger_haller/population-biases/finetuning/leo-hessianai-7b/klexikon_sentences")

    
class Llama13Scorer(LMScorerBase):
    def __init__(self):
        super().__init__()
        self.name = "llama-13b"

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            "LeoLM/leo-hessianai-13b",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
        model.eval()
        return model
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained("LeoLM/leo-hessianai-13b")


class MixtralScorer(LMScorerBase):
    def __init__(self):
        super().__init__()
        self.name = "mixtral"

    def load_model(self):
        model_id = "mistralai/Mixtral-8x7B-v0.1"
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            token = TOKEN,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model.eval()
        return model
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    