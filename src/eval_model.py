import argparse
import evaluate as e
import torch
from EDTransf import EDTransf
from main import DatasetHandler, VocabularyStoreHandler
from paddingMask import PaddingMask
from tokenizer import CodeTokenizer

bleu = e.load("bleu")
# meteor = e.load("meteor")
# rouge = e.load("rouge")

class ModelEvaluator:

    def __init__(self, saved_data, dataset_handler, split = None):
        self.saved_data = saved_data

        self.config = saved_data['config']

        self.model = EDTransf(embedding_dim=self.config['model']['embedding_dim'],
                         input_max_len=self.config['model']['max_code_len'],
                         input_vocabulary_size= self.config['model']['code_voc_len'],
                         output_max_len=self.config['model']['max_text_len'],
                         output_vocabulary_size= self.config['model']['text_voc_len'])

        self.model.load_state_dict(saved_data['model_state_dict'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.dataset_handler = dataset_handler

        if split is not None:
            self.dataset = dataset_handler.load_dataset(split)

    def evaluate(self):
        generate_summs = []
        target_sentences = []
        self.model.eval()
        for sample in self.dataset:
            summ_sentence, target_sentence = SampleEvaluator.eval_sample(sample, self.model, self.dataset_handler.englishTokenizer, self.device)
            generate_summs.append(summ_sentence)
            target_sentences.append([target_sentence])

        # compute metrix
        blue_result = bleu.compute(predictions=generate_summs, references=target_sentences)
        meteor_result = None #meteor.compute(predictions=generate_summs, references=target_sentences)
        rouge_result = None #rouge.compute(predictions=generate_summs, references=target_sentences)

        return blue_result, meteor_result, rouge_result

    def summarize(self, code):
        code = self.dataset_handler.codeTokenizer.tokenize(code)
        target = self.dataset_handler.englishTokenizer.tokenize("") # no target
        sample = (code, target)
        summ_sentence, _ = SampleEvaluator.eval_sample(sample, self.model,
                                                                     self.dataset_handler.englishTokenizer, self.device)
        return summ_sentence

class SampleEvaluator:

    @staticmethod
    def eval_sample(sample, model, tokenizer, device) -> tuple[str, str]:
        code, target = sample
        with torch.no_grad():
            code = code.to(device)
            mask = PaddingMask.generate_padding_mask(code).squeeze(1).to(device)
            cls = tokenizer.tokenize("[CLS]")[0]
            sep = tokenizer.tokenize("[SEP]")[0]
            summ_ids = model.predict(code, mask, cls, sep)

        summ_list = summ_ids[0].tolist()
        summ = tokenizer.detokenize(summ_list)
        summ_sentence = summ.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "").strip()

        target = tokenizer.detokenize(target)
        target_sentence = target.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "").strip()
        return summ_sentence, target_sentence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TBD")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path best bleu weight file")
    parser.add_argument('--split', type=str, default=None, help="Dataset split")

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    split = args.split

    saved_data = torch.load(checkpoint_path, map_location='cpu')

    dataset_handler = DatasetHandler(
        saved_data['config'],
        saved_data['config']['model']['max_code_len'],
        saved_data['config']['model']['max_text_len']
    )

    model_evaluator = ModelEvaluator(saved_data, dataset_handler, split)
    bleu_result, meteor_result, rouge_result = model_evaluator.evaluate()

    print(bleu_result)
    print(meteor_result)
    print(rouge_result)