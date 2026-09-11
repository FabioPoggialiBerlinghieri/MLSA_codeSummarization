import sys

import yaml
from torch import optim, nn
from torch.utils.data import DataLoader
from EDTransf import EDTransf
from eval_model import SampleEvaluator
from main import VocabularyStoreHandler, DatasetHandler
from paddingMask import PaddingMask
import argparse
import torch
import evaluate as e

bleu = e.load("bleu")

class ModelTrainer:

    def __init__(self, dataset_handler):

        self.dataset_handler = dataset_handler

        self.input_max_len = dataset_handler.max_code_len
        self.output_max_len = dataset_handler.max_sum_len

        self.train_dataset = None
        self.validation_dataset = None
        self.model = None

    def initialize_model(self):
        self.train_dataset = self.dataset_handler.load_dataset("train")
        self.validation_dataset = self.dataset_handler.load_dataset("val")

        embedding_dim = self.dataset_handler.config_yaml['model']['embedding_dim']
        python_voc = VocabularyStoreHandler.load_vocabulary(self.dataset_handler.python_voc_path)
        code_voc = VocabularyStoreHandler.load_vocabulary(self.dataset_handler.english_voc_path)

        self.model = EDTransf(embedding_dim,
                              self.input_max_len, len(python_voc),
                              self.output_max_len, len(code_voc))

    def train(self, save_every):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        print("device:", device)

        self.model.to(device)

        learning_rate = self.dataset_handler.config_yaml['learning_rate']
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
        epochs = self.dataset_handler.config_yaml['epochs']

        batch_size = self.dataset_handler.config_yaml['batch_size']
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=self.validation_dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        best_bleu = 0

        for epoch in range(1, epochs + 1):
            training_loss = 0.0
            valid_loss = 0.0
            self.model.train()  # train status for the mode

            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()  # clear gradients for next train
                inputs, targets = batch
                print(f'input shape: {inputs.shape}')
                print(f'target shape: {targets.shape}')

                inputs = inputs.to(device)
                inputs_mask = PaddingMask.generate_padding_mask(inputs).to(device)
                targets = targets.to(device)
                shifted_target = targets[:, :-1].to(device)
                targets_mask = PaddingMask.generate_padding_mask(shifted_target).to(device)

                output = self.model(inputs, inputs_mask, shifted_target, targets_mask)

                loss = loss_fn(output, targets[:, 1:])  # target without cls
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
                training_loss += loss.data.item() * inputs.size(0)

                if step % save_every == 0 and step > 0:
                    checkpoint_latest = {
                        'config': self.dataset_handler.config_yaml,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'step': step
                    }
                    torch.save(checkpoint_latest, self.dataset_handler.config_yaml['checkpoint_path'])
                    print("Checkpoint saved...")

            training_loss /= len(self.train_dataset)

            with torch.no_grad():  # we are not updating the model
                self.model.eval()  # the status of the model is in eval
                num_correct = 0
                num_examples = 0

                for batch in val_loader:
                    inputs, targets = batch

                    inputs = inputs.to(device)
                    inputs_mask = PaddingMask.generate_padding_mask(inputs).to(device)
                    targets = targets.to(device)
                    shifted_target = targets[:, :-1].to(device)
                    target_out = targets[:, 1:].to(device)
                    targets_mask = PaddingMask.generate_padding_mask(shifted_target).to(device)

                    output = self.model(inputs, inputs_mask, shifted_target, targets_mask)

                    loss = loss_fn(output, target_out)  # target without cls
                    valid_loss += loss.data.item() * inputs.size(0)
                    predictions = torch.argmax(output, dim=1)

                    # mask to ignore 0 (padding)
                    valid_tokens_mask = (target_out != 0)

                    correct = torch.eq(predictions, target_out) & valid_tokens_mask
                    num_correct += torch.sum(correct).item()
                    num_examples += torch.sum(valid_tokens_mask).item()
                valid_loss /= len(self.validation_dataset)

            if valid_loss < best_loss:
                best_loss = valid_loss
                checkpoint_loss = {
                    'config': self.dataset_handler.config_yaml,
                    'model_state_dict': self.model.state_dict()
                }
                torch.save(checkpoint_loss, self.dataset_handler.config_yaml['best_loss_path'])
                print(f"New best loss saved ({valid_loss}) ...")

            sample_batch = next(iter(val_loader))
            inputs, targets = sample_batch

            generate_summs = []
            target_sentences = []

            for i in range(inputs.size(0)):
                sample = (inputs[i], targets[i])
                summ_sentence, target_sentence = SampleEvaluator.eval_sample(sample, self.model, self.dataset_handler.englishTokenizer, device)
                generate_summs.append(summ_sentence)
                target_sentences.append([target_sentence])

            bleu_result = bleu.compute(predictions=generate_summs, references=target_sentences)['bleu']

            if bleu_result > best_bleu:
                best_bleu = bleu_result
                checkpoint_loss = {
                    'config': self.dataset_handler.config_yaml,
                    'model_state_dict': self.model.state_dict()
                }
                torch.save(checkpoint_loss, self.dataset_handler.config_yaml['best_bleu_path'])
                print(f"New best bleu saved ({best_bleu}) ...")


            print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}, bleu = {:.4f}'.format(
                epoch,training_loss, valid_loss, num_correct / num_examples, bleu_result))

parser = argparse.ArgumentParser(description="TBD")
parser.add_argument('--config', type=str, required=True, help="YAML file path")
parser.add_argument('--max-code-len', type=int, default=None, help="Max code length")
parser.add_argument('--max-sum-len', type=int, default=None, help="Max text length")
parser.add_argument('--save-every', type=int, default=None, help="Save model weights every n epochs")
parser.add_argument('--resume', type=str, default=None, help="Resume file path")

args = parser.parse_args()

config_filepath = args.config
max_code_len = args.max_code_len
max_text_len = args.max_sum_len
save_every = args.save_every
resume = args.resume

config_filepath = config_filepath
try:
    with open(config_filepath, 'r') as file:
        config_yaml = yaml.safe_load(file)
except FileNotFoundError:
    print(f"Fatal error: '{config_filepath}' does not exist!")
    sys.exit(1)

dataset_handler = DatasetHandler(config_yaml, max_code_len, max_text_len)
model_trainer = ModelTrainer(dataset_handler)
model_trainer.initialize_model()
model_trainer.train(save_every)








