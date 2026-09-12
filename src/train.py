import sys
import yaml
from torch import optim, nn
from torch.utils.data import DataLoader
from EDTransf import EDTransf
from eval_model import SampleEvaluator
from datasetHandler import VocabularyStoreHandler, DatasetHandler
from paddingMask import PaddingMask
import argparse
import torch
import evaluate as e
import wandb

bleu = e.load("bleu")

class ModelTrainer:

    def __init__(self, dataset_handler):

        self.dataset_handler = dataset_handler

        self.input_max_len = dataset_handler.max_code_len
        self.output_max_len = dataset_handler.max_sum_len

        self.train_dataset = None
        self.validation_dataset = None
        self.model = None
        self.python_voc_len = 0
        self.english_voc_len = 0

    def initialize_model(self):
        train = self.dataset_handler.config_yaml['split_train']
        self.train_dataset = self.dataset_handler.load_dataset(train)

        val = self.dataset_handler.config_yaml['split_val']
        self.validation_dataset = self.dataset_handler.load_dataset(val)

        embedding_dim = self.dataset_handler.config_yaml['model']['embedding_dim']
        python_voc = VocabularyStoreHandler.load_vocabulary(self.dataset_handler.python_voc_path)
        english_voc = VocabularyStoreHandler.load_vocabulary(self.dataset_handler.english_voc_path)

        self.python_voc_len = len(python_voc)
        self.english_voc_len = len(english_voc)

        self.model = EDTransf(embedding_dim,
                              self.input_max_len, len(python_voc),
                              self.output_max_len, len(english_voc),
                              self.dataset_handler.config_yaml['model']['num_layers'],
                              self.dataset_handler.config_yaml['model']['num_heads'])

    def train(self, save_every, resume_path=None):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        print("device:", device)

        model_config = self.dataset_handler.config_yaml['model']
        model_config['max_code_len'] = self.input_max_len
        model_config['max_text_len'] = self.output_max_len
        model_config['code_voc_len'] = self.python_voc_len
        model_config['text_voc_len'] = self.english_voc_len

        self.model.to(device)

        learning_rate = self.dataset_handler.config_yaml['learning_rate']
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
        epochs = self.dataset_handler.config_yaml['epochs']

        batch_size = self.dataset_handler.config_yaml['batch_size']
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=self.validation_dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        best_bleu = 0
        start_epoch = 1
        wandb_id = None

        if resume_path is not None:
            try:
                checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_loss = checkpoint.get('best_loss', float('inf'))
                best_bleu = checkpoint.get('best_bleu', 0)
                wandb_id = checkpoint.get('wandb_id')
                print("Resume training...")
            except Exception as e:
                print(f"Error while resume: {e}")
                sys.exit(1)

        if wandb_id is not None:
            wandb.init(project="CodeSummarization_MLSA", config=self.dataset_handler.config_yaml, id=wandb_id,
                       resume="must")
        else:
            wandb.init(project="CodeSummarization_MLSA", config=self.dataset_handler.config_yaml)
            wandb_id = wandb.run.id

        bleu_eval_size = min(batch_size, len(self.validation_dataset))
        bleu_subset = [self.validation_dataset[i] for i in range(bleu_eval_size)]

        print("Start training...")

        for epoch in range(start_epoch, epochs + 1):
            print("Train:")
            print("Epoch:", epoch)
            training_loss = 0.0
            valid_loss = 0.0
            self.model.train()  # train status for the mode

            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()  # clear gradients for next train
                inputs, targets = batch

                inputs = inputs.to(device)
                inputs_mask = PaddingMask.generate_padding_mask(inputs).to(device)
                targets = targets.to(device)
                shifted_target = targets[:, :-1].to(device)
                targets_mask = PaddingMask.generate_padding_mask(shifted_target).to(device)

                output = self.model(inputs, inputs_mask, shifted_target, targets_mask)

                loss = loss_fn(output, targets[:, 1:])  # target without cls
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                current_batch_loss = loss.data.item()
                wandb.log({"train_batch_loss": current_batch_loss})

                training_loss += loss.data.item() * inputs.size(0)

                if step % save_every == 0 and step > 0:
                    checkpoint_latest = {
                        'config': self.dataset_handler.config_yaml,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'step': step,
                        'best_loss': best_loss,
                        'best_bleu': best_bleu,
                        'wandb_id': wandb_id
                    }
                    torch.save(checkpoint_latest, self.dataset_handler.config_yaml['checkpoint_path'])
                    print("Checkpoint saved...")

            training_loss /= len(self.train_dataset)

            print("Validation:")
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

            print("Validation loss done.")
            generate_summs = []
            target_sentences = []

            for sample in bleu_subset:
                summ_sentence, target_sentence = SampleEvaluator.eval_sample(sample, self.model,
                                                                             self.dataset_handler.englishTokenizer,
                                                                             device)
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
            print("Validation bleu done.")

            wandb.log({
                "epoch": epoch,
                "train_epoch_loss": training_loss,
                "val_loss": valid_loss,
                "val_bleu": bleu_result
            })

            print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}, bleu = {:.4f}'.format(
                epoch,training_loss, valid_loss, num_correct / num_examples, bleu_result))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TBD")
    parser.add_argument('--config', type=str, required=True, help="YAML file path")
    parser.add_argument('--max-code-len', type=int, default=512, help="Max code length")
    parser.add_argument('--max-sum-len', type=int, default=128, help="Max text length")
    parser.add_argument('--save-every', type=int, default=500, help="Save model weights every n epochs")
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
    dataset_handler.set_yaml_path(config_filepath)

    model_trainer = ModelTrainer(dataset_handler)
    model_trainer.initialize_model()
    model_trainer.train(save_every, resume)