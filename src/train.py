import json
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
import EDTransf as e
import paddingHandler
from dataset import CodeDataset
from paddingMask import PaddingMask
from vocabulary_generator import PythonVocabularyGenerator
import tokenizer as tc


def train(model,
          optimizer,
          loss_fn,
          train_loader,
          val_loader,
          epochs=20, device="cpu"):
    model.to(device)
    for epoch in range(1, epochs+1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train() #train status for the mode
        for batch in train_loader:
            optimizer.zero_grad() # clear gradients for next train
            inputs, targets = batch
            print(f'input shape: {inputs.shape}')
            print(f'target shape: {targets.shape}')

            inputs = inputs.to(device)
            inputs_mask = PaddingMask.generate_padding_mask(inputs).to(device)
            targets = targets.to(device)
            targets_mask = PaddingMask.generate_padding_mask(targets).to(device)

            output = model(inputs, inputs_mask, targets[:, :-1, :], targets_mask[:, :-1, :])

            loss = loss_fn(output, targets[:, 1:, :])
            loss.backward() # backpropagation, compute gradients
            optimizer.step() # apply gradients
            training_loss += loss.data.item() * inputs.size(0)
            # print(training_loss,loss.data.item(),inputs.size(0))
        training_loss /= len(train_loader.dataset)

        with torch.no_grad(): # we are not updating the model
          model.eval() #the status of the model is in eval
          num_correct = 0
          num_examples = 0
          for batch in val_loader:
              inputs, targets = batch

              inputs = inputs.to(device)
              inputs_mask = PaddingMask.generate_padding_mask(inputs).to(device)
              targets = targets.to(device)
              output = model(inputs, inputs_mask, targets[:, :-1, :], targets_mask[:, :-1, :])

              loss = loss_fn(output, targets[:, 1:, :])
              valid_loss += loss.data.item() * inputs.size(0)
              correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
              num_correct += torch.sum(correct).item()
              num_examples += correct.shape[0]
          valid_loss /= len(val_loader.dataset)

        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))

with open("../data/code_vocab.json", "r") as f:
    code_vocabulary = json.load(f)

with open("../data/english_vocab.json", "r") as f:
    english_vocabulary = json.load(f)

optimusPy = e.EDTransf(512, 1024, 512, len(code_vocabulary), 128, len(english_vocabulary))

train_dataframe = pd.read_json("../data/dataset_train.json", orient='records')
train_dataset = CodeDataset(torch.tensor(train_dataframe["code"], dtype=torch.long), torch.tensor(train_dataframe["text"], dtype=torch.long))

validation_dataframe = pd.read_json("../data/dataset_validation.json", orient='records')
validation_dataset = CodeDataset(torch.tensor(validation_dataframe["code"], dtype=torch.long), torch.tensor(validation_dataframe["text"], dtype=torch.long))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=True)

optimizer = optim.Adam(optimusPy.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss(ignore_index=0)

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
print("device:", device)

optimusPy.to(device)

train(optimusPy, optimizer, loss, train_loader, val_loader, epochs=10, device=device)

# PRIMO TEST

code = """
def rectangle_perimeter(l,b):
    perimeter = l*(l+b)
    return perimeter
"""

with open("../data/code_vocab.json", "w") as f:
    json.dump(code_vocabulary, f, indent=4)

main_keywords = PythonVocabularyGenerator.get_main_keywords()
codeTokenizer = tc.CodeTokenizer(code_vocabulary, main_keywords)
code_padding_handler = paddingHandler.PaddingHandler(512)

with open("../data/english_vocab.json", "w") as f:
    json.dump(english_vocabulary, f, indent=4)

englishTokenizer = tc.EnglishTextTokenizer(english_vocabulary)
text_padding_handler = paddingHandler.PaddingHandler(128)

# tokenization and padding
code = code_padding_handler.padding(codeTokenizer.tokenize(code))

optimusPy.eval()
with torch.no_grad():
    code = torch.tensor([code], device=device)
    mask = PaddingMask.generate_padding_mask(code).to(device)
    cls_tokens = englishTokenizer.tokenize("[CLS]")
    cls_tensor = torch.tensor([cls_tokens], dtype=torch.long, device=device)
    summ_ids = optimusPy.predict(code, mask, cls_tensor)

summ_list = summ_ids[0].tolist()
summ = englishTokenizer.detokenize(summ_list)
print("E il nostro primo commento di x = x + 1 e':", summ)