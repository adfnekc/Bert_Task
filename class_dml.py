import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

os.environ["HTTP_PROXY"] = "socks5://172.18.224.1:20080"
os.environ["HTTPS_PROXY"] = "socks5://172.18.224.1:20080"


class CustomTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 手动进行编码和padding
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[: self.max_len - 2]
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        padding_length = self.max_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length

        attention_mask = [1] * len(tokens) + [0] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


class BertClassifier(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # BERT返回的第二个值是[CLS] token的输出
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def train(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0

    accumulation_steps = 8

    for i, batch in tqdm(enumerate(data_loader)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)

        loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.zero_grad()
        optimizer.step()

    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(
        data_loader
    )


def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(
        data_loader
    )


# 加载数据集
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
ds = load_dataset(
    "csv", data_files="./dataset/toutiao_news_class/toutiao_cat_data_formate.csv"
)["train"]
labels = ds["label"]

# 将字符串标签转换为数值ID
label_encoder = LabelEncoder()
ds = ds.add_column("label_ids", label_encoder.fit_transform(labels))

split_ds = ds.train_test_split(test_size=0.2, seed=42)
train_dataset = CustomTextDataset(
    split_ds["train"]["content"], split_ds["train"]["label_ids"], tokenizer, max_len=128
)
val_dataset = CustomTextDataset(
    split_ds["test"]["content"], split_ds["test"]["label_ids"], tokenizer, max_len=128
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# 设置模型
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch_directml

device = torch_directml.device()
print("device.index", device.index)
model = BertClassifier(bert_model_name="bert-base-chinese", num_labels=len(set(labels)))


def quantize_model(model):
    def print_size_of_model(model):
        torch.save(model.state_dict(), "temp.p")
        print("Size (MB):", os.path.getsize("temp.p") / 1e6)
        os.remove("temp.p")

    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    print_size_of_model(model)
    print_size_of_model(quantized_model)

    return quantized_model


modle = quantize_model(model)
model = model.to(device)
# model = torch.compile(model)

# 损失函数和优化器
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# 训练模型
epochs = 3
for epoch in range(epochs):
    train_acc, train_loss = train(model, train_loader, loss_fn, optimizer, device)
    val_acc, val_loss = evaluate(model, val_loader, loss_fn, device)

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
