import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import numpy as np
import tqdm

class PpvPooling2D(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    def forward(self, input):
        # returns values between 0 and 1, mostly ones or zeros
        transformed_input = ((F.hardtanh(input, min_val=-self.threshold, max_val=+self.threshold) / self.threshold) + 1) / 2
        # average pooling on this transformed input is almost equivalent to ppv pooling, except that the transition between one and zero is smooth instead of hard
        return self.avg(transformed_input)
    
class R2D2(nn.Module):
    def __init__(self, h, w, ch=1, threshold=1e-7, n1=10, n2=1000, auto_pca=False, device="cpu"):
        super().__init__()
        self.n1 = n1
        self.n2 = n2
        self.flatten = nn.Flatten().to(device=device)
        self.ppv = PpvPooling2D(threshold=threshold).to(device=device)
        self.layer1 = nn.ModuleList([nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=3, padding=1) for _ in range(n1)]).to(device=device)
        
        self.last_layer = nn.ModuleList()
        for i in range(n2):
            # kernels
            kernel_rows = np.random.choice((2, 3, 4))
            kernel_cols = np.random.choice((2, 3, 4))
            # dilations
            dilation_row = np.int32(2 ** np.random.uniform(0, np.log2((h - 1) / (kernel_rows - 1))))
            dilation_col = np.int32(2 ** np.random.uniform(0, np.log2((w - 1) / (kernel_cols - 1))))
            # padding
            if np.random.randint(2) == 1:
                padding_row = ((kernel_rows - 1) * dilation_row) // 2
                padding_col = ((kernel_cols - 1) * dilation_col) // 2
            else:
                padding_row = 0
                padding_col = 0
            # create kernel
            random_kernel = nn.Conv2d(in_channels=1, 
                                      out_channels=1, 
                                      kernel_size=(kernel_rows, kernel_cols), 
                                      dilation=(dilation_row, dilation_col),
                                      padding=(padding_row, padding_col))
            # weight initialization
            nn.init.normal_(random_kernel.weight)
            # bias initialization
            nn.init.uniform_(random_kernel.bias, a=-1, b=1)
            self.last_layer.append(random_kernel)
        self.last_layer = self.last_layer.to(device=device)
        
        # pca initialization
        self.number_of_features = n1*n2
        self.auto_pca = False
        if auto_pca == True:
            self.pca = PCA(0.999)
            random_batch = torch.zeros(n1*n2, ch, h, w).to(device=device)
            nn.init.normal_(random_batch)
            transformed_batch = self(random_batch)
            self.pca.fit(transformed_batch.to("cpu").numpy())
            self.auto_pca = auto_pca
            self.number_of_features = self.pca.n_components_


    def forward(self, x: torch.Tensor, ) -> torch.Tensor:
        with torch.no_grad():
            features_list = []
            for i in range(self.n1):
                for j in range(self.n2):
                    feature = self.flatten(self.ppv(self.last_layer[j](self.layer1[i](x))))
                    features_list.append(feature)
                    del feature
                    torch.cuda.empty_cache()
            features = torch.cat(features_list, dim=-1)
        if self.auto_pca == False:
            return features.to("cpu")
        else:
            return torch.tensor(self.pca.transform(features.to(device="cpu").numpy()))
    
class RidgeClassifier(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.linear = nn.Linear(in_features=inputs, out_features=outputs)

    def forward(self, x):
        return self.linear(x)
    
class MultiRidgeClassifier(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.outputs = outputs
        self.layer = nn.ModuleList([nn.Linear(in_features=inputs, out_features=1) for _ in range(outputs)])

    def forward(self, x):
        return [lin(x) for lin in self.layer]


class LogisticRegression(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.linear = nn.Linear(in_features=inputs, out_features=outputs)

    def forward(self, x):
        return nn.functional.sigmoid(self.linear(x))
    
def fit_model(epochs, model, loss_func, opt, train_dl, val_dl, cumulative_gradient, scheduler=None, is_ridge=False, num_classes=-1):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_length = 0
        train_correct = 0
        for xb, yb in train_dl:
            pred = model.forward(xb)
            if is_ridge == True:
                binarized_labels = (F.one_hot(yb, num_classes=num_classes)*2)-1
                binarized_labels = binarized_labels.type(torch.float32)
                loss = loss_func(pred, binarized_labels)
            else:
                loss = loss_func(pred, yb)
            loss.backward()
            opt.step()
            cumulative_gradient += model.linear.weight.grad
            opt.zero_grad()
            train_loss += loss.item()
            train_length += len(xb)
            train_correct += (torch.argmax(pred.data, 1) == yb).sum()
        
        model.eval()
        val_loss = 0
        val_length = 0
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model.forward(xb)
                if is_ridge == True:
                    binarized_labels = (F.one_hot(yb, num_classes=num_classes)*2)-1
                    binarized_labels = binarized_labels.type(torch.float32)
                    loss = loss_func(pred, binarized_labels)
                else:
                    loss = loss_func(pred, yb)
                val_loss += loss.item() 
                val_length += len(xb)
                val_correct += (torch.argmax(pred.data, 1) == yb).sum()
        if scheduler is not None:
            scheduler.step(val_correct/val_length)
        print(f"""Epoch {epoch} norm: {cumulative_gradient.norm():.2f} lr: {scheduler._last_lr if scheduler is not None else 0}\ntrain loss: {train_loss:.6f}    acc: {(train_correct/train_length)*100:.2f} %\nvalid loss: {val_loss:.6f}    acc: {(val_correct/val_length)*100:.2f} %\n---------------------------""")
        cumulative_gradient -= cumulative_gradient        

def fit_model_multi(epochs, model, loss_func, opt, train_dl, val_dl, scheduler=None, num_classes=-1):
    for epoch in range(epochs):

        model.train()
        train_loss = 0
        train_length = 0
        train_correct = 0
        for xb, yb in train_dl:
            preds = model.forward(xb)
            #print(preds)
            pred = torch.cat(preds, dim=-1)
            #print(preds)
            binarized_labels = (F.one_hot(yb, num_classes=num_classes)*2)-1
            binarized_labels = binarized_labels.type(torch.float32)

            losses = []
            for i in range(num_classes):
                loss = loss_func(preds.pop(0), binarized_labels[:, i].flatten())
                #print(model.layer[0].weight.grad)
                loss.backward()
                #print(model.layer[3].weight.grad)
                opt.step()
                opt.zero_grad()
                losses.append(loss)
            
            train_loss += sum([loss.item() for loss in losses])
            train_length += len(xb)
            #print(torch.argmax(pred.data, 1), yb)
            train_correct += (torch.argmax(pred.data, 1) == yb).sum()
        
        model.eval()
        val_loss = 0
        val_length = 0
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                preds = model.forward(xb)
                pred = torch.cat(preds, dim=-1)
                binarized_labels = (F.one_hot(yb, num_classes=num_classes)*2)-1
                binarized_labels = binarized_labels.type(torch.float32)
                losses = [loss_func(preds.pop(0), binarized_labels[:, i].flatten()) for i in range(num_classes)]
                #[loss.backward() for loss in losses]
                #cumulative_gradient += model.linear.weight.grad
                val_loss += sum([loss.item() for loss in losses])
                val_length += len(xb)
                val_correct += (torch.argmax(pred.data, 1) == yb).sum()

        if scheduler is not None:
            scheduler.step(val_correct/val_length)
        print(f"""Epoch {epoch} lr: {scheduler._last_lr if scheduler is not None else 0}\ntrain loss: {train_loss:.6f}    acc: {(train_correct/train_length)*100:.2f} %\nvalid loss: {val_loss:.6f}    acc: {(val_correct/val_length)*100:.2f} %\n---------------------------""")
        #cumulative_gradient -= cumulative_gradient        
