import torch
import pandas as pd
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from utils6_loss import WeightedFocalLoss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils6_loss import weighted_mse_loss
from utils6_loss import weighted_focal_mse_loss
from utils6_loss import BMCLossMD

class Animator():
    def __init__(self, xlim, xlabel=None, ylabel=None, legend=None,ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'),figsize=(3.5, 2.5)) -> None:
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xscale = xscale
        self.yscale = yscale
        self.xlim = xlim
        self.ylim = ylim
        self.legend = legend
        if legend is None:
            legend = []
        backend_inline.set_matplotlib_formats('svg')
        self.fig, self.axes = plt.subplots(figsize=figsize)
        self.x = None
        self.y = None
        self.fmts = fmts

    def set_axes(self):
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_xscale(self.xscale)
        self.axes.set_yscale(self.yscale)
        self.axes.set_xlim(self.xlim)
        self.axes.set_ylim(self.ylim)
        if self.legend:
            self.axes.legend(self.legend)
        self.axes.grid()


    def show(self,x,y):
        self.axes.cla()
        for i in range(len(x)):
            self.axes.plot(x[i],y[i],self.fmts[i])
        self.set_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def model_training(model, device, train_loader, val_loader, use_metadata, optimizer, scheduler, loss_function, alpha, epochs, best_model_file_path, training_curve_file):
    print("Start training")

    model.to(device)

    if loss_function == 'WeightedFocalLoss': criterion = WeightedFocalLoss(alpha = alpha)

    fig = Animator(xlim=(-0.1,epochs+0.1),legend=["train_MAE","val_MAE"])
    epoch_x = [[],[]]
    training_curve = [[],[]]

    if loss_function == 'WeightedFocalLoss':
        best_accuracy = 0
    else:
        best_MAE = 999

    model.train()
    for epoch in range(epochs):
        tot_loss = 0.0
        tot_acc = 0.0
        train_preds = []
        train_trues = []
        model.train()
        for i,train_batch in enumerate(train_loader):
            x, edge_attr, edge_index, edge_type, batch, train_label_batch = train_batch.x.to(device), train_batch.edge_attr.to(device), train_batch.edge_index.to(device), train_batch.edge_type.to(device), train_batch.batch.to(device), train_batch.y.to(device)
            if loss_function == 'WeightedFocalLoss':
                train_label_batch[train_label_batch > 0] = 1.0

            if use_metadata:
                MetaData = train_batch.MetaData.to(device)
                train_outputs = model(x, edge_attr, edge_index, edge_type, MetaData, batch) # perform a single forward pass
            else:
                train_outputs = model(x, edge_attr, edge_index, edge_type, batch) # perform a single forward pass
                
            if loss_function == 'WeightedFocalLoss': loss = criterion(train_outputs.flatten(), train_label_batch.float())
            elif loss_function == 'weighted_mse_loss': loss = weighted_mse_loss(train_outputs.flatten(), train_label_batch, weights=torch.Tensor(train_batch.weight).to(device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tot_loss += loss.data
            if loss_function == 'WeightedFocalLoss': train_preds.extend((torch.sigmoid(train_outputs) >= 0.5).float().detach().cpu().numpy())
            else: train_preds.extend(train_outputs.detach().cpu().numpy())
            train_trues.extend(train_label_batch.detach().cpu().numpy())
        
        scheduler.step()

        if loss_function == 'WeightedFocalLoss':
            train_accuracy = accuracy_score(train_trues, train_preds) 
            train_precision = precision_score(train_trues, train_preds)
            train_recall = recall_score(train_trues, train_preds)
            train_f1 = f1_score(train_trues, train_preds)
            print("[sklearn_metrics] Epoch:{} loss:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(epoch, tot_loss, train_accuracy, train_precision, train_recall, train_f1))
        else:
            train_MSE = mean_squared_error(train_trues, train_preds) 
            train_MAE = mean_absolute_error(train_trues, train_preds)
            train_R2 = r2_score(train_trues, train_preds)
            print("[sklearn_metrics] Epoch:{} loss:{:.4f}".format(epoch, tot_loss))
            print("[sklearn_metrics] dropout_MSE:{:.4f} dropout_MAE:{:.4f} dropout_R2:{:.4f}".format(train_MSE, train_MAE, train_R2))

        train_preds = []
        train_trues = []
        test_preds = []
        test_trues = []
        model.eval()
        with torch.no_grad():
            for i,test_batch in enumerate(val_loader):
                x, edge_attr, edge_index, edge_type, batch, test_data_label = test_batch.x.to(device), test_batch.edge_attr.to(device), test_batch.edge_index.to(device), test_batch.edge_type.to(device), test_batch.batch.to(device), test_batch.y.to(device)
                test_data_label = test_batch.y
                if loss_function == 'WeightedFocalLoss':
                    test_data_label[test_data_label > 0] = 1.0

                if use_metadata:
                    MetaData = test_batch.MetaData.to(device)
                    test_outputs = model(x, edge_attr, edge_index, edge_type, MetaData, batch) # perform a single forward pass
                else:
                    test_outputs = model(x, edge_attr, edge_index, edge_type, batch) # perform a single forward pass

                if loss_function == 'WeightedFocalLoss': test_preds.extend((torch.sigmoid(test_outputs) >= 0.5).float().detach().cpu().numpy())
                else: test_preds.extend(test_outputs.detach().cpu().numpy())
                test_trues.extend(test_data_label.numpy())

            if loss_function == 'WeightedFocalLoss':
                test_accuracy = accuracy_score(test_trues, test_preds) 
                test_precision = precision_score(test_trues, test_preds)
                test_recall = recall_score(test_trues, test_preds)
                test_f1 = f1_score(test_trues, test_preds)
                conf_matrix = confusion_matrix(test_trues, test_preds)
                print("[sklearn_metrics] accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(test_accuracy, test_precision, test_recall, test_f1))
            else:  
                test_MSE = mean_squared_error(test_trues, test_preds) 
                test_MAE = mean_absolute_error(test_trues, test_preds)
                test_R2 = r2_score(test_trues, test_preds)
                print("[sklearn_metrics] test_MSE:{:.4f} test_MAE:{:.4f} test_R2:{:.4f}".format(test_MSE, test_MAE, test_R2))

            torch.save(model, best_model_file_path[:best_model_file_path.index(".pt")] + "_realtime" + best_model_file_path[best_model_file_path.index(".pt"):])
            if loss_function == 'WeightedFocalLoss':
                if test_f1 > best_accuracy:
                    best_accuracy = test_f1
                    torch.save(model, best_model_file_path)
            else:
                if test_MAE < best_MAE:
                    best_MAE = test_MAE
                    torch.save(model, best_model_file_path)
            
            epoch_x[0].append(epoch)
            if loss_function == 'WeightedFocalLoss':
                training_curve[0].append(train_f1)
            else:
                training_curve[0].append(train_MAE)
            epoch_x[1].append(epoch)
            if loss_function == 'WeightedFocalLoss':
                training_curve[1].append(test_f1)
            else:
                training_curve[1].append(test_MAE)
            fig.show(epoch_x,training_curve)

        trainResult = pd.DataFrame(training_curve).T
        if loss_function == 'WeightedFocalLoss':
            trainResult.columns = ['train_F1','val_F1']
        else:
            trainResult.columns = ['train_MAE','val_MAE']
        trainResult.to_csv(training_curve_file)
    return training_curve


def model_predict(data, model, device, loss_function, use_metadata):
    model.to(device)
    with torch.no_grad():
        x, edge_attr, edge_index, edge_type, batch = data.x.to(device), data.edge_attr.to(device), data.edge_index.to(device), data.edge_type.to(device), data.batch.to(device)
        
        if use_metadata:
            MetaData = data.MetaData.to(device)
            out = model(x, edge_attr, edge_index, edge_type, MetaData, batch)
        else:
            out = model(x, edge_attr, edge_index, edge_type, batch)

        if loss_function == 'WeightedFocalLoss': pred = (torch.sigmoid(out) >= 0.5).float().flatten()
        else: pred = out
        return pred