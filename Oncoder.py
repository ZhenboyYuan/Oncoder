import pandas as pd
import random
from numpy.random import choice
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributions as dist
import torch.nn as nn
from scipy.stats import wilcoxon, ttest_ind
from torch.utils.data import Dataset
from numpy.random import choice
from torch.utils.data import Dataset
from numpy.random import choice
from scipy.stats import beta
import warnings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reproducibility(seed=1):

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def showloss(loss):
    sns.set()
    plt.plot(loss)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()


def showpcc(pred, truth):
    correlation_coefficient = np.corrcoef(pred[:, 1], truth[:, 1])[0, 1]
    print(correlation_coefficient)


def showrmse(pred, truth):
    RMSE = np.sqrt(np.mean((pred[:, 1] - truth[:, 1]) ** 2))
    print(RMSE)

class dataloader(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float().to(device)
        y = torch.from_numpy(self.Y[index]).float().to(device)
        return x, y


class Autoencoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.name = 'Oncoder'
        self.state = 'train'
        self.inputdim = input_dim
        self.outputdim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.inputdim, 512),
            nn.CELU(),

            nn.Dropout(),
            nn.Linear(512, 256),
            nn.CELU(),

            nn.Dropout(),
            nn.Linear(256, 128),
            nn.CELU(),

            nn.Dropout(),
            nn.Linear(128, 64),
            nn.CELU(),

            nn.Linear(64, self.outputdim))

        self.decoder = nn.Sequential(
            nn.Linear(self.outputdim, 64, bias=False),
            nn.Linear(64, 128, bias=False),
            nn.Linear(128, 256, bias=False),
            nn.Linear(256, 512, bias=False),
            nn.Linear(512, self.inputdim, bias=False))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def refraction(self, z):
        z_sum = torch.sum(z, dim=1, keepdim=True)
        return z / z_sum

    def methyatlas(self):
        w0 = (self.decoder[0].weight.T)
        w1 = (self.decoder[1].weight.T)
        w2 = (self.decoder[2].weight.T)
        w3 = (self.decoder[3].weight.T)
        w4 = (self.decoder[4].weight.T)
        w01 = (torch.mm(w0, w1))
        w02 = (torch.mm(w01, w2))
        w03 = (torch.mm(w02, w3))
        w04 = (torch.mm(w03, w4))
        return F.sigmoid(w04)

    def forward(self, x):
        methyatlas = self.methyatlas()
        z = self.encode(x)
        if self.state == 'train':
            pass
        elif self.state == 'test':
            z = F.relu(z)
            z = self.refraction(z)

        x_recon = torch.mm(z, methyatlas)
        return x_recon, z, methyatlas


class NLLloss(nn.Module):
    def __init__(self, df):
        super(NLLloss, self).__init__()
        self.alpha = torch.tensor(df['alpha'].astype('float64').values).to(device)
        self.beta = torch.tensor(df['beta'].astype('float64').values).to(device)
        self.mode = torch.tensor(df['mode'].astype('float64').values).to(device)
        self.mode_pdf = torch.tensor(df['mode_pdf'].astype('float64').values).to(device)
        return

    def forward(self, data, state):
        beta_distribution = dist.Beta(self.alpha, self.beta)
        if state == 'H':
            prob_data = beta_distribution.log_prob(torch.clamp(data[0], 0, 1)).exp()  
        elif state == 'T':
            prob_data = beta_distribution.log_prob(torch.clamp(data[1], 0, 1)).exp()
        normalized_pdf = prob_data / self.mode_pdf
        normalized_pdf = torch.clamp(normalized_pdf, 1e-20, 1)
        nll_loss = -torch.log(normalized_pdf)
        nll_loss = torch.mean(nll_loss) / 50
        return nll_loss  # .item()


def filterdata(data):
    """
    This function filters outliers from the data using a threshold based on 4 * standard deviation.
    Args:
        data: numpy.ndarray
            The input data array of shape (samples,).
        
    Returns:
        filtered_data: numpy.ndarray
            the filtered data array of shape (samples,).
    """
    mean_value = np.mean(data)
    std_dev = np.std(data)
    threshold = 4 * std_dev
    outliers_index = (data > mean_value + threshold) | (data < mean_value - threshold)
    filtered_data = data[~outliers_index]
    return filtered_data


def evaluateBetapara(filepath, typename):
    """
    This function evaluates the parameters of the beta distribution for each CpG site.
    Args:
        filepath: str
            The file path of reference data.
        typename: str
            The type of samples to evaluate ('plasma' or 'tumor').
    Returns:
        except_cpg: list
            A list of CpG sites that were excluded due to fitting errors.
        betadf: DataFrame
            A DataFrame containing the beta distribution parameters for each CpG site.
    """
    n = pd.read_csv(filepath, sep='\t', index_col=0, header=0)
    n.replace({0: 0.0001, 1: 0.9999}, inplace=True)
    n = n.filter(like=typename)
    dic = {}
    except_cpg = []
    for i, j in n.iterrows():
        try:
            dic[i] = list(beta.fit(filterdata(j.values), floc=0, fscale=1))
            mode = (dic[i][0] - 1) / (dic[i][0] + dic[i][1] - 2)
            mode = np.clip(mode, 0.001, 0.999)
            dic[i].append(mode)
            beta_distribution = dist.Beta(dic[i][0], dic[i][1])
            mode_pdf = beta_distribution.log_prob(torch.tensor(mode)).exp()
            dic[i].append(mode_pdf.item())
            if mode_pdf == 0:
                except_cpg.append(i)
        except RuntimeError as e:
            except_cpg.append(i)
            dic[i] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    betadf = pd.DataFrame.from_dict(dic, orient='index')
    betadf.drop(columns=[2, 3], inplace=True)
    betadf.columns = ['alpha', 'beta', 'mode', 'mode_pdf']
    return except_cpg, betadf


def training(model, data_loader, betadf, epochs=256, seed=1, lr=1e-4):
    """
    This function trains the Oncoder model.
    Args:
        model: Autoencoder
            The Oncoder model to be trained.
        data_loader: DataLoader
            The DataLoader containing training data and labels.
        betadf: list of DataFrame
            A list containing two DataFrames with beta distribution parameters for healthy and tumor samples.
        epochs: int, optional 
            The number of training epochs, by default 256.
        seed: int, optional
            A seed for the random number generator for reproducibility, 
            by default 1.
        lr: float, optional
            The learning rate for the optimizer, by default 1e-4.
    Returns:
        model : Autoencoder
            The trained Oncoder model.
        loss : list
            A list of loss values during training.
        recon_loss : list
            A list of reconstruction loss values during training.
        methyH_loss : list
            A list of healthy methylation loss values during training.
        methyT_loss : list
            A list of tumor methylation loss values during training.
    """
    decoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'decoder' in n]}]
    encoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'encoder' in n]}]
    optimizerD = torch.optim.Adam(decoder_parameters, lr=lr)
    optimizerE = torch.optim.Adam(encoder_parameters, lr=lr)
    optimizer = Adam(model.parameters(), lr=lr)
    loss = []
    recon_loss = []
    mylossH = NLLloss(betadf[0]).to(device)
    mylossT = NLLloss(betadf[1]).to(device)
    methyH_loss = []
    methyT_loss = []
    for i in range(epochs):
        model.train()
        model.state = 'train'
        epoch_comp_loss_sum = 0.0
        epoch_recon_loss_sum = 0.0
        epoch_methyH_loss_sum = 0.0
        epoch_methyT_loss_sum = 0.0
        batch_size = 0
        samples_size = 0
        
        for k, (data, label) in enumerate(data_loader):
            reproducibility(seed)
            batch_size = data.shape[0]
            samples_size += batch_size
            
            optimizer.zero_grad()
            x_recon, comp_prop, methy = model(data)
            batch_loss =  0.25 * F.l1_loss(comp_prop, label) + 0.25 * F.l1_loss(x_recon, data) + 0.25 * mylossH(methy,'H') + 0.5 * mylossT(methy, 'T')
            batch_loss.backward()
            optimizer.step()

            optimizerD.zero_grad()
            x_recon, _, methy = model(data)
            batch_loss =  F.l1_loss(x_recon, data) + mylossH(methy, 'H') + mylossT(methy, 'T')
            batch_loss.backward()
            optimizerD.step()

            optimizerE.zero_grad()
            x_recon, comp_prop, _ = model(data)
            batch_loss =  F.l1_loss(label, comp_prop) + F.l1_loss(x_recon, data)
            batch_loss.backward()
            optimizerE.step()

            loss.append(F.l1_loss(label, comp_prop).cpu().detach().numpy())
            recon_loss.append(F.l1_loss(x_recon, data).cpu().detach().numpy())
            methyH_loss.append(mylossH(methy, 'H').cpu().detach().numpy())
            methyT_loss.append(mylossT(methy, 'T').cpu().detach().numpy())
            
            epoch_comp_loss_sum +=  F.l1_loss(comp_prop, label).item() * batch_size
            epoch_recon_loss_sum += F.l1_loss(x_recon, data).item() * batch_size
            epoch_methyH_loss_sum += mylossH(methy,'H').item() * batch_size
            epoch_methyT_loss_sum += mylossT(methy, 'T').item() * batch_size
            
        if samples_size > 0:
            avg_comp = epoch_comp_loss_sum / samples_size
            avg_recon = epoch_recon_loss_sum / samples_size
            avg_methyH = epoch_methyH_loss_sum / samples_size
            avg_methyT = epoch_methyT_loss_sum / samples_size
            print(f"Epoch {i+1}/{epochs} | Comp: {avg_comp:.4f} | Recon: {avg_recon:.4f} | Methy-H: {avg_methyH:.4f} | Methy-T: {avg_methyT:.4f}")
    return model, loss, recon_loss, methyH_loss, methyT_loss


def train_Oncoder(train_x, train_y,refpath, model_name=None, batch_size=128, epochs=256, seed=1,lr=1e-4):
    """
    This function trains the Oncoder model.
    
    Args:
        train_x: numpy.ndarray
            The training data matrix of shape (samples, cpg_sites).
        train_y: numpy.ndarray
            The training labels matrix of shape (samples, tumor_fractions).
        refpath: str
            The file path of reference data.
        model_name: str, optional
            The name to save the trained model, by default None.
        batch_size: int, optional
            The batch size for training, by default 128.
        epochs: int, optional
            The number of training epochs, by default 256.
        seed: int, optional
            A seed for the random number generator for reproducibility, 
            by default 1.
        lr: float, optional
            The learning rate for the optimizer, by default 1e-4.
    Returns:
        model: Autoencoder
            The trained Oncoder model. 
    """
    print('Loading data')
    data_loader = DataLoader(dataloader(train_x, train_y), batch_size=batch_size, shuffle=True)
    model = Autoencoder(train_x.shape[1], train_y.shape[1]).to(device)
    _, H_beta = evaluateBetapara(refpath, 'plasma')
    _, T_beta = evaluateBetapara(refpath, 'tumor')
    beta_df = [H_beta, T_beta]
    print('Start training')
    model, loss, recon_loss, methyH_loss, methyT_loss = training(model, data_loader, beta_df, epochs=epochs, seed=seed, lr=lr)
    print('training done')
    showloss(loss)
    showloss(recon_loss)
    showloss(methyH_loss)
    showloss(methyT_loss)
    if model_name is not None:
        print('Model is saved')
        torch.save(model, model_name + '.pth')
    return model


def predict(test_x, model=None, model_path=None):
    """
    This function predicts tumor fractions using the trained Oncoder model.

    Args:
        test_x: numpy.ndarray: 
            The test data matrix of shape (samples, cpg_sites).
        model: Autoencoder: 
            The pre-loaded trained Oncoder model.
        model_name: str, optional: 
            The file path of the saved trained model. Defaults to None.

    Returns:
        pred: numpy.ndarray: 
            The predicted tumor fractions of shape (samples, tumor_fractions).
        methyatlas: torch.Tensor: 
            The methylation atlas learned by the model.
    """
    if model is None and model_path is None:
        raise ValueError("Either model or model_path must be provided.")
    print('Start prediction')
    if model is not None and model_path is not None:
        warnings.warn("Both model and model_path are provided. The model will be used for prediction.",UserWarning)
        
    if model is None:
        print('loading model')
        model = torch.load(model_path)
    model.eval()
    model.state = 'test'
    data = torch.from_numpy(test_x).float().to(device)
    _, pred, methyatlas = model(data)
    pred = pred.cpu().detach().numpy()
    print('predicition is done')
    return pred, methyatlas


def generate_simulated_data(refpath, prior=[0.9, 0.1], samplenum=5000, random_state=1, method='Dirichlet'):
    """
    This function generates simulated data with tumor fraction labels based on dirichlet or uniform distribution.
    
    Args:
        refpath: the file path of reference data, str
            The file contains a matrix of CpG sites by samples. The index should be CpG sites, and columns should represent sample types 
            ('plasma_1', 'plasma_2',...'plasma_n', 'tumor_1', 'tumor_2',...'tumor_n'). Should not contain null values.
        samplenum: int, optional
            The number of simulated samples to generate, by default 5000.
        random_state: int, optional
            A seed for the random number generator for reproducibility, 
            by default 1.
        method: {'Dirichlet', 'Uniform'}, optional
            The method to generate tumor fractions, by default 'Dirichlet'.
    
    Returns: 
        x: numpy.ndarray
            The simulated data matrix of shape (samples, cpg_sites).
        y: numpy.ndarray
            The simulated labels matrix of shape (samples, tumor_fractions). Column 1: Healthy fractions; Column 2: Tumor fractions.
    """
    
    print("reading ref dataset")
    n = pd.read_csv(refpath, sep='\t', index_col=0, header=0)
    n.columns = [i.split('_')[0] for i in n.columns]
    n = n.T
    n['sampletype'] = n.index
    n.index = range(len(n))
    cpgname = n.columns[:-1]
    sampletype_groups = n.groupby("sampletype").groups
    n.drop(columns="sampletype", inplace=True)
    for key, value in sampletype_groups.items():
        sampletype_groups[key] = np.array(value)
    np.random.seed(random_state)
    np.set_printoptions(precision=4, suppress=True)
    if method == 'Dirichlet':
        prop = np.random.dirichlet(prior, samplenum)
        prop = prop / np.sum(prop, axis=1).reshape(-1, 1)
    elif method == 'Uniform':
        data_1 = np.random.uniform(prior[0], 1, samplenum)
        data_2 = np.random.uniform(0, prior[1], samplenum)
        prior_distribution_data = np.column_stack((data_1, data_2))
        prop = prior_distribution_data / np.sum(prior_distribution_data, axis=1, keepdims=True)
    sample = np.zeros((prop.shape[0], n.shape[1]))
    prop = pd.DataFrame(prop, columns=['H','T'])
    for i, sample_prop in tqdm(prop.iterrows()):
        sample[i] = sample_prop["H"] * n.iloc[choice(sampletype_groups["plasma"])] + sample_prop["T"] * n.iloc[choice(sampletype_groups["tumor"])]
    train_x = pd.DataFrame(sample, columns=cpgname)
    train_y = prop
    return train_x.values, train_y.values

def adaptive_learning(model, X, y, refdatapath, epochs=256, seed=1, batch_size=128, lr=1e-4):
    """
    This function performs adaptive learning on the Oncoder model using new data.
    Args:
        model: Autoencoder
            The pre-trained Oncoder model to be fine-tuned.
        X: numpy.ndarray
            The new data matrix of shape (samples, cpg_sites).
        y: numpy.ndarray
            The new labels matrix of shape (samples, tumor_fractions).
        refdatapath: str
            The file path of additional reference data.
        epochs: int, optional
            The number of training epochs for fine-tuning, by default 256.
        seed: int, optional
            A seed for the random number generator for reproducibility, 
            by default 1.
        batch_size: int, optional
            The batch size for training, by default 128.
        lr: float, optional
            The learning rate for the optimizer, by default 1e-4.
    Returns:
        model: Autoencoder
            The fine-tuned Oncoder model.
        loss: list
            A list of loss values during fine-tuning.
        recon_loss: list
            A list of reconstruction loss values during fine-tuning.
        methyH_loss: list
            A list of healthy methylation loss values during fine-tuning.
        methyT_loss: list
            A list of tumor methylation loss values during fine-tuning.
    """
    print('adaptive stage')
    data_loader = DataLoader(dataloader(X, y), batch_size=batch_size, shuffle=True)
    decoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'decoder' in n]}]
    encoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'encoder' in n]}]
    optimizerD = torch.optim.Adam(decoder_parameters, lr=lr)
    optimizerE = torch.optim.Adam(encoder_parameters, lr=lr)
    optimizer = Adam(model.parameters(), lr=lr)
    loss = []
    recon_loss = []
    _, H_beta, _ = evaluateBetapara(refdatapath, 'plasma')
    _, T_beta, _ = evaluateBetapara(refdatapath, 'tumor')
    betadf = [H_beta, T_beta]
    mylossH = NLLloss(betadf[0]).to(device)
    mylossT = NLLloss(betadf[1]).to(device)
    methyH_loss = []
    methyT_loss = []
    
    model.train()
    model.state = 'train'
    for i in tqdm(range(epochs)):
        for k, (data, label) in enumerate(data_loader):
            reproducibility(seed)
            optimizer.zero_grad()
            x_recon, comp_prop, methy = model(data)
            batch_loss = 0.25 * F.l1_loss(comp_prop, label) + 0.25 * F.l1_loss(x_recon, data) + 0.25 * mylossH(methy,'H') + 0.5 * mylossT(methy, 'T')
            batch_loss.backward()
            optimizer.step()

            optimizerD.zero_grad()
            x_recon, _, methy = model(data)
            batch_loss = F.l1_loss(x_recon, data) + mylossH(methy, 'H') + mylossT(methy, 'T')
            batch_loss.backward()
            optimizerD.step()

            optimizerE.zero_grad()
            x_recon, comp_prop, _ = model(data)
            batch_loss = F.l1_loss(label, comp_prop) + F.l1_loss(x_recon, data)
            batch_loss.backward()
            optimizerE.step()

            loss.append(F.l1_loss(label, comp_prop).cpu().detach().numpy())
            recon_loss.append(F.l1_loss(x_recon, data).cpu().detach().numpy())
            methyH_loss.append(mylossH(methy, 'H').cpu().detach().numpy())
            methyT_loss.append(mylossT(methy, 'T').cpu().detach().numpy())
        
    model.eval()
    model.state = 'test'
    return model, loss, recon_loss, methyH_loss, methyT_loss

