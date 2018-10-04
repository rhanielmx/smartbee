import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import getpass, platform, glob, os
import random
import requests
from bs4 import BeautifulSoup


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

#Função para calcular o Percentual de cada cluster em um determinado conjunto
def cluster_percentual(dataset):
    """
    Calcula o percentual de clusters existentes em um dataset.
    
    """
    data=pd.DataFrame(dataset)
    labels = data.iloc[:,-1].values
    clusters = np.unique(labels)
    percent_cluster=[]
    for cluster in clusters:
        cont = 0
        for i in labels:
            if i == cluster:
                cont = cont+1
        percent_cluster.append(100*cont/len(labels))      

    return(percent_cluster)


#Funções para remoção de outliers
def prepare_to_remove(dataset):
    """
    Calcula algumas estatísticas para a remoção de outliers do dataset
    
    Input
    ----------
    dataset: 
    Dataset a ser preparado para remoção de outliers.
    
    Output
    ----------
    cols:
    Lista contendo os nomes das colunas dos datasets.
  
    means:
    Lista contendo as médias das colunas dos datasets.
  
    stds:
    Lista contendo os desios padrões das colunas dos datasets.
  
    """
    cols, means, stds = [],[],[]
    for col in dataset.columns:
        try:
            cols.append(col)
            means.append(dataset[col].mean())
            stds.append(dataset[col].std())
        except TypeError:
            means.append(np.nan)
            stds.append(np.nan)
            print(f'Coluna {col} possui valores em formato não númerico!')
    return(cols, means, stds)

def remove_outliers(dataset):
    """
    Soma todos os elementos de um vetor, exceto o que está na posição n.
    
    Input
    ----------
    dataset: 
    Dataset para remoção de outliers.
    
    Output
    ----------
    dataset:
    O dataset da entrada, agora com os outliers removidos
    
    
    Obs.: São consideradas outliers da variável x, observações além do intervalo: mean(x) ± 2*std(x).
    """
    cols, means, stds = prepare_to_remove(dataset)
    k=0
    for col in cols:
        dataset = dataset.loc[(dataset[col] > means[k]-2*stds[k]) & (dataset[col] < means[k]+2*stds[k])]
        k=k+1
    return(dataset)


#Funções para dividir o dataset em conjuntos de treino, validação e teste

def remove_from_dataset(df,subset):
    df = pd.concat([df, subset, subset]).drop_duplicates(keep='first')
    df = pd.concat([df,df.iloc[0:1,:],df.iloc[0:1,:]]).drop_duplicates(keep=False)
    return df

def sum_except(lista, n):
    """
    Soma todos os elementos de uma tupla, exceto o que está na posição n.
    
    Input
    ----------
    lista: 
    Lista de elementos a serem somados.
    
    n: int
    Representa a posição do elemento que será excluído do somatório.

    
    Output
    ----------
    soma:
    Soma dos elementos que cumprem a condição requerida.
  
    """
    if n not in range(len(lista)):
        raise InputError('Posição não existente na lista.')
        
    soma=0
    for i in range(0,len(lista)):
        if i!=n:
            soma=soma+lista[i]
    return(soma)

def separator(dataframe, train_size=0.6, validate_size=0.2, test_size=0.2):
    """
    Separa um dataframe em conjuntos de treino, validação e teste.
    
    Input
    ----------
    dataframe: Dataframe que deseja-se dividir nos conjuntos de treino, validação e teste.
    
    train_size: float, int, None
    Se float, representa a porcentagem do dataset que será destinada
    ao conjunto de treino e deve estar entre 0 e 1. Se int, representa
    o valor absoluto do número de amostras destinadas ao treino. Se None,
    a quantidade de amostradas destinada ao treino será o complemento da
    quantidade destinada a teste e validação e os valores de validate_size
    e test_size não podem ser None. Valor default = 0.6.
    
    validate_size: float, int, None
    Se float, representa a porcentagem do dataset que será destinada
    ao conjunto de validação e deve estar entre 0 e 1. Se int, representa
    o valor absoluto do número de amostras destinadas à validação. Se None,
    a quantidade de amostradas destinada à validação será o complemento da
    quantidade destinada a treino e teste e os valores de validate_size
    e test_size não podem ser None. Valor default = 0.2.
    
    test_size: float, int, None
    Se float, representa a porcentagem do dataset que será destinada
    ao conjunto de teste e deve estar entre 0 e 1. Se int, representa
    o valor absoluto do número de amostras destinadas ao teste. Se None,
    a quantidade de amostradas destinada ao teste será o complemento da
    quantidade destinada a treino e validação e os valores de validate_size
    e test_size não podem ser None. Valor default = 0.2.
    
    Output
    ----------
    train_set:
    Conjunto de treino como uma instância de pandas.DataFrame().
    
    validate_set:
    Conjunto de validação como uma instância de pandas.DataFrame().
    
    test_set:
    Conjunto de teste como uma instância de pandas.DataFrame().\n
    
    
    """
    
    dataframe = dataframe.sample(frac=1)
    sizes = [train_size, validate_size, test_size]
    sizes_bool = [size is None for size in sizes]
    n_samples=[]
    
    if ((sizes_bool[0]) & (sizes_bool[1])) | ((sizes_bool[0]) & (sizes_bool[2])) | ((sizes_bool[1]) & (sizes_bool[2])):
        raise InputError('Apenas um dos tamanhos pode ser declarado como None!')
    
    for size in sizes:
        if type(size) == int:
            n_samples.append(size)
        elif type(size) == float:
            if not ((size >= 0) & (size <= 1)):
                raise InputError('Quando float, o valor deve estar entre 0 e 1!')
            n_samples.append(round(len(dataframe)*size))
        elif size is None:
            n_samples.append('placeholder')
        else:
            raise InputError('Insira um tamanho em formato válido!')
                
    for i in range(0,len(n_samples)):
        if n_samples[i] == 'placeholder':
            n_samples[i] = len(dataframe)-sum_except(n_samples,i)
            
    n_train=n_samples[0]
    n_validate=n_samples[1]
    n_test=n_samples[2]

    if (n_train+n_validate+n_test) > len(dataframe):
        n_train = n_train-1
    elif (n_train+n_validate+n_test) < len(dataframe):
        n_train = n_train+1
    
    if(n_train+n_validate+n_test!=len(dataframe)):
        raise InputError('A soma dos tamanhos dos conjuntos deve equivaler ao tamanho do dataset')

    dataframe.index=np.sort(dataframe.index)

    train_set = dataframe.iloc[:n_train,:]
    validate_set = dataframe.iloc[n_train:n_train+n_validate,:]
    validate_set.index= list(range(0,len(validate_set)))
    test_set = dataframe.iloc[n_train+n_validate:,:]
    test_set.index= list(range(0,len(test_set)))

    return(train_set,validate_set,test_set)
    
#Funções para dividir o dataset em conjuntos de treino, validação e teste levando em conta a proporção de cada classe

#Função para dividir o dataset nas classes fornecidas
def split_in_classes(data, labels):
    datasets = []
    
    if type(labels) == bool:
        return("labels deve ser uma lista com as labels ou o padrão None.")
   
    for label in labels:
        datasets.append(data[data.iloc[:,-1] == label])
        
    return datasets

#Função que reorganiza o dataframe de forma aleatória e corrige o índice
def shuffle_and_fix_index(dataframe):
    dataframe=dataframe.sample(frac=1)
    dataframe.index=np.sort(range(len(dataframe.index)))
    return dataframe
    
#Função que realmente separa o dataframe nos conjuntos de treino, validação e teste levando em conta as classes  
def separate_for_classes(dataframe, train_size=0.6, validate_size=0.2, test_size=0.2):
    labels = np.unique(dataframe.iloc[:,-1])
    n_classes = len(labels)
    datasets = []
    for i in range(n_classes):
        datasets.append(split_in_classes(data=dataframe, labels=labels)[i])
    
    new_datasets=[]    
    for dataset in datasets:
        new_datasets.append(separator(dataset,train_size=train_size, validate_size=validate_size, test_size=test_size))
    train_set, validate_set, test_set = new_datasets[0][0],new_datasets[0][1],new_datasets[0][2]
    
    for new_dataset in new_datasets[1:]:
        train_set=train_set.append(new_dataset[0])
        validate_set=validate_set.append(new_dataset[1])
        test_set=test_set.append(new_dataset[2])
    
    train_set=shuffle_and_fix_index(train_set)
    validate_set=shuffle_and_fix_index(validate_set)
    test_set=shuffle_and_fix_index(test_set)    
    
    return train_set, validate_set, test_set


#Função para pegar a base do diretório para o dataset

def get_base():

    user = getpass.getuser()
    so = platform.system()

    if so == 'Windows':
        base='C:/Users/'+user+'/Dropbox/Pesquisa/Notebooks/Arnas e Emil/Arnas/Datasets/'
    elif so == 'Linux':
        base='/home/'+user+'/Dropbox/Pesquisa/Notebooks/Arnas e Emil/Arnas/Datasets/'
    return(base)

#Função para pegar as urls de arquivos com certa extensão dentro de um outro diretório web
#Achei a get_url na internet e alterei para a get_files

def get_url_paths(url, ext='', params={}):
    """
    Retorna os nomes de arquivos com uma determinada extensão existentes em uma determinada url.
    
    Input
    ----------
    url: url onde irá procurar-se pelos arquivos.
    
    ext: 
    Extensão de arquivos desejada e.g. doc, pdf, csv.
    
    
    Output
    ----------
    parent:
    Tupla contendo as urls completas dos arquivos com a extensão desejada. 
    
    """
    response = requests.get(url, params=params)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    parent = [url+node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
    return parent

def get_files_paths(url, exts=[''], params={}):
    """
    Retorna os nomes de arquivos com uma determinada extensão existentes em uma determinada url.
    
    Input
    ----------
    url: url onde irá procurar-se pelos arquivos.
    
    ext: 
    Lista de strings com as extensões de arquivos desejada e.g. [doc, pdf, csv].
    
    
    Output
    ----------
    parent:
    Lista contendo os nomes dos arquivos com a extensão desejada. 
    
    """
    response = requests.get(url, params=params)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    parent = [[node.get('href').split('/')[-1] for node in soup.find_all('a') if node.get('href').endswith(ext)] for ext in exts]
    return(parent)
