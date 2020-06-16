#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[8]:


black_friday = pd.read_csv("black_friday.csv")
black_friday


# ## Inicie sua análise a partir daqui

# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[12]:


def q1():
    """
    Answer question 1
    
    Returns
    -------
    tuple
        A tuple with quantity of instances and features in the following
        format `(instance_qtd, features_qtd)`
    """
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[22]:


def q2():
    """
    Answer question 2
    
    Returns
    -------
    int
        Number of women between 26 and 35 years old
    """
    # Generate the mask to be used on indexing
    female_mask = black_friday['Gender']=='F'
    age_26_35_mask = black_friday['Age']=='26-35'

    # Dataset with only women with age between 26 and 35 years old
    female_26_35 = black_friday[age_26_35_mask & female_mask]
    
    return female_26_35.shape[0]
    


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[32]:


def q3():
    """
    Answer question 3
    
    Returns
    -------
    int
        Unique users on dataset
    """
    # Drop all users duplicated
    unique_users = black_friday['User_ID'].drop_duplicates() 
    
    return unique_users.size


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[37]:


def q4():
    """
    Answer question 4
    
    Returns
    -------
    int
        Number of diferent datatypes on the dataset
    """
    return black_friday.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[20]:


def q5():
    """
    Answer question 5
    
    Returns
    -------
    float
        Relation between number of lines with Nan and number of 
        all lines
    """
    lines_qtd = black_friday.shape[0]
    without_nan = black_friday.dropna().shape[0]
    
    return 1 - (without_nan/lines_qtd)


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[11]:


def q6():
    """
    Answer question 6
    
    Returns
    -------
    int
        Number of missing data of the feature with the highest number
        of `NaN` 
    """
    total_size = black_friday.shape[0]
    
    # Number of values of the feature with the highest NaN occurrence
    lowest_data_qtd = black_friday.count().min()
    
    return int(total_size-lowest_data_qtd)


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[87]:


def q7():
    """
    Answer question 7
    
    Returns
    -------
    int
        Number of missing data of the feature with the highest number
        of `NaN` 
    """
    return int(black_friday['Product_Category_3'].mode()[0])


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[110]:


def q8():
    """
    Answer question 8
    
    Returns
    -------
    float
        Purchase mean after its normalization
    """
    purchase = black_friday['Purchase']

    purchase_min = purchase.min()
    purchase_max = purchase.max()

    purchase_norm = (purchase-purchase_min) / (purchase_max-purchase_min)
    
    return float(purchase_norm.mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[102]:


def q9():
    """
    Answer question 9
    
    Returns
    -------
    int
        Number of purchases between -1 and 1 after its standardization
    """
    purchase = black_friday['Purchase']
    
    purchase_mean = purchase.mean()
    purchase_std = purchase.std()
    
    purchase_standardized = (purchase-purchase_mean) / purchase_std
    
    purchase_mask = purchase_standardized.between(-1, 1)
    
    return purchase[purchase_mask].size


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[107]:


def q10():
    """
    Answer question 10
    
    Returns
    -------
    bool
        `True` if all missing data in `Product_Category_3` is also a 
        missing value in `Product_Category_2` and `False` otherwise.
    """
    
    product_cat_2_mask = black_friday['Product_Category_2'].isna()
    product_cat_3 = black_friday['Product_Category_3']
    
    product_cat_3_answer = product_cat_3[product_cat_2_mask]
    
    return product_cat_3_answer.dropna().empty

q10()


# In[ ]:




