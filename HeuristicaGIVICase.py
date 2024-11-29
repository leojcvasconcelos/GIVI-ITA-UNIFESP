#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import matplotlib.pyplot as plt
#import pyomo.environ as pyo
#from pyomo.opt import SolverFactory
from typing import Tuple


# In[2]:


# Função para ler os processing_times de processamento de uma planilha Excel
def ler_planilha(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=0)  # Lendo a primeira planilha
    df = df.drop(df.columns[[0,1,2,27,28,29,30]], axis=1)
    df = df.drop([0,1,32,33,34,35,36], axis=0)
    df = df.drop(df.columns[0], axis=1)
    df = df.fillna(0).infer_objects(copy=False)
    df = df.round(2)
    df = df.reset_index(drop=True)
    df.columns = range(df.shape[1])
    return df


# In[3]:




# Função para aplicar a heurística NEH
def neh(processing_times: pd.DataFrame, params: list) -> pd.Index:
    num_jobs = processing_times.shape[0]

    # Passo 1: Ordena os jobs com base na soma dos processing_times de processamento
    soma_processing_times = np.sum(processing_times, axis=1)
    ordem_inicial = np.argsort(-soma_processing_times)  # Ordena em ordem decrescente

    # Passo 2: Constrói a solução sequencialmente
    sequencia_final = [ordem_inicial[0]]
    for i in range(1, num_jobs):
        melhor_sequencia = None
        melhor_obj = float('inf')

        # Tenta inserir o próximo job em todas as posições possíveis da sequência atual
        for posicao in range(len(sequencia_final) + 1):
            nova_sequencia = sequencia_final[:posicao] + [ordem_inicial[i]] + sequencia_final[posicao:]
            novo_completion_times = calcular_completion_times(processing_times.loc[nova_sequencia] ,pd.Index(nova_sequencia))
            novo_obj, novo_makespan, novo_fluxo_total, novo_times_espera, novo_qtde_esperas, novo_esperas_simul = obf_func(processing_times, novo_completion_times, nova_sequencia, params)
            
            # Se a nova sequência é melhor, atualiza a melhor sequência
            if novo_obj < melhor_obj:
                melhor_qtde_esperas = novo_qtde_esperas
                melhor_sequencia = nova_sequencia
                melhor_makespan = novo_makespan
                melhor_fluxo_total = novo_fluxo_total
                melhor_obj = novo_obj
                melhor_times_espera = novo_times_espera
                melhor_esperas_simul = novo_esperas_simul


        # Atualiza a sequência final com a melhor sequência encontrada
        sequencia_final = melhor_sequencia
        finais = [melhor_obj, melhor_makespan, melhor_fluxo_total, melhor_times_espera, melhor_qtde_esperas, melhor_esperas_simul]
        
    return pd.Index(sequencia_final), finais


# In[6]:


def busca_local_swap(job_order: pd.Index, processing_times: pd.DataFrame, params: list) -> Tuple[pd.Index, list]:
    """
    Procura uma melhor solução a partir de uma solução inicial a partir do swap e devolve a nova sequência e todos os resultados necessários.

    Args:
        processing_times (pd.DataFrame): 
            DataFrame onde cada linha representa um job e cada coluna representa o tempo de processamento em uma máquina específica, lembrar de passar como argumento o dataframe ordenado de acordo com job_order.
        job_order (pd.Index): 
            Índice com a ordem dos jobs a serem processados. Define a sequência de execução.

    Returns:
        pd.DataFrame:
            DataFrame com os tempos de conclusão dos jobs em cada máquina.
        list: todas as variáveis necessárias para o algoritmo (FO, makespan, fluxo total, tempo total de espera, numero de acessos ao buffer)

    """
    melhorou = True
    melhor_sequencia = job_order.copy()
    melhor_completion_times = calcular_completion_times(processing_times.loc[melhor_sequencia],melhor_sequencia)
    melhor_obj, melhor_makespan, melhor_fluxo_total, melhor_times_espera, melhor_qtde_esperas, melhor_esperas_simul = obf_func(processing_times, melhor_completion_times, melhor_sequencia, params)
    

    while melhorou:
        melhorou = False

        # Tenta todas as trocas possíveis de dois jobs
        for i in range(len(melhor_sequencia)):
            for j in range(i + 1, len(melhor_sequencia)):
                # Cria uma nova sequência com os jobs i e j trocados
                nova_sequencia = melhor_sequencia.copy().tolist()
                nova_sequencia[i], nova_sequencia[j] = nova_sequencia[j], nova_sequencia[i]
                nova_sequencia = pd.Index(nova_sequencia)

                # Calcula o obj da nova sequência
                novo_completion_times = calcular_completion_times(processing_times.loc[nova_sequencia],nova_sequencia)
                novo_obj, novo_makespan, novo_fluxo_total, novo_times_espera, novo_qtde_esperas, novo_esperas_simul = obf_func(processing_times, novo_completion_times, nova_sequencia, params)

                # Se a nova sequência é melhor, atualiza a melhor sequência
                if novo_obj < melhor_obj:
                    melhor_qtde_esperas = novo_qtde_esperas
                    melhor_sequencia = nova_sequencia
                    melhor_makespan = novo_makespan
                    melhor_fluxo_total = novo_fluxo_total
                    melhor_obj = novo_obj
                    melhor_times_espera = novo_times_espera
                    melhor_esperas_simul = novo_esperas_simul
                    melhorou = True

    sequencia_final = melhor_sequencia
    finais = [melhor_obj, melhor_makespan, melhor_fluxo_total, melhor_times_espera, melhor_qtde_esperas, melhor_esperas_simul]

    return sequencia_final, finais


def busca_local_swap_sol_alternativas(job_order: pd.Index, processing_times: pd.DataFrame, params: list) -> Tuple[pd.Index, list]:
    """
    Procura uma melhor solução a partir de uma solução inicial a partir do swap e devolve a nova sequência e todos os resultados necessários.

    Args:
        processing_times (pd.DataFrame): 
            DataFrame onde cada linha representa um job e cada coluna representa o tempo de processamento em uma máquina específica, lembrar de passar como argumento o dataframe ordenado de acordo com job_order.
        job_order (pd.Index): 
            Índice com a ordem dos jobs a serem processados. Define a sequência de execução.

    Returns:
        pd.DataFrame:
            DataFrame com os tempos de conclusão dos jobs em cada máquina.
        list: todas as variáveis necessárias para o algoritmo (FO, makespan, fluxo total, tempo total de espera, numero de acessos ao buffer)

    """
    melhorou = True
    melhor_sequencia = job_order.copy()
    melhor_completion_times = calcular_completion_times(processing_times.loc[melhor_sequencia],melhor_sequencia)
    melhor_obj, melhor_makespan, melhor_fluxo_total, melhor_times_espera, melhor_qtde_esperas, melhor_esperas_simul = obf_func(processing_times, melhor_completion_times, melhor_sequencia, params)
    

    while melhorou:
        melhorou = False

        # Tenta todas as trocas possíveis de dois jobs
        for i in range(len(melhor_sequencia)):
            for j in range(i + 1, len(melhor_sequencia)):
                # Cria uma nova sequência com os jobs i e j trocados
                nova_sequencia = melhor_sequencia.copy().tolist()
                nova_sequencia[i], nova_sequencia[j] = nova_sequencia[j], nova_sequencia[i]
                nova_sequencia = pd.Index(nova_sequencia)

                # Calcula o obj da nova sequência
                novo_completion_times = calcular_completion_times(processing_times.loc[nova_sequencia],nova_sequencia)
                novo_obj, novo_makespan, novo_fluxo_total, novo_times_espera, novo_qtde_esperas, novo_esperas_simul = obf_func(processing_times, novo_completion_times, nova_sequencia, params)

                # Se a nova sequência é melhor, atualiza a melhor sequência
                if novo_obj < melhor_obj and novo_fluxo_total <= melhor_fluxo_total:
                    melhor_qtde_esperas = novo_qtde_esperas
                    melhor_sequencia = nova_sequencia
                    melhor_makespan = novo_makespan
                    melhor_fluxo_total = novo_fluxo_total
                    melhor_obj = novo_obj
                    melhor_times_espera = novo_times_espera
                    melhor_esperas_simul = novo_esperas_simul
                    melhorou = True

    sequencia_final = melhor_sequencia
    finais = [melhor_obj, melhor_makespan, melhor_fluxo_total, melhor_times_espera, melhor_qtde_esperas, melhor_esperas_simul]

    return sequencia_final, finais





# In[7]:


def calcular_makespan(completion_times: pd.DataFrame) -> float:
    makespan = np.max(completion_times)
    return makespan

def calcular_fluxo_total(completion_times: pd.DataFrame) -> float:
    fluxo_total = np.sum(np.max(completion_times, axis=0),axis=0)
    return fluxo_total


# In[8]:


def calcular_processing_times_qtde_esperas(job_order: pd.Index, completion_times: pd.DataFrame, processing_times: pd.DataFrame) -> Tuple[float, int, int]:
    # Ajuste para máquinas paralelas (combina colunas 15 e 16)
    completion_times.loc[:, 15] = completion_times.loc[:, 15] + completion_times.loc[:, 16]
    completion_times = completion_times.drop(completion_times.columns[16], axis=1)
    completion_times.columns = range(completion_times.shape[1])
    completion_matrix = completion_times.values

    processing_times = processing_times.loc[job_order].reset_index(drop=True)
    processing_times.loc[:, 15] = processing_times.loc[:, 15] + processing_times.loc[:, 16]
    processing_times = processing_times.drop(processing_times.columns[16], axis=1)
    processing_times.columns = range(processing_times.shape[1])

    num_jobs, num_machines = processing_times.shape
    diff = np.zeros((num_jobs, num_machines))  # Matriz para diferenças
    
    processing_times_de_espera = []
    quantidade_de_esperas = []

    # Para calcular as esperas simultâneas
    eventos = []

    for i in range(num_jobs):
        indices_ativos = processing_times.columns[processing_times.loc[i, :] != 0]
        
        # Calcular as diferenças de tempos
        for j in range(len(indices_ativos[::-1]) - 1):
            diff[i, indices_ativos[-j - 1]] = (
                completion_matrix[i, indices_ativos[-j - 1]] - 
                completion_matrix[i, indices_ativos[-j - 2]]
            )
        
        diff[i, :] = diff[i, :].round(2) - processing_times.iloc[i].values
        
        # Filtrar valores positivos
        valores_positivos = diff[i, :][diff[i, :] > 0]
        quantidade_de_esperas.append(len(valores_positivos))
        processing_times_de_espera.append(np.sum(valores_positivos, axis=0))

        # Registrar eventos de início e término para cada espera positiva
        for idx, valor in enumerate(diff[i, :]):
            if valor > 0:
                inicio = completion_matrix[i, idx] - valor
                fim = completion_matrix[i, idx]
                eventos.append((inicio, 'inicio'))
                eventos.append((fim, 'fim'))

    # Ordenar eventos pelo tempo (priorizando 'fim' em empates)
    eventos.sort(key=lambda x: (x[0], x[1] == 'inicio'))
    
    # Calcular número máximo de esperas simultâneas
    esperas_simultaneas = 0
    max_esperas = 0

    for _, tipo in eventos:
        if tipo == 'inicio':
            esperas_simultaneas += 1
            max_esperas = max(max_esperas, esperas_simultaneas)
        elif tipo == 'fim':
            esperas_simultaneas -= 1

    return (
        np.sum(processing_times_de_espera, axis=0), 
        np.sum(quantidade_de_esperas, axis=0), 
        max_esperas
    )


# In[9]:


def obf_func(processing_times: pd.DataFrame, completion_times: pd.DataFrame, job_order: pd.Index, params: list) -> Tuple[float, float, float, float, int, int]:
    makespan = calcular_makespan(completion_times)
    fluxo_total = calcular_fluxo_total(completion_times)
    tempo_de_espera, qtde_esperas,esperas_simul = calcular_processing_times_qtde_esperas(job_order,completion_times,processing_times)
    alpha = params[0]
    beta = params[1]
    gama = params[2]
    theta = params[3]
    omega = params[4]
    obj = alpha*makespan/4934.44 + beta*fluxo_total/52002.63 + gama*tempo_de_espera/4313.03 + theta*qtde_esperas/21 + omega*esperas_simul/4
    return obj, makespan, fluxo_total, tempo_de_espera, qtde_esperas,esperas_simul


# In[10]:


# Função para calcular a matriz de temos de finalização para uma dada ordem de jobs
def calcular_completion_times(processing_times: pd.DataFrame, job_order: pd.Index) -> pd.DataFrame:
    num_jobs, num_machines = processing_times.shape
    completion_times = np.zeros((num_jobs, num_machines))  # Matriz de processing_times de conclusão
    processing_matrix = processing_times.values
    # Primeiro job
    completion_times[0, 0] = processing_matrix[0,0]
    for m in range(1, num_machines):
        if processing_matrix[0, m] != 0:
            completion_times[0, m] = max(completion_times[0, list(range(0,(m)))]) + processing_matrix[0, m]
        else:
            completion_times[0, m] = 0

    # Restante dos jobs
    for j in range(1, num_jobs):
        if processing_matrix[j, 0] != 0:
            completion_times[j, 0] = max(completion_times[list(range(0,(j))), 0]) + processing_matrix[j, 0]
        else:
            completion_times[j, 0] = 0

        for m in range(1, 15):
            if processing_matrix[j, m] != 0:
                completion_times[j, m] = max(max(completion_times[list(range(0,(j))), m]), max(completion_times[j, list(range(0,(m)))])) + processing_matrix[j, m]
            else:
                completion_times[j, m] = 0

        if processing_matrix[j, 15] != 0:
            if (np.max(completion_times[:j, 16]) > np.max(completion_times[:j, 15])):
                completion_times[j, 15] = max(max(completion_times[list(range(0,(j))), 15]), max(completion_times[j, list(range(0,(15)))])) + processing_matrix[j, 15]
            else:
                completion_times[j, 16] = max(max(completion_times[list(range(0,(j))), 16]), max(completion_times[j, list(range(0,(15)))])) + processing_matrix[j, 15]
        else:
            completion_times[j, 15] = 0
            completion_times[j, 16] = 0

        for m in range(17, num_machines):
            if processing_matrix[j, m] != 0:
                completion_times[j, m] = max(max(completion_times[list(range(0,(j))), m]), max(completion_times[j, list(range(0,(m)))])) + processing_matrix[j, m]
            else:
                completion_times[j, m] = 0

    return pd.DataFrame(completion_times)


# In[11]:


# Função para gerar o gráfico de Gantt
def plot_gantt_chart_J(processing_times: pd.DataFrame, job_order: pd.Index, completion_times: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    num_jobs, num_machines = processing_times.shape
    completion_times = completion_times.values

    # Iterando sobre cada job e cada máquina para construir as barras
    for j in range(num_jobs):
        job_index = job_order[j]
        start_time = 0
        for m in range(num_machines):
            if m == 15 or m == 16:
                proc_time = processing_times.loc[job_index].iloc[15]
            else:
                proc_time = processing_times.loc[job_index].iloc[m]
            
            if proc_time <= 0 or completion_times[j, m] <= 0:
                ax.barh(y=f"Job {job_index+1}", width=0, left=0, color=f"C{m}", edgecolor="black")
                continue

            end_time = completion_times[j, m]
            start_time = max(0, end_time - proc_time)
            
            # Adicionando barra para cada tarefa no gráfico de Gantt
            ax.barh(y=f"Job {job_index+1}", width=proc_time, left=start_time, color=f"C{m}", edgecolor="black")
            ax.text(start_time + proc_time / 2, j, f"M{m+1}", ha="center", va="center", color="black", fontsize=8)

    ax.set_xlabel("Tempo")
    ax.set_ylabel("Jobs")
    ax.set_title("Gráfico de Gantt")
    ax.invert_yaxis()
    # Obter os limites do eixo X
    min_x, max_x = ax.get_xlim()

    # Definir os ticks a cada múltiplo de 600, até o máximo do eixo X
    xticks = np.arange(0, max_x, 600)  # Garante que o último múltiplo seja até o limite máximo

    # Definir as marcações no eixo X
    ax.set_xticks(xticks)

    # Exibir o grid
    ax.grid(True, which='both', axis='x', linestyle='--', alpha=0.5)
    plt.show()
    plt.close(fig)

def plot_gantt_chart_M(processing_times: pd.DataFrame, job_order: pd.Index, completion_times: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 8))
    num_jobs, num_machines = processing_times.shape
    completion_times = completion_times.values

    colors = plt.cm.tab20.colors  # Conjunto de cores para diferenciar os jobs

    # Iterando sobre cada máquina para plotar as tarefas
    for m in range(num_machines):
        #start_time = 0
        for j in range(num_jobs):
            job_index = job_order[j]
            if m == 15 or m == 16:
                proc_time = processing_times.loc[job_index].iloc[15]
            else:
                proc_time = processing_times.loc[job_index].iloc[m]

            if proc_time <= 0 or completion_times[j, m] <= 0:
                ax.barh(y=f"M{m+1}", width=0, left=0, color=colors[j % len(colors)], edgecolor="black")
                continue

            end_time = completion_times[j, m]
            start_time = max(0, end_time - proc_time)

            ax.barh(y=f"M{m+1}", width=proc_time, left=start_time, color=colors[j % len(colors)], edgecolor="black")
            ax.text(start_time + proc_time / 2, m, f"J{job_index+1}", ha="center", va="center", color="black", fontsize=8)
    
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Máquinas")
    ax.set_title("Gráfico de Gantt por Máquina")
    ax.invert_yaxis()
    # Obter os limites do eixo X
    min_x, max_x = ax.get_xlim()

    # Definir os ticks a cada múltiplo de 600, até o máximo do eixo X
    xticks = np.arange(0, max_x, 600)  # Garante que o último múltiplo seja até o limite máximo

    # Definir as marcações no eixo X
    ax.set_xticks(xticks)

    # Exibir o grid
    ax.grid(True, which='both', axis='x', linestyle='--', alpha=0.5)
    plt.show()
    plt.close(fig)
    
# In[12]:

def ajustar_processing_times(completion_times, processing_times, tempo_atual,job_order):
    """
    Cria cópias de completion_times e processing_times e ajusta os tempos de processamento 
    para zero para as atividades já finalizadas até o instante especificado (tempo_atual).
    
    Args:
        completion_times (pd.DataFrame): DataFrame com tempos de término (completion times).
        processing_times (pd.DataFrame): DataFrame com tempos de processamento.
        tempo_atual (int): Tempo limite para considerar atividades como finalizadas.
    
    Returns:
        pd.DataFrame: Cópia ajustada de processing_times.
    """
    # Criar cópias dos DataFrames
    completion_times_copy = completion_times.copy()
    processing_times_copy = processing_times.copy()

    #print(processing_times_copy)
    #print(completion_times_copy)

    # Iterar sobre as atividades e ajustar os tempos
    for job in completion_times_copy.index:  # Itera pelas linhas (jobs)
        for machine in completion_times_copy.columns:  # Itera pelas colunas (máquinas)
            if machine == 15 or machine == 16:
                if completion_times_copy.loc[job, 15] <= tempo_atual and completion_times_copy.loc[job, 16] <= tempo_atual:
                    processing_times_copy.loc[job_order[job], 15] = 0
            else:
                if completion_times_copy.loc[job, machine] <= tempo_atual:
                    processing_times_copy.loc[job_order[job], machine] = 0

    #print(processing_times_copy)
    return processing_times_copy

# In[13]:


# Função principal para rodar a heurística, calcular o makespan e gerar o gráfico de Gantt
def main(file_path):
    processing_times = ler_planilha(file_path)

    #print(processing_times)

    #pesos para makespan,fluxo_total,tempo_esperas,qtde_esperas,esperas simultaneas
    params = 0,0.5,0.5,0,0

    job_order, vec_infos = neh(processing_times, params)
    print("\nOrdem inicial dos jobs pela heurística:")
    print(job_order)
    print(vec_infos)
    
    job_order, vec_infos = busca_local_swap(job_order,processing_times, params)
    print("\nOrdem final dos jobs pela busca local:")
    print(job_order)

    completion_times = calcular_completion_times(processing_times.loc[job_order], job_order)
    
    print("\nTempo de término do job (linha i) na maquina (coluna m):")
    print(pd.DataFrame(completion_times))
    
    makespan = vec_infos[1]
    print(f"\nMakespan para a ordem calculada: {makespan}")
    fluxo = vec_infos[2]
    print(f"\nFluxo Total para a ordem calculada: {fluxo}")
    processing_times_de_espera = vec_infos[3]
    print(f"\nTempo total de esperas: {processing_times_de_espera}")
    quantidade_de_esperas = vec_infos[4]
    print(f"\nQTDE de esperas: {quantidade_de_esperas}")
    esperas_simultaneas = vec_infos[5]
    print(f"\nQTDE de esperas simultaneas: {esperas_simultaneas}")

    # Gerar o gráfico de Gantt
    plot_gantt_chart_J(processing_times, job_order, completion_times)
    plot_gantt_chart_M(processing_times, job_order, completion_times)

    #####################################################################################
    #            INCLUINDO UM ITEM NOVO DURANTE A EXECUÇÃO DO SEQUENCIAMENTO            #
    #####################################################################################

    tempo_atual = 4800  #final do oitavo dia
    # Ajustar tempos de processamento
    processing_times_ajustados = ajustar_processing_times(completion_times, processing_times, tempo_atual, job_order)
    #print(processing_times_ajustados)

    # Adicionar um job existente como job adicional
    job_a_copiar = 13  # Índice do job existente a ser copiado (item com mais tempo de processamento e mais tempo nos robos)
    novo_job_indice = len(processing_times_ajustados)  # Novo índice do job adicional
    novo_job = processing_times.loc[job_a_copiar].copy()  # Copia os tempos do job
    processing_times_ajustados.loc[novo_job_indice] = novo_job  # Adiciona o job ao DataFrame
    #print(processing_times_ajustados)

    novo_job_order, vec_infos = neh(processing_times_ajustados, params)
    print("\nOrdem inicial dos jobs pela heurística:")
    print(novo_job_order)
    print(vec_infos)
    
    novo_job_order, vec_infos = busca_local_swap(novo_job_order,processing_times_ajustados, params)
    print("\nOrdem final dos jobs pela busca local:")
    print(novo_job_order)

    novo_completion_times = calcular_completion_times(processing_times_ajustados.loc[novo_job_order], novo_job_order)
    
    print("\nTempo de término do job (linha i) na maquina (coluna m):")
    print(pd.DataFrame(novo_completion_times))
    
    makespan = vec_infos[1]
    print(f"\nMakespan para a ordem calculada: {makespan}")
    fluxo = vec_infos[2]
    print(f"\nFluxo Total para a ordem calculada: {fluxo}")
    processing_times_de_espera = vec_infos[3]
    print(f"\nTempo total de esperas: {processing_times_de_espera}")
    quantidade_de_esperas = vec_infos[4]
    print(f"\nQTDE de esperas: {quantidade_de_esperas}")
    esperas_simultaneas = vec_infos[5]
    print(f"\nQTDE de esperas simultaneas: {esperas_simultaneas}")

    # Gerar o gráfico de Gantt
    plot_gantt_chart_J(processing_times_ajustados, novo_job_order, novo_completion_times)
    plot_gantt_chart_M(processing_times_ajustados, novo_job_order, novo_completion_times)




    #####################################################################################
    # PROCURANDO SOLUÇÕES ALTERNATIVAS MINIMIZANDO OUTRO OBJETIVO MANTENDO O FLUXO TOTAL#
    #####################################################################################

    #pesos para makespan,fluxo_total,tempo_esperas,qtde_esperas,esperas simultaneas
    #melhor solucao para o fluxo total
    params = 0,1,0,0,0

    job_order, vec_infos = neh(processing_times, params)
    print("\nOrdem inicial dos jobs pela heurística:")
    print(job_order)
    print(vec_infos)
    
    job_order, vec_infos = busca_local_swap(job_order,processing_times, params)
    print("\nOrdem final dos jobs pela busca local:")
    print(job_order)

    completion_times = calcular_completion_times(processing_times.loc[job_order], job_order)
    
    print("\nTempo de término do job (linha i) na maquina (coluna m):")
    print(pd.DataFrame(completion_times))
    
    makespan = vec_infos[1]
    print(f"\nMakespan para a ordem calculada: {makespan}")
    fluxo = vec_infos[2]
    print(f"\nFluxo Total para a ordem calculada: {fluxo}")
    processing_times_de_espera = vec_infos[3]
    print(f"\nTempo total de esperas: {processing_times_de_espera}")
    quantidade_de_esperas = vec_infos[4]
    print(f"\nQTDE de esperas: {quantidade_de_esperas}")
    esperas_simultaneas = vec_infos[5]
    print(f"\nQTDE de esperas simultaneas: {esperas_simultaneas}")

    #pesos para makespan,fluxo_total,tempo_esperas,qtde_esperas,esperas simultaneas
    # solucao alternativa minimizando soma dos tempos de esperas mantendo o fluxo total
    params = 0,0,1,0,0

    job_order, vec_infos = busca_local_swap_sol_alternativas(job_order,processing_times, params)
    print("\nOrdem final dos jobs pela busca local:")
    print(job_order)

    makespan = vec_infos[1]
    print(f"\nMakespan para a ordem calculada: {makespan}")
    fluxo = vec_infos[2]
    print(f"\nFluxo Total para a ordem calculada: {fluxo}")
    processing_times_de_espera = vec_infos[3]
    print(f"\nTempo total de esperas: {processing_times_de_espera}")
    quantidade_de_esperas = vec_infos[4]
    print(f"\nQTDE de esperas: {quantidade_de_esperas}")
    esperas_simultaneas = vec_infos[5]
    print(f"\nQTDE de esperas simultaneas: {esperas_simultaneas}")


    # Gerar o gráfico de Gantt
    plot_gantt_chart_J(processing_times, job_order, completion_times)
    plot_gantt_chart_M(processing_times, job_order, completion_times)


# In[14]:


file_path = 'TempoProducaoTeste.xlsx'  # Substitua com o caminho correto do arquivo Excel
main(file_path)