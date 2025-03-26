# ViT-Beans-Classifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ronbragaglia/ViT-Beans-Classifier/blob/main/ViT_Beans_Classifier.ipynb) <!-- *** IMPORTANTE: Substitua 'ViT_Beans_Classifier.ipynb' pelo nome real do seu arquivo .ipynb no GitHub *** -->
[![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Repositório com um exemplo completo de fine-tuning do modelo Vision Transformer (ViT) para classificação de imagens de folhas de feijão, identificando diferentes doenças ou folhas saudáveis. Este projeto utiliza PyTorch e o ecossistema Hugging Face (Transformers, Datasets, Evaluate).

Desenvolvido por [Ronbragaglia](https://github.com/Ronbragaglia).

## 🎯 Objetivo

O objetivo deste projeto é demonstrar um pipeline robusto para treinar um modelo de visão computacional de última geração (ViT) em um dataset específico ('beans' da Hugging Face), alcançando alta acurácia na classificação das imagens, mesmo em ambientes com recursos limitados (como CPU no Colab).

## ✨ Principais Características

*   **Modelo:** Fine-tuning do Vision Transformer pré-treinado (`google/vit-base-patch16-224-in21k`).
*   **Framework:** PyTorch.
*   **Bibliotecas:** Hugging Face `transformers`, `datasets`, `evaluate`.
*   **Pré-processamento:** Uso de `torchvision.transforms` (v2/v1) para data augmentation e normalização, garantindo estabilidade.
*   **Loop de Treinamento Customizado:** Controle explícito sobre o treinamento, avaliação, otimização e atualização do learning rate.
*   **Técnicas Avançadas:**
    *   Otimizador AdamW com Weight Decay.
    *   Learning Rate Scheduler (Cosine Annealing com Warmup).
    *   Label Smoothing na função de perda CrossEntropy.
    *   Gradient Clipping para estabilização.
    *   Mixed Precision (AMP) via `torch.cuda.amp` (configurado para ativar se GPU disponível, desativado na execução em CPU).
*   **Monitoramento:** Integração com [Weights & Biases (W&B)](https://wandb.ai/) para logar métricas, hiperparâmetros, matriz de confusão e salvar o melhor modelo como artefato.
*   **Avaliação:** Cálculo de acurácia e geração de matriz de confusão no conjunto de teste.
*   **Reprodutibilidade:** Uso de seed para inicialização de pesos e processos.
*   **Código:** Implementado em um notebook Google Colab (`.ipynb`) com tratamento de erros e comentários detalhados.

## 🌱 Dataset

Utilizamos o dataset [`beans`](https://huggingface.co/datasets/beans) disponível na Hugging Face Datasets Hub. Ele contém imagens de folhas de feijão classificadas em três categorias:
*   `angular_leaf_spot` (Mancha-angular)
*   `bean_rust` (Ferrugem)
*   `healthy` (Saudável)

## 📈 Resultados

Após o fine-tuning por 4 épocas (executado em CPU no ambiente Colab), o modelo alcançou:

*   **Melhor Acurácia de Validação:** **99.25%** (atingida na Época 3)
*   **Acurácia Final no Conjunto de Teste:** **96.88%**

A matriz de confusão gerada (e logada no W&B) confirma o excelente desempenho do modelo, com poucos erros de classificação no conjunto de teste.

*(Nota: Os avisos `FutureWarning: torch.cuda.amp.autocast(...) is deprecated` foram observados, mas como a execução foi em CPU, o `autocast` estava desabilitado e o aviso não impactou o resultado. Para execuções futuras em GPU, a sintaxe pode ser atualizada para `torch.amp.autocast('cuda', ...)`)*

## 🛠️ Tecnologias Utilizadas

*   Python 3.11+
*   PyTorch
*   Hugging Face Transformers
*   Hugging Face Datasets
*   Hugging Face Evaluate
*   Torchvision
*   Weights & Biases (wandb)
*   Scikit-learn
*   NumPy
*   Matplotlib
*   Tqdm
*   Google Colab

## Resultados obtidos:
![image](https://github.com/user-attachments/assets/0d999558-f3a2-48fc-93f8-01aeff6d04d5)

![image](https://github.com/user-attachments/assets/182046a3-0131-4d7f-a7fe-7d8c4a35a13d)

![image](https://github.com/user-attachments/assets/a5349ecd-bd42-4d4e-b850-ac364a008a8a)

![image](https://github.com/user-attachments/assets/4a02af4b-d985-4e11-8303-7e4778efc9e6)

![image](https://github.com/user-attachments/assets/9dec4d45-6ee0-45b5-8cfb-41b7126fc70a)



## ⚙️ Configuração e Instalação

1.  **Clone o Repositório:**
    ```bash
    git clone https://github.com/Ronbragaglia/ViT-Beans-Classifier.git
    cd ViT-Beans-Classifier
    ```

2.  **Crie um Ambiente Virtual (Recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

3.  **Instale as Dependências:**
    Crie um arquivo `requirements.txt` com o seguinte conteúdo:
    ```txt
    transformers
    datasets
    evaluate
    accelerate
    torch
    torchvision
    wandb
    scikit-learn
    tqdm
    matplotlib
    numpy
    ```
    E então instale:
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Como Usar

A maneira mais fácil de executar este projeto é usando o Google Colab:

1.  **Abra no Colab:** Clique no botão "Open In Colab" no topo deste README ou acesse o link diretamente: [`https://colab.research.google.com/github/Ronbragaglia/ViT-Beans-Classifier/blob/main/ViT_Beans_Classifier.ipynb`](https://colab.research.google.com/github/Ronbragaglia/ViT-Beans-Classifier/blob/main/ViT_Beans_Classifier.ipynb) <!-- *** ATUALIZE ESTE LINK com o nome correto do seu notebook .ipynb *** -->
2.  **Configure o Ambiente de Execução:**
    *   Vá em `Ambiente de execution` -> `Alterar o tipo de ambiente de execution`.
    *   Selecione `GPU` como acelerador de hardware (altamente recomendado para velocidade) ou `None` para CPU.
    *   Clique em `Salvar`.
3.  **Execute as Células:** Execute as células do notebook sequencialmente (`Shift + Enter` ou botão de Play).
4.  **Weights & Biases Login:** A Célula 1 tentará fazer login no W&B. Siga as instruções no output:
    *   Geralmente, digite `2` para usar uma conta existente.
    *   **SE** solicitado, cole sua chave de API do W&B (encontrada em [wandb.ai/authorize](https://wandb.ai/authorize)).
    *   Se preferir não usar o W&B, pode escolher a opção `3` ou deixar o login falhar (o script continuará desabilitando o W&B).
5.  **Acompanhe o Treinamento:** O progresso será exibido no notebook (barras de progresso, logs de época). Se o W&B estiver ativo, acesse o link da run fornecido no output para ver gráficos e logs detalhados.
6.  **Resultados:** Ao final, o modelo treinado será avaliado no conjunto de teste, a matriz de confusão será exibida/salva, e um exemplo de inferência será mostrado. O melhor modelo será salvo no diretório `./vit-beans-torchvision-run/best_model_pretrained`.

## 📂 Estrutura do Projeto
.
├── ViT_Beans_Classifier.ipynb # <-- CONFIRME/AJUSTE O NOME DO ARQUIVO .ipynb
├── requirements.txt # Dependências Python
├── README.md # Este arquivo
└── (Opcional) LICENSE # Arquivo de licença (ex: MIT)
└── (Gerado pela execução) vit-beans-torchvision-run/ # Diretório com modelo salvo, matriz, etc.


## 📜 Licença

Este projeto está licenciado sob a Licença MIT.

## 🙏 Agradecimentos

*   À equipe da [Hugging Face](https://huggingface.co/) pelas bibliotecas `transformers`, `datasets`, e `evaluate`.
*   À equipe do [PyTorch](https://pytorch.org/) e [Torchvision](https://pytorch.org/vision/stable/index.html).
*   Ao Google pela pesquisa e disponibilização do modelo Vision Transformer (ViT).
*   Aos criadores do dataset `beans`.
*   À equipe do [Weights & Biases](https://wandb.ai/) pela ferramenta de monitoramento.

*   
