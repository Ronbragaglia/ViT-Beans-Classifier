# ViT-Beans-Classifier (ou o nome que você escolher)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[SEU-USUARIO-GITHUB]/[NOME-DO-REPO]/blob/main/NomeDoSeuNotebook.ipynb) <!-- Substitua pelo link direto para o notebook no GitHub -->
[![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Repositório com um exemplo completo de fine-tuning do modelo Vision Transformer (ViT) para classificação de imagens de folhas de feijão, identificando diferentes doenças ou folhas saudáveis. Este projeto utiliza PyTorch e o ecossistema Hugging Face (Transformers, Datasets, Evaluate).

## 🎯 Objetivo

O objetivo deste projeto é demonstrar um pipeline robusto para treinar um modelo de visão computacional de última geração (ViT) em um dataset específico ('beans' da Hugging Face), alcançando alta acurácia na classificação das imagens.

## ✨ Principais Características

*   **Modelo:** Fine-tuning do Vision Transformer pré-treinado (`google/vit-base-patch16-224-in21k`).
*   **Framework:** PyTorch.
*   **Bibliotecas:** Hugging Face `transformers`, `datasets`, `evaluate`.
*   **Pré-processamento:** Uso de `torchvision.transforms` para data augmentation e normalização.
*   **Loop de Treinamento Customizado:** Controle explícito sobre o treinamento, avaliação, otimização e atualização do learning rate.
*   **Técnicas Avançadas:**
    *   Otimizador AdamW com Weight Decay.
    *   Learning Rate Scheduler (Cosine Annealing com Warmup).
    *   Label Smoothing na função de perda CrossEntropy.
    *   Gradient Clipping para estabilização.
    *   Mixed Precision (AMP) via `torch.cuda.amp` (ativado se GPU disponível).
*   **Monitoramento:** Integração com [Weights & Biases (W&B)](https://wandb.ai/) para logar métricas, hiperparâmetros, matriz de confusão e salvar o melhor modelo como artefato.
*   **Avaliação:** Cálculo de acurácia e geração de matriz de confusão no conjunto de teste.
*   **Reprodutibilidade:** Uso de seed para inicialização de pesos e divisões de dados.
*   **Código:** Implementado em um notebook Google Colab (`.ipynb`) para fácil execução e experimentação.

## 🌱 Dataset

Utilizamos o dataset [`beans`](https://huggingface.co/datasets/beans) disponível na Hugging Face Datasets Hub. Ele contém imagens de folhas de feijão classificadas em três categorias:
*   `angular_leaf_spot` (Mancha-angular)
*   `bean_rust` (Ferrugem)
*   `healthy` (Saudável)

## 📈 Resultados

Após o fine-tuning por 4 épocas (conforme o log fornecido), o modelo alcançou:

*   **Melhor Acurácia de Validação:** 99.25%
*   **Acurácia Final no Conjunto de Teste:** 96.88%

A matriz de confusão gerada (e logada no W&B) mostra um excelente desempenho na distinção entre as classes no conjunto de teste.

*(Opcional: Você pode incluir a imagem da matriz de confusão aqui)*
```markdown
![Matriz de Confusão](caminho/para/sua/imagem/confusion_matrix_test.png) 
<!-- Se você fizer upload da imagem para o repo, ajuste o caminho -->

🛠️ Tecnologias Utilizadas
Python 3.11+

PyTorch

Hugging Face Transformers

Hugging Face Datasets

Hugging Face Evaluate

Torchvision

Weights & Biases (wandb)

Scikit-learn (para matriz de confusão)

NumPy

Matplotlib


git clone https://github.com/[SEU-USUARIO-GITHUB]/[NOME-DO-REPO].git
cd [NOME-DO-REPO]

python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

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
# Adicione outras dependências se necessário

pip install -r requirements.txt

▶️ Como Usar
A maneira mais fácil de executar este projeto é usando o Google Colab:

Abra no Colab: Clique no botão "Open In Colab" no topo deste README ou acesse o link diretamente: https://colab.research.google.com/github/[SEU-USUARIO-GITHUB]/[NOME-DO-REPO]/blob/main/NomeDoSeuNotebook.ipynb <!-- **ATUALIZE ESTE LINK** -->

Configure o Ambiente de Execução:

Vá em Ambiente de execução -> Alterar o tipo de ambiente de execução.

Selecione GPU como acelerador de hardware (recomendado para velocidade).

Clique em Salvar.

Execute as Células: Execute as células do notebook sequencialmente, uma por uma.

Weights & Biases Login: A primeira célula pedirá para você fazer login no W&B. Siga as instruções:

Escolha a opção 2 (Use an existing W&B account).

SE for solicitado, cole sua chave de API do W&B (encontrada em wandb.ai/authorize).

Se preferir não usar o W&B, pode escolher a opção 3.

Acompanhe o Treinamento: O progresso será exibido no notebook e, se o W&B estiver ativo, no seu dashboard do W&B.

Resultados: Ao final, o modelo treinado será avaliado no conjunto de teste, a matriz de confusão será exibida/salva, e um exemplo de inferência será mostrado. O melhor modelo será salvo no diretório especificado (./vit-beans-torchvision-run/best_model_pretrained por padrão).

.
├── NomeDoSeuNotebook.ipynb     # O notebook principal com todo o código
├── requirements.txt            # Dependências Python
├── README.md                   # Este arquivo
└── (Opcional) LICENSE          # Arquivo de licença (ex: MIT)
└── (Gerado pela execução) vit-beans-torchvision-run/ # Diretório com modelo salvo, matriz, etc.

📜 Licença
Este projeto está licenciado sob a Licença MIT. Veja o arquivo LICENSE para mais detalhes.
(Sugestão: Crie um arquivo LICENSE na raiz com o texto da licença MIT).

🙏 Agradecimentos
À equipe da Hugging Face pelas excelentes bibliotecas transformers, datasets, e evaluate.

Ao Google pela pesquisa e disponibilização do modelo Vision Transformer (ViT).

Aos criadores do dataset beans.

À equipe do Weights & Biases pela ferramenta de monitoramento.
