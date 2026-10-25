# Reconhecedor Facial

[![CI](https://github.com/Amaraldo0/Reconhecedor-Facial/actions/workflows/ci.yml/badge.svg)](https://github.com/Amaraldo0/Reconhecedor-Facial/actions/workflows/ci.yml)

Trabalho da disciplina de Inteligência Artificial (UFMS). O projeto tem duas partes:

1. **Experimentos:** comparação de modelos de reconhecimento facial (FaceNet, ArcFace e
   MobileFaceNet), de fine-tuning parcial e completo, de funções de perda (Triplet e ArcFace), de
   comparadores e da escolha do threshold.
2. **Aplicação:** página web local com webcam para cadastrar, reconhecer e excluir pessoas, que
   também identifica quem não está cadastrado.

**Principais resultados no LFW (6000 pares, 10 folds):**
- o melhor modelo foi o **ArcFace iResNet-50**, com 99,85% de acurácia e nenhum falso aceite;
- o **MobileFaceNet** chega a 99,67%, sendo ~18× mais rápido e ~13× menor;
- escolher o rosto errado em fotos com várias pessoas custava mais acurácia do que a troca de
  modelo ([detalhes](#impacto-do-pré-processamento)).

---

## Como funciona

```
foto ─► detecção (MTCNN) ─► alinhamento ─► rede neural ─► embedding (512-d, L2)
                                                               │
                         cadastro (galeria de embeddings) ─► comparação ─► score ≥ threshold?
                                                                              │
                                        "Pessoa reconhecida: X (0,83)"  ◄─ sim │ não ─►  "Pessoa desconhecida"
```

- **Detecção e qualidade:** a MTCNN encontra o rosto e os 5 pontos de referência. Na aplicação,
  fotos sem rosto, com mais de um rosto, com rosto pequeno ou desfocadas são recusadas, cada caso
  com uma mensagem própria.
- **Alinhamento:** a FaceNet usa um recorte da caixa do rosto (160×160). O ArcFace e o
  MobileFaceNet usam uma transformação que leva os 5 pontos a posições fixas (112×112).
- **Embedding:** vetor de 512 números normalizado. Fotos da mesma pessoa geram vetores próximos
  (alta similaridade cosseno).
- **Comparação:** cinco comparadores disponíveis: cosseno, euclidiana, protótipo (média por
  pessoa), k-NN e SVM.
- **Threshold:** é o que separa "reconhecida" de "desconhecida". É sempre escolhido na validação,
  nunca no teste.

## Início rápido

Requer **Python 3.11 ou 3.12** (detalhes em [Instalação](#instalação)).

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\Activate.ps1
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . --no-deps

pytest                                 # 151 testes, ~1 min, só dados sintéticos
python -m facerec app                  # aplicação em http://127.0.0.1:8000
```

`python -m facerec --help` lista todos os comandos.

## Instalação

O `facenet-pytorch` 2.6 exige torch ≤ 2.3 e numpy < 2, versões que não existem para o Python
3.13+. Por isso o projeto precisa do 3.11 ou do 3.12. Ocupa ~3 GB. A GPU NVIDIA é opcional e só
acelera o fine-tuning. Testado no Windows 11 e no Ubuntu (via CI).

**Windows (PowerShell)**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1      # se bloquear: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . --no-deps
```

**Linux / macOS**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . --no-deps
```

- **GPU NVIDIA:** troque o índice do PyTorch por `https://download.pytorch.org/whl/cu121`.
- **Windows, caminhos longos:** algumas bibliotecas têm arquivos com nomes longos, e o limite do
  Windows é de 260 caracteres. Clone numa pasta curta (ex.: `C:\projetos\reconhecedor-facial`) ou
  ative `LongPathsEnabled`. O sintoma é o erro `DLL load failed ... nome do arquivo ou a extensão
  é muito grande`.
- **Pesos dos modelos:** a FaceNet é baixada automaticamente no primeiro uso. Os modelos ArcFace e
  MobileFaceNet são baixados com `python scripts/download_arcface.py` (e `--pack buffalo_s`).

## Uso

### Aplicação web

```bash
python -m facerec app            # http://127.0.0.1:8000  (--reload para desenvolvimento)
```

A aplicação escuta só em `localhost`, então as imagens não saem da máquina. Ao abrir pela primeira
vez, mostra **"Nenhuma pessoa cadastrada."**

| Função | Como funciona | Spec |
|---|---|---|
| Cadastro | Recebe um nome e até 10 fotos (webcam ou arquivo). Recusa foto sem rosto, com várias pessoas ou de baixa qualidade. Recusa também fotos de pessoas diferentes no mesmo cadastro e quem já está cadastrado. | [002](docs/specs/002-person-enrollment.md) |
| Reconhecimento | Responde "Pessoa reconhecida: X (score)" ou "Pessoa desconhecida" | [003](docs/specs/003-face-recognition.md), [004](docs/specs/004-unknown-person.md) |
| Exclusão | Apaga a pessoa e todos os seus embeddings e fotos (`secure_delete` + `VACUUM`) | [006](docs/specs/006-person-deletion.md) |
| Interface | Página local com webcam | [005](docs/specs/005-camera-interface.md) |

O modelo, o comparador, o threshold e os limites de qualidade ficam em
[`configs/app.yaml`](configs/app.yaml). Hoje são a FaceNet, o comparador protótipo e o threshold
0,55, que é provisório até o E07 ser rodado.

Para trocar de modelo é preciso cadastrar de novo: embeddings de redes diferentes não são
comparáveis, e a aplicação avisa quando isso acontece.

### Pasta do grupo (`pessoas/`)

O dataset do grupo é uma pasta local, com uma subpasta por pessoa. Ela é ignorada pelo git e não
vai para o GitHub.

```
pessoas/
├── pessoa_a/   foto1.jpg, foto2.jpg, ...   (mín. 2 fotos; recomendado 5+ com variações)
├── pessoa_b/   ...
└── ...         (mín. 2 pessoas)
```

```bash
python -m facerec evaluate-folder                 # avalia modelos × comparadores na pasta
python -m facerec evaluate-folder --pre-aligned   # se as fotos já forem recortes do rosto
python -m facerec recognize foto_nova.jpg         # reconhece uma foto (ou uma pasta) usando pessoas/ como cadastro
python -m facerec recognize pessoas/              # cada foto do cadastro contra as OUTRAS: mostra acertos e erros
python scripts/import_people.py                   # cadastra a pasta na aplicação web
```

O `evaluate-folder` mede duas coisas: se cada pessoa é **reconhecida** e se quem está fora do
cadastro é **rejeitado**.

- **Pessoas conhecidas:** cada foto sai do cadastro e é consultada contra as outras fotos da mesma
  pessoa.
- **Pessoas desconhecidas:** cada pessoa sai inteira do cadastro e as fotos dela são consultadas.
- **Threshold:** é escolhido por validação cruzada entre pessoas. O threshold aplicado às fotos de
  uma pessoa nunca é calculado com as fotos dela.

A saída é **agregada e anônima** (nenhum nome de pasta ou de arquivo) e fica em
`experiments/results/dataset_grupo/`.

### Todos os comandos

| Comando | O que faz |
|---|---|
| `python -m facerec preprocess` | detecta e alinha as faces de um dataset (LFW, CelebA, privado) |
| `python -m facerec train` | fine-tuning da FaceNet (E03–E05) |
| `python -m facerec evaluate` | LFW 10-fold + validação privada (E01–E05, E08) |
| `python -m facerec compare-matchers` | comparadores em conjunto aberto (E06) |
| `python -m facerec threshold` | varredura de threshold + teste único (E07) |
| `python -m facerec report` | tabela de resultados e ROC comparativa |
| `python -m facerec evaluate-folder` | avaliação na pasta `pessoas/` |
| `python -m facerec recognize` | reconhece fotos usando `pessoas/` como cadastro |
| `python -m facerec app` | aplicação web |

Todos aceitam `--help`. Os comandos antigos (`python -m facerec.evaluate`, `facerec.train`, ...)
continuam funcionando.

## Experimentos

O protocolo completo de cada experimento está em [`docs/experiments.md`](docs/experiments.md).
Cada YAML em [`configs/experiments/`](configs/experiments) registra hipótese, dados, modelo,
hiperparâmetros e seed. As versões e o hardware vão para o `metrics.json` de cada resultado.

| ID | Modelo | Estratégia | Hipótese | Situação |
|---|---|---|---|---|
| E01 | FaceNet (VGGFace2) | pré-treinado, cosseno | baseline | ✅ |
| E02 | ArcFace iResNet-50 | pré-treinado, cosseno | margem angular gera embeddings mais separáveis que o E01 | ✅ confirmada |
| E03 | FaceNet | fine-tuning parcial + Triplet (batch-hard) | adapta ao domínio sem overfitting | ✅ refutada |
| E04 | FaceNet | fine-tuning parcial + ArcFace loss | converge melhor que Triplet com poucos dados | config pronta |
| E05 | FaceNet | fine-tuning completo (LR 10× menor) | ganho menor ou overfitting em relação ao parcial | config pronta |
| E06 | melhor de E01–E05 | cosseno × euclidiana × k-NN × SVM × protótipo | escolha do comparador | precisa do dataset por sessão |
| E07 | configuração final | varredura de threshold na validação + teste único | threshold ótimo (EER ou FAR-alvo) | precisa do dataset por sessão |
| E08 | MobileFaceNet | pré-treinado, cosseno | rede leve perde pouco e é bem mais rápida que o E02 | ✅ confirmada |

E04 e E05 não eram exigidos nesta entrega e usam a mesma infraestrutura do E03. Para E06 e E07, o
`evaluate-folder` já compara os comparadores e calibra o threshold na pasta do grupo.

## Resultados

Ambiente: CPU Intel com 12 threads, **sem GPU**, Windows 11, Python 3.11, torch 2.2.2+cpu,
onnxruntime 1.17.3 e facenet-pytorch 2.6.0. Os números vêm de `experiments/results/*/metrics.json`
e foram consolidados com `python -m facerec report`.

### LFW: protocolo oficial (6000 pares, 10 folds)

O threshold de cada fold é escolhido nos outros 9; a acurácia é a média ± desvio dos 10 folds.

| Exp. | Modelo | Acurácia | F1 | EER | ROC-AUC | FAR | FRR | ms/face (CPU) | Parâmetros |
|---|---|---|---|---|---|---|---|---|---|
| E01 | FaceNet (VGGFace2) | 99,10% ± 0,43 | 99,10% | 0,93% | 0,9991 | 0,97% | 0,83% | 61,7 | 23,5 M |
| E02 | ArcFace iResNet-50 (WebFace600K) | **99,85% ± 0,20** | **99,85%** | **0,30%** | **0,9995** | **0,00%** | **0,30%** | 90,3 | 43,6 M |
| E08 | MobileFaceNet (WebFace600K) | 99,67% ± 0,20 | 99,67% | 0,32% | 0,9993 | 0,27% | 0,40% | **5,0** | **3,4 M** |
| E03 | FaceNet + fine-tuning: melhor época (= pré-treinado) | 99,10% ± 0,43 | 99,10% | 0,93% | 0,9991 | 0,97% | 0,83% | 61,7¹ | 23,5 M |
| E03 | FaceNet + fine-tuning: última época (4) | 98,40% ± 0,54 | 98,39% | 1,77% | 0,9982 | 1,23% | 1,97% | 61,7¹ | 23,5 M |

¹ O E03 foi avaliado na GPU. Como a arquitetura é a mesma do E01, o custo em CPU é o do E01.

Matrizes de confusão (soma dos 10 folds; 3000 pares de cada tipo):

| Modelo | Mesma pessoa aceita | Mesma pessoa rejeitada | Diferentes aceitas | Diferentes rejeitadas |
|---|---|---|---|---|
| FaceNet | 2975 | 25 | 29 | 2971 |
| ArcFace iResNet-50 | 2991 | 9 | **0** | 3000 |
| MobileFaceNet | 2988 | 12 | 8 | 2992 |

Gráficos: [ROC comparativa](experiments/results/roc_comparison_lfw.png). Em
`experiments/results/E0x/` ficam a ROC, o histograma de similaridades e a matriz de confusão de
cada experimento.

### Impacto do pré-processamento

1812 das 13 233 imagens do LFW têm mais de um rosto. A primeira versão do pré-processamento usava
o **maior** rosto. No LFW, porém, a pessoa rotulada é a do **centro** da imagem, e o maior rosto às
vezes é de outra pessoa. Assim, vários pares de "mesma pessoa" comparavam, na prática, duas
pessoas diferentes.

| Modelo | Acurácia (maior rosto → rosto central) | EER | Pares "mesma pessoa" rejeitados |
|---|---|---|---|
| FaceNet | 95,90% → 99,10% | 6,10% → 0,93% | 219 → 25 |
| ArcFace iResNet-50 | 96,42% → 99,85% | 6,02% → 0,30% | 215 → 9 |
| MobileFaceNet | 96,28% → 99,67% | 6,30% → 0,32% | 216 → 12 |

O sinal do problema foi que três modelos muito diferentes rejeitavam quase os **mesmos** ~216
pares e quase nunca aceitavam impostores. O erro estava nos dados de entrada, não nos modelos. Os
resultados de antes da correção estão em
[`experiments/results/ablation_face_selection/`](experiments/results/ablation_face_selection).

### Fine-tuning (E03)

- **Dados:** treino em 695 identidades do CelebA (17 129 imagens) e validação em 149 identidades
  **diferentes** (3588 imagens). Foram excluídas 7 identidades suspeitas de também estarem no LFW,
  e não houve fotos quase duplicadas entre os conjuntos.
- **Treino:** só os últimos blocos (`repeat_3`, `block8`, `last_linear`, `last_bn`, ou seja,
  10,5 M dos 23,5 M parâmetros). Levou 177 s numa RTX 4050 com precisão mista.

| Época | Loss treino | Loss validação | EER validação |
|---|---|---|---|
| 0 (pré-treinado) | — | 0,059 | **3,10%** |
| 1 | 0,080 | 0,069 | 3,63% |
| 2 | 0,073 | 0,070 | 3,40% |
| 3 | 0,067 | 0,072 | 3,60% |
| 4 | 0,060 | 0,075 | 3,80% |

A loss de treino cai e a de validação sobe desde a primeira época: é **overfitting**. O early
stopping manteve o modelo pré-treinado como o melhor. Curvas em
[`loss_curves.png`](experiments/results/E03/loss_curves.png).

### Análise

- **Melhor modelo:** o ArcFace iResNet-50 (E02) teve a maior acurácia, o menor EER e nenhum falso
  aceite em 3000 pares de pessoas diferentes. A margem angular e um pré-treino com mais identidades
  geram embeddings mais separáveis: o EER da FaceNet é ~3× maior.
- **Custo × desempenho:** com a mesma loss e os mesmos dados de pré-treino, o MobileFaceNet perde
  só 0,18 ponto de acurácia. Em troca, é ~18× mais rápido (5,0 contra 90,3 ms/face) e ~13× menor.
  A diferença de acurácia está dentro do desvio entre folds (±0,20). É a melhor opção para tempo
  real num notebook sem GPU.
- **Pré-processamento importa tanto quanto o modelo:** escolher o rosto errado custava 3,2–3,4
  pontos de acurácia e multiplicava o EER por 7 a 20, mais do que a diferença entre os modelos.
- **Erros restantes:** com o rosto certo, os poucos erros são quase todos pares da mesma pessoa
  rejeitados, normalmente com grande diferença de pose, idade ou iluminação.
- **Fine-tuning (E03) piorou a rede:** a FaceNet já foi pré-treinada no VGGFace2 (~3,3 M imagens,
  ~9 mil pessoas). Com só 17 mil imagens de 695 pessoas e 10,5 M parâmetros livres, a rede passa a
  decorar as pessoas do treino. No LFW, a última época caiu de 99,10% para 98,40%. Melhorias
  possíveis, não testadas: learning rate menor, menos blocos treináveis, mais identidades ou
  ArcFace loss (E04).
- **Threshold depende do modelo:** a média dos thresholds calibrados no LFW foi 0,37 na FaceNet,
  0,25 no ArcFace e 0,21 no MobileFaceNet. Por isso cada modelo precisa de um threshold próprio.
- **Pasta do grupo:** o teste preliminar, com 3 pessoas × 3 fotos, foi pequeno demais para
  diferenciar os modelos (todos tiveram AUC 1,0). Os resultados serão publicados em
  `experiments/results/dataset_grupo/` quando a pasta for ampliada.

### Limitações

- O dataset privado é pequeno, então as métricas nele variam muito.
- O pré-treino (VGGFace2, WebFace600K) pode conter pessoas do LFW, e isso não é controlável. Por
  isso o LFW serve como referência externa, não como critério de escolha.
- Não há anti-spoofing: uma foto da pessoa mostrada numa tela pode ser reconhecida.
- O ArcFace e o MobileFaceNet são usados congelados (ONNX); só a FaceNet passa por fine-tuning.
- Os datasets públicos são de celebridades e podem ter viés demográfico.

## Reproduzir os experimentos

```bash
# 1. dados e pesos
python scripts/download_lfw.py
python scripts/download_arcface.py
python scripts/download_arcface.py --pack buffalo_s
#    CelebA: download manual (scripts/download_celeba.md) · privado: data/README.md
python -m facerec preprocess --dataset lfw     --model facenet
python -m facerec preprocess --dataset lfw     --model arcface
python -m facerec preprocess --dataset celeba  --model facenet
python -m facerec preprocess --dataset private --model facenet
python -m facerec preprocess --dataset private --model arcface
python scripts/check_identity_overlap.py --model facenet
python scripts/check_duplicates.py --dataset celeba  --config configs/experiments/E03_facenet_partial_triplet.yaml --exclude-out
python scripts/check_duplicates.py --dataset private --config configs/experiments/E06_matchers_open_set.yaml

# 2. modelos pré-treinados
python -m facerec evaluate --config configs/experiments/E01_facenet_baseline.yaml
python -m facerec evaluate --config configs/experiments/E02_arcface_pretrained.yaml
python -m facerec evaluate --config configs/experiments/E08_mobilefacenet_pretrained.yaml

# 3. fine-tuning (treino + avaliação)
python -m facerec train    --config configs/experiments/E03_facenet_partial_triplet.yaml
python -m facerec evaluate --config configs/experiments/E03_facenet_partial_triplet.yaml
python -m facerec train    --config configs/experiments/E04_facenet_partial_arcface.yaml
python -m facerec evaluate --config configs/experiments/E04_facenet_partial_arcface.yaml
python -m facerec train    --config configs/experiments/E05_facenet_full_finetune.yaml
python -m facerec evaluate --config configs/experiments/E05_facenet_full_finetune.yaml

# 4. com o modelo vencedor (ADR-001) definido em E06/E07:
python -m facerec compare-matchers --config configs/experiments/E06_matchers_open_set.yaml
python -m facerec threshold        --config configs/experiments/E07_threshold_sweep.yaml

# 5. tabela e ROC comparativa
python -m facerec report
```

Para um experimento real e curto, o E01 só precisa do LFW (~180 MB, download automático): rode
`download_lfw.py`, o `preprocess --dataset lfw --model facenet` e o `evaluate` do E01. Os
experimentos que dependem do dataset privado pulam essa parte com um aviso quando ele não existe.

## Datasets e modelos

| Dataset | Fonte | Uso | Termos |
|---|---|---|---|
| **LFW** | <http://vis-www.cs.umass.edu/lfw/> | benchmark de verificação | pesquisa; não redistribuído aqui (`scripts/download_lfw.py`) |
| **CelebA** | <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html> | fine-tuning (identidades com ≥ 15 fotos) | só pesquisa não comercial; **redistribuição proibida**; download manual ([instruções](scripts/download_celeba.md)) |
| **Privado / grupo** | coletado pelo grupo | domínio da aplicação; E06/E07; validação | consentimento LGPD, só adultos; local ([`data/README.md`](data/README.md)) |

| Modelo | Fonte | Termos |
|---|---|---|
| FaceNet: InceptionResnetV1 (VGGFace2) | `facenet-pytorch` (timesler) | baixado automaticamente |
| ArcFace iResNet-50 `w600k_r50` (buffalo_l) | `insightface` (deepinsight) | **apenas uso não comercial / acadêmico** |
| MobileFaceNet `w600k_mbf` (buffalo_s) | `insightface` (deepinsight) | **apenas uso não comercial / acadêmico** |

O CelebA não publica nomes. Por isso a sobreposição de pessoas entre CelebA e LFW é verificada
comparando embeddings (`scripts/check_identity_overlap.py`), e as identidades suspeitas são
excluídas do treino ([ADR-004](docs/decisions/ADR-004-data-split-and-leakage.md)).

## Privacidade (LGPD)

Dados biométricos são dados pessoais sensíveis. Este repositório **não contém** fotos,
embeddings, bancos de dados, modelos treinados com dados privados nem termos de consentimento.

- **Arquivos:** o `.gitignore` e um hook de pre-commit
  ([`check_private_files.py`](scripts/check_private_files.py)) bloqueiam fotos, bancos e pesos.
- **Logs:** só registram IDs opacos (UUID), nunca nomes, imagens ou embeddings.
- **Exclusão:** apagar uma pessoa remove tudo dela, e isso é verificado por teste.
- **Resultados:** os resultados com dados privados são salvos só de forma agregada.
- **Fim da disciplina:** `python scripts/purge_private_data.py` simula a eliminação de todos os
  dados locais, e `--yes` apaga de fato.

Detalhes em [`docs/privacy.md`](docs/privacy.md).

## Estrutura do projeto

```
src/facerec/
├── core/         config, exceções de domínio, logs sem dados pessoais
├── vision/       foto → rosto alinhado (MTCNN, qualidade, alinhamento, FaceLoader)
├── models/       redes de embedding (FaceNet, ArcFace, MobileFaceNet)
├── recognition/  embedding → identidade: comparadores, métricas, threshold
├── data/         leitura dos datasets, splits sem vazamento, augmentation, pré-processamento
├── training/     fine-tuning e funções de perda (Triplet batch-hard, ArcFace)
├── evaluation/   protocolos (LFW, pasta do grupo, E06/E07), gráficos e relatório
├── app/          casos de uso, banco SQLite, API FastAPI + página com webcam
└── cli/          comandos `python -m facerec <comando>`
configs/          app.yaml e um YAML por experimento
experiments/      resultados agregados (metrics.json, gráficos)
scripts/          downloads, checagens de dados e de privacidade
docs/             specs, decisões (ADRs), protocolo dos experimentos, privacidade
tests/            testes com dados sintéticos
```

As dependências seguem uma direção só: `cli` → `evaluation` / `training` / `app` →
`recognition` / `vision` / `models` / `data` → `core`.

O projeto seguiu **Spec-Driven Development**. As specs estão em [`docs/specs/`](docs/specs) e as
decisões de arquitetura em [`docs/decisions/`](docs/decisions): escolha do modelo, armazenamento,
threshold e prevenção de vazamento. A ligação entre o enunciado e o código está em
[`docs/parte1_checklist.md`](docs/parte1_checklist.md), e o roteiro do vídeo em
[`docs/video_parte1.md`](docs/video_parte1.md).

## Desenvolvimento

```bash
pytest                                            # 151 testes, só dados sintéticos
ruff check . && ruff format --check .             # lint e formatação
python scripts/check_private_files.py --history   # nenhum dado privado no histórico do git
pre-commit install                                # ativa o bloqueio de dados privados nos commits
```

Os testes não usam fotos reais nem baixam nada: usam imagens geradas, embeddings aleatórios e um
detector falso. O teste de ponta a ponta
([`tests/test_pipeline_smoke.py`](tests/test_pipeline_smoke.py)) roda os YAMLs reais dos
experimentos E01, E03, E04, E06 e E07 com essas imagens. O CI roda tudo a cada push, no Linux e no
Windows, com Python 3.11 e 3.12 ([`ci.yml`](.github/workflows/ci.yml)).
