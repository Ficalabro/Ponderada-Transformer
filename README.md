# Transformer PT→EN

## Objetivo
Treinar um **Transformer** (Keras/TensorFlow) para traduzir **Português → Inglês**, inspirado no tutorial oficial, ajustando subset de dados e tamanho do modelo para **rodar rápido** e ainda permitir **analisar resultados**. O mesmo notebook foi executado **na GPU e na CPU** para comparação.

---

## Arquitetura do modelo
- **Encoder–Decoder Transformer** com:
  - Embedding + **positional encoding** senoidal
  - **Multi-Head Attention** (self e cross)
  - **Feed-Forward** + LayerNorm + Dropout
- Hiperparâmetros (alinhados ao código):
  - `NUM_LAYERS=4`, `D_MODEL=128`, `DFF=512`, `NUM_HEADS=8`, `DROPOUT=0.1`
  - `MAX_TOKENS=128`, `BATCH_SIZE=64`, `EPOCHS=8`
- **Treino:** GPU **ou** CPU (mesmo notebook), **Adam** com **scheduler** (warmup).  
- **Inferência:** **greedy** simples (sem beam search).

---

## Como rodar
1. Abra o notebook no **Google Colab**.
2. **GPU:** *Runtime → Change runtime type → GPU* e execute as células na ordem.  
   **CPU:** rode sem GPU (Runtime = None) **ou** use a versão do notebook que seleciona o device automaticamente (GPU se houver; caso contrário CPU).
3. (Opcional) Ajuste na célula de configs:
   - `TRAIN_TAKE`, `VAL_TAKE`, `EPOCHS`
   - `NUM_LAYERS`, `D_MODEL`, `DFF`, `NUM_HEADS`
   - `MAX_TOKENS`, `BATCH_SIZE`
4. Use `demo(...)` para testar traduções.

> **Comparação justa:** mantenha os **mesmos hiperparâmetros** entre GPU e CPU. (Se o tempo na CPU ficar inviável, reduza temporariamente `TRAIN_TAKE`/`EPOCHS` só para inspecionar o pipeline.)

---

## Resultados

| Setup | Épocas | Subset (train/val) | Tempo/época (mediana) | Tempo total | Val Loss (final) | Observações |
|------:|:------:|:-------------------:|----------------------:|------------:|-----------------:|-------------|
| **GPU** | 8 | 30k / 1.5k | **108 s** | **959 s** | **1.9675** | `val_masked_accuracy ≈ 0.5104`; melhor *val_loss* 1.9675 (época 8); ~230 ms/step (~469 steps/época). |
| **CPU** | 8 | 30k / 1.5k | **1296 s (≈ 21.6 min)** | **11 508 s (≈ 3h12m)** | ≈ **1.97** *(estimado)* | Estimativa **12×** a GPU; ~**2.76 s/step**; ~**469 steps/época**. |

---

## Percepções pessoais

### O que foi fácil
- Montar o pipeline com **`tf.data`**.
- Reaproveitar os **tokenizers** do tutorial.
- Usar **`MultiHeadAttention`** e camadas do Keras para compor o Transformer.

### O que foi difícil
- **Encontrar o equilíbrio entre tempo e qualidade**: pequenos ajustes em `TRAIN_TAKE`, `EPOCHS` e `D_MODEL` mudam bastante o custo e o resultado (principalmente na CPU).

### Trade-offs
- **Mais dados/épocas/modelo ⇒ melhor qualidade**, porém **maior tempo/memória**.
- `MAX_TOKENS` maior reduz truncamento, mas **custo cresce ~quadrático**.

### Gargalos
- Tokenização + E/S do **TFDS** (principalmente com subsets maiores).
- Limites de **VRAM** (GPU) e throughput (CPU) ao aumentar `MAX_TOKENS`/`BATCH_SIZE`.

---

## Minha experiência

- **Teacher Forcing:** no treinamento, alimento o decoder com o **token correto anterior** (alvo deslocado), o que **acelera a convergência** e reduz a **propagação de erro**. Na inferência, o próximo token vem da **própria predição**; por isso, alinhar `y_in`/`y_out` e aplicar **máscaras** corretamente é crucial.

- **Positional Encoding (senoidal):** adiciona **informação de posição** aos embeddings **sem novos parâmetros**, permitindo que a atenção considere **ordem**. `MAX_TOKENS` impacta custo/memória, pois a atenção cresce aproximadamente **O(n²)** com o tamanho da sequência.

---

## Fui além
- **Balanceei tempo × qualidade** de forma explícita (subset + épocas + tamanho do modelo) para obter **resultados úteis** em tempo **viável**.
- Mantive a essência do Transformer do tutorial com **“knobs”** claros para novas rodadas de análise e comparação **GPU vs CPU**.
