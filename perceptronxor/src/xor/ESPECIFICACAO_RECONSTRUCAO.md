# Especificação de reconstrução: Laboratório MLP XOR

## Objetivo

Reconstruir em outra plataforma a aplicação didática atualmente implementada em `index.html`. Ela demonstra uma rede neural MLP para XOR, permite inspecionar inferência e treinamento passo a passo, executar treinamento completo, comparar referências do Julia e explorar portas lógicas.

A implementação atual é autocontida: estrutura, estilos, lógica, parâmetros e resultados de referência estão no próprio HTML. Não depende de bibliotecas, serviços ou ativos externos. A nova versão pode usar qualquer framework, mas deve preservar os comportamentos e os dados descritos neste documento.

## Experiência e navegação

- Idioma da interface: português do Brasil.
- Título: **Laboratório MLP XOR — Treinamento e Inferência**.
- Cabeçalho mostra a arquitetura atual no formato `2 → H → 1` e explica que os cálculos são reais e atualizados ao vivo.
- Navegação superior fixa, com links para sete seções e rolagem até cada uma:
  1. Arquitetura da rede
  2. Inferência passo a passo
  3. Laboratório de treinamento — uma época por dentro
  4. Treinamento completo
  5. Fórmulas e correspondência com o Julia
  6. Resultados do seu `xor.jl`
  7. Gráfico interativo das portas lógicas
- A página é uma única experiência vertical. Em telas estreitas, colunas passam a uma coluna e tabelas largas podem rolar horizontalmente.
- Direção visual atual: interface didática clara, fundo azul-acinzentado muito pálido, cabeçalho e títulos de seção em azul, cartões brancos, cores semânticas verde/vermelho/âmbar/roxo e tipografia de sistema. Preservar contraste, hierarquia e estados visuais, sem exigir reprodução literal do CSS.
- Usar notação matemática e valores numéricos legíveis; tabelas devem alinhar números e manter cabeçalhos reconhecíveis.

## Modelo e dados

### Problema XOR

Entradas, na ordem usada para treinamento e tabelas:

| x₁ | x₂ | y esperado |
| ---: | ---: | ---: |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

### Arquitetura

- MLP totalmente conectada `2 → H → 1`; `H` é configurável (2, 3, 4, 5, 6 ou 8; padrão 4).
- Cada neurônio oculto e o neurônio de saída usam sigmoide: `σ(z) = 1 / (1 + e⁻ᶻ)`.
- Derivada calculada a partir da ativação: `σ′ = a(1 − a)`.
- Formas dos parâmetros: `W₁` é `2 × H`, `b₁` tem `H` valores, `W₂` tem `H` valores e `b₂` é escalar. Quantidade total: `4H + 1` parâmetros.
- Propagação direta: `z₁ = xW₁ + b₁`; `a₁ = σ(z₁)`; `z₂ = a₁W₂ + b₂`; `ŷ = σ(z₂)`.
- Classificação: `ŷ ≥ 0,5` significa VERDADEIRO; abaixo de `0,5`, FALSO.
- Precisão compara a classe prevista com os quatro alvos XOR. Perda exibida: `MSE = (1/4) Σ(yᵢ − ŷᵢ)²`.

### Parâmetros de referência embutidos

Implementar duas fontes distintas para a seção de inferência:

1. **Julia treinado**: parâmetros de 4 neurônios ocultos abaixo, correspondentes ao modelo binário `xor_modelo.jls` usado pela página. Manter precisão integral nos cálculos.
2. **Laboratório**: parâmetros vivos do estado atual do treinamento interativo.

```text
W1 = [
  [ 1.6814560737091662, -5.411913583606491, -4.252425809330088, -5.948768885089716 ],
  [ 1.926922219036426,  -5.489400312597228,  6.380032333769705,  2.489341759318851 ]
]
b1 = [1.1994079632866756, 1.4489043341572911, 1.584539181089928, -0.15292923115257884]
W2 = [1.914953378379854, -6.628302027589323, -8.091861015935287, 8.329250313121682]
b2 = 2.4080086154575935
```

Inicialização didática de Julia, usada para reproduzir a primeira época com `H = 4`:

```text
W1 = [
  [ 0.076642, -0.944916, -0.768287, -0.989216 ],
  [ 0.994709, -0.859274,  0.811087, -0.261340 ]
]
b1 = [0, 0, 0, 0]
W2 = [0.279172, -0.006885, -0.394897, 0.130522]
b2 = 0
```

Valores numéricos mostrados na interface são arredondados; não arredondar os valores internos do modelo.

## Seção 1: Arquitetura da rede

- Desenhar rede com duas entradas, `H` neurônios ocultos e uma saída; mostrar ligações, pesos quando disponíveis, vieses e a sigmoide.
- Diferenciar visualmente conexões positivas e negativas; a espessura pode representar magnitude.
- Mostrar `H`, arquitetura e quantidade de parâmetros. Explicar que as entradas são booleanas, a camada oculta permite representar XOR e a saída é uma probabilidade.
- Mostrar a curva sigmoide e sua fórmula, inclusive a interpretação da derivada.
- Atualizar a arquitetura mostrada quando `H` mudar no laboratório.

## Seção 2: Inferência passo a passo

### Controles e estado

- Seletor de modelo: **Julia treinado** (padrão) ou **Laboratório**.
- Quatro seletores para entrada: `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)`. A entrada selecionada inicia em `(0,1)`.
- A seleção da entrada atualiza imediatamente todas as visualizações da seção.
- Ao selecionar o laboratório na época zero, informar que a rede ainda não foi treinada.
- Mostrar origem, arquitetura, época, perda e precisão do modelo escolhido.

### Passos explicativos

Um stepper permite anterior, próximo, seleção direta por número, reprodução automática e opção “Mostrar todos os passos”. Reprodução avança aproximadamente a cada 1,7 s. Alterar modelo ou entrada reinicia o stepper. Incluir:

1. Entrada, alvo XOR e rede destacada.
2. Um passo por neurônio oculto: produtos de entrada e peso, viés, soma `z`, ativação e ponto correspondente na curva sigmoide.
3. Vetor completo de ativações ocultas.
4. Soma ponderada da saída, contribuições de `W₂`, viés e ativação.
5. Decisão por limiar `0,5`, classe esperada e indicação visual de acerto/erro.

### Resumos sincronizados

- Tabela dos valores de cada neurônio: pesos de entrada, viés, `z` e `σ(z)`, além de `z₂` e `ŷ`.
- Tabela-verdade do modelo ativo, clicável para escolher a linha/entrada.
- Mapa quadrado de decisão no domínio `[0,1] × [0,1]`: verde para probabilidade acima de `0,5`, vermelho abaixo; marcar as quatro entradas booleanas e suas probabilidades.
- Bloco de texto com propagação equivalente à saída de terminal Julia, preservando precisão de cálculo e apresentando valores com seis casas para `z`/ativações e três casas para pesos quando apropriado.

## Seção 3: Laboratório de treinamento, época a época

### Configuração

- Taxa de aprendizagem `η`: padrão `0,5`.
- Neurônios ocultos `H`: opções `2, 3, 4, 5, 6, 8`; padrão `4`.
- Inicialização: pesos de Julia (padrão) ou Glorot/Xavier com gerador JavaScript.
- Semente do gerador JavaScript: padrão `1234`.
- Os pesos iniciais de Julia só se aplicam a `H = 4`; para outro tamanho, usar inicialização Glorot com a semente informada e explicar essa substituição.
- Alterar `H`, tipo de inicialização ou semente reinicia modelo e histórico. Alterar `η` afeta as próximas épocas, sem reinicializar.

### Ações e explicação de uma época

- “Voltar à época 0” reinicializa pesos e histórico.
- “+100 épocas” e “+1000 épocas” atualizam o modelo diretamente.
- O stepper de uma época mostra os quatro exemplos no mesmo lote e os passos:
  1. Propagação direta dos quatro exemplos.
  2. MSE calculado antes de atualizar os parâmetros.
  3. Delta da saída por exemplo: `δ₂ = (ŷ − y) ⊙ ŷ(1 − ŷ)`.
  4. Delta oculto: `δ₁ = (δ₂ W₂ᵀ) ⊙ a₁(1 − a₁)`; utilizar `W₂` anterior à atualização.
  5. Gradientes médios do lote (`n = 4`).
  6. Atualização dos pesos e vieses, mostrando valor anterior, gradiente, `η × gradiente` e valor posterior.
- No passo final, “Aplicar” efetiva exatamente uma época calculada sobre o estado atual, registra a perda anterior à atualização e abre a próxima época.
- A seleção de exemplo destaca a linha correspondente, mas os gradientes continuam usando os quatro exemplos.

### Equações do lote

```text
∂W₂ = (a₁ᵀ δ₂) / n       ∂b₂ = Σδ₂ / n
∂W₁ = (Xᵀ δ₁) / n        ∂b₁ = Σδ₁ / n
parâmetro_novo = parâmetro_antigo − η × gradiente
```

Observação necessária: a perda exibida é MSE sem fator `1/2`, mas a implementação compatível com Julia usa `δ₂ = (ŷ − y)σ′`, sem multiplicar por 2. Explicar que isso equivale ao gradiente de `MSE/2` e preservar esse comportamento para paridade.

## Seção 4: Treinamento completo

### Controles e comportamento

- Épocas máximas: padrão `20000`, mínimo `1`, máximo `500000`.
- Tolerância de parada: padrão `0,0001`; interromper uma tentativa quando a perda calculada antes da atualização ficar abaixo da tolerância.
- Tentativas/reinicializações: padrão `5`, mínimo `1`, máximo `20`.
- “Treinar do zero” executa tentativas com novas inicializações. A primeira respeita a configuração atual; as seguintes usam Glorot e semente `seed + tentativa − 1`. Guardar o modelo com melhor precisão e parar na primeira tentativa com 100% de precisão.
- “Continuar o modelo atual” mantém parâmetros e histórico, executando até o número de épocas informado ou até a tolerância.
- “Parar” solicita cancelamento. Atualizar progresso e interface em blocos de até 1000 épocas para permitir interação.
- Durante execução, desabilitar controles que possam invalidar a configuração/estado; habilitar “Parar”.
- Mostrar barra e texto de progresso, log, tabela-verdade final, precisão e perda.
- O gráfico de perda usa eixo vertical logarítmico, série de perdas por época, referência opcional dos pontos Julia e tooltip/indicador de época ao passar o mouse. Redesenhar ao redimensionar.
- Exportar o modelo do laboratório como JSON baixável `xor_modelo_js.json`.

Formato do JSON exportado:

```json
{
  "descricao": "MLP XOR 2-H-1 (simulador JavaScript)",
  "arquitetura": [2, 4, 1],
  "epocas": 20000,
  "perda": 0.00066,
  "W1": [[0, 0, 0, 0], [0, 0, 0, 0]],
  "b1": [0, 0, 0, 0],
  "W2": [0, 0, 0, 0],
  "b2": 0
}
```

Os zeros são apenas ilustrativos do formato; exportar os valores reais do estado ativo.

### Paridade com Julia

- Com `H = 4`, pesos iniciais de Julia, `η = 0,5` e ordem de exemplos acima, a primeira época deve reproduzir o log didático Julia.
- O gerador aleatório Glorot da página é JavaScript (`mulberry32`), não `MersenneTwister`; não prometer igualdade de sequências aleatórias com Julia.
- Quando o modo Julia e as condições de paridade forem válidos, comparar perdas nos checkpoints registrados abaixo, informar diferença máxima e permitir abrir tabela detalhada.
- Checkpoints de perda de referência:

| Época | Perda |
| ---: | ---: |
| 1 | 0,250062 |
| 1.000 | 0,233640 |
| 2.000 | 0,131209 |
| 3.000 | 0,031048 |
| 4.000 | 0,011828 |
| 5.000 | 0,006590 |
| 6.000 | 0,004404 |
| 7.000 | 0,003252 |
| 8.000 | 0,002554 |
| 9.000 | 0,002091 |
| 10.000 | 0,001764 |
| 11.000 | 0,001521 |
| 12.000 | 0,001335 |
| 13.000 | 0,001188 |
| 14.000 | 0,001068 |
| 15.000 | 0,000970 |
| 16.000 | 0,000888 |
| 17.000 | 0,000818 |
| 18.000 | 0,000758 |
| 19.000 | 0,000706 |
| 20.000 | 0,000660 |

## Seção 5: Fórmulas e Julia

Apresentar em formato consultável:

- Propagação direta, MSE, retropropagação e atualização em lote.
- Nota sobre a ausência do fator 2 no delta usado para compatibilidade.
- Tabela de correspondência entre comandos (`train --didatico true`, treinamento completo, `infer A B`, arquivo de modelo e log) e as seções da interface.
- Nota de que arredondamento é apenas visual e que o gerador aleatório JS difere do `MersenneTwister` de Julia.

## Seção 6: Resultados estáticos do `xor.jl`

Esta seção é um retrato histórico independente do modelo interativo. Não recalcular nem substituir esses valores quando o usuário treinar o laboratório.

- Arquitetura `2 → 4 → 1`, sigmoide.
- Execução de referência: até 30.000 épocas, `η = 0,5`, tolerância por perda; convergiu na tentativa `1/10`, seed `1234`, época `14.677`.
- Precisão final `100%`; perda `0,00099992`.
- Inferência selecionada `(0,1)`: `0,968206`, VERDADEIRO.
- Tabela de probabilidades:

| Entrada | Probabilidade | Predição | Esperado |
| --- | ---: | --- | --- |
| (0,0) | 0,016098 | FALSO | FALSO |
| (0,1) | 0,968206 | VERDADEIRO | VERDADEIRO |
| (1,0) | 0,967147 | VERDADEIRO | VERDADEIRO |
| (1,1) | 0,040625 | FALSO | FALSO |

- Para `(0,1)`, mostrar valores de inferência: `h₁ z=3,061437, a=0,955274`; `h₂ z=−3,985272, a=0,018248`; `h₃ z=7,773335, a=0,999579`; `h₄ z=2,258073, a=0,905345`; saída `z=3,416183`, `y=0,968206`.
- Manter clara a diferença entre este retrato (`14.677` épocas) e o modelo binário embutido usado na inferência, associado ao log de `20.000` épocas (`perda 0,000660`). São execuções de referência distintas.

## Seção 7: Portas lógicas interativas

- Operações: XOR, AND, OR, NAND, NOR e XNOR.
- Entradas A e B são botões binários independentes; valores iniciais `A=0`, `B=1`; operação inicial XOR.
- Trocar entrada ou operação sincroniza diagrama de circuito, resultado textual, explicação, fórmula e tabela-verdade. Destacar a linha atual.
- Funções lógicas:
  - XOR: `A ≠ B`
  - AND: `A ∧ B`
  - OR: `A ∨ B`
  - NAND: `¬(A ∧ B)`
  - NOR: `¬(A ∨ B)`
  - XNOR: `A = B`
- Resultado verdadeiro usa estado visual verde; falso usa vermelho. O SVG atual exibe dois fios de entrada, bloco da porta e fio/saída.

## Acessibilidade e responsividade

- Todo controle tem rótulo associado; botões de bits refletem estado com `aria-pressed`.
- Navegação e controles devem funcionar por teclado, ter foco visível e estados desabilitados claros.
- Diagramas SVG têm nome/descrição acessível; gráficos canvas devem ter resumo textual ou alternativa equivalente com dados/tabela.
- Não depender apenas de cor para indicar acerto, classe ou estado: incluir texto, símbolos ou valores.
- Em celular, converter composições de duas colunas em coluna única, manter alvos de toque confortáveis e evitar que conteúdo ou controles transbordem a viewport.

## Critérios de aceite

1. As sete seções existem, a navegação leva à seção correta e a interface está em pt-BR.
2. A inferência para as quatro entradas usa os parâmetros completos do modelo selecionado e respeita o limiar `0,5`; trocar fonte ou entrada atualiza stepper, tabela, mapa e texto.
3. A arquitetura, os cálculos e as dimensões dos parâmetros são corretos para todos os valores permitidos de `H`.
4. Uma época aplica gradientes de lote usando `W₂` anterior à atualização; a perda registrada é anterior à atualização e a seção didática mostra as quatro amostras.
5. Reiniciar, avançar épocas, treinar do zero, continuar, parar, acompanhar log/progresso e exportar JSON funcionam sem corromper o estado.
6. Com inicialização Julia, `H=4` e `η=0,5`, a paridade esperada é apresentada e os checkpoints são comparáveis dentro do arredondamento informado.
7. Os resultados estáticos da seção Julia permanecem inalterados e explicitamente separados do modelo interativo.
8. As seis portas lógicas produzem as tabelas-verdade corretas e todas as visualizações da seção permanecem sincronizadas.
9. A experiência funciona em desktop e celular, com foco de teclado, rótulos e alternativa acessível para os gráficos.
