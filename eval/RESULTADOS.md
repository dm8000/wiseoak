# WiseOak — resultados

5 células · bench `vf-dev`

## A/A — o piso de ruído

| modelo | setup | acerto 1 | acerto 2 | diferença | discordantes | p |
|---|---|---:|---:|---:|---:|---:|
| medgemma-clinical | v2|rac=nenhum|ix=h512|k=10/5 | 52.5% | 52.5% | +0.0% | 0/39 | 1.000 |

**Leia assim:** qualquer ganho de A/B menor que a diferença acima está dentro do ruído, por mais bonito que seja o número.

## medgemma-clinical

| setup | n | acerto | IC 95% | truncou | fidelidade | pág. ok | p50 s |
|---|---:|---:|---|---:|---:|---:|---:|
| v0|rac=nenhum|ix=h512|k=10/5 | 40 | 60.0% | 44.6%–73.7% | 0.0% | 0.0% | 0.0% | 0.8 |
| v0|rac=nenhum|ix=h512|k=20/5 | 5 | 60.0% | 23.1%–88.2% | 0.0% | 0.0% | 0.0% | 0.8 |
| v1|rac=nenhum|ix=h512|k=10/5 | 40 | 57.5% | 42.2%–71.5% | 0.0% | 61.5% | 61.5% | 4.7 |
| v2|rac=nenhum|ix=h512|k=10/5|rep=1 | 40 | 52.5% | 37.5%–67.1% | 0.0% | 55.2% | 51.7% | 20.5 |
| v2|rac=nenhum|ix=h512|k=10/5|rep=2 | 40 | 52.5% | 37.5%–67.1% | 0.0% | 48.3% | 45.0% | 20.1 |

Ganho pareado sobre `v0|rac=nenhum|ix=h512|k=10/5` (McNemar + Holm):

| setup | ganho | ±dp | p bruto | p corrigido | signif. |
|---|---:|---:|---:|---:|:--:|
| v0|rac=nenhum|ix=h512|k=20/5 | +0.0% | 0.0% | 1.0000 | 1.0000 | não |
| v1|rac=nenhum|ix=h512|k=10/5 | -2.6% | 9.2% | 1.0000 | 1.0000 | não |
| v2|rac=nenhum|ix=h512|k=10/5|rep=1 | -7.7% | 9.2% | 0.5811 | 1.0000 | não |
| v2|rac=nenhum|ix=h512|k=10/5|rep=2 | -7.7% | 9.2% | 0.5811 | 1.0000 | não |
