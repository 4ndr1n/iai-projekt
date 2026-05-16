# intro project to AI

# Docker
## build
```shell
docker build -t iai-auftrag .
```
## run

```
docker run -d --name iai-projekt iai-projekt
```
Results will be printet into logs.

# Results
| model |accuracy | macro_f1 | time_s |
| ----- | ------- | ----- | ---- |
| classical |0.9713004484304932 | 0.9317029845489487| 0.08747696876525879
| transformer | 0.45112107623318387 |0.4232758620689655 | 6.899314880371094
